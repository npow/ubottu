from __future__ import division
import argparse
import cPickle
import lasagne
import lasagne as nn
import numpy as np
import pyprind
import re
import sys
import theano
import theano.tensor as T
import time
from collections import defaultdict, OrderedDict
from theano.ifelse import ifelse
from theano.printing import Print as pp
from lasagne import nonlinearities, init, utils
from lasagne.layers import Layer, InputLayer, DenseLayer, helper
from rnn_em import RecurrentLayer
sys.setrecursionlimit(10000)
np.random.seed(42)

class Model(object):
    def __init__(self,
                 data,
                 U,
                 img_h=160,
                 img_w=300,
                 hidden_size=100,
                 batch_size=50,
                 lr=0.001,
                 lr_decay=0.95,
                 sqr_norm_lim=9,
                 fine_tune_W=True,
                 fine_tune_M=False,
                 optimizer='adam',
                 filter_sizes=[3,4,5],
                 num_filters=100,
                 encoder='rnn',
                 elemwise_sum=True,
                 n_recurrent_layers=1,
                 n_memory_slots=0,
                 is_bidirectional=False):
        self.data = data
        self.img_h = img_h
        self.batch_size = batch_size
        self.fine_tune_W = fine_tune_W
        self.fine_tune_M = fine_tune_M
        self.lr = lr
        self.lr_decay = lr_decay
        self.optimizer = optimizer
        self.sqr_norm_lim = sqr_norm_lim
        self.external_memory_size = (hidden_size, n_memory_slots) if n_memory_slots > 0 else None

        c = T.imatrix('c')
        r = T.imatrix('r')
        y = T.ivector('y')
        c_mask = T.fmatrix('c_mask')
        r_mask = T.fmatrix('r_mask')
        c_seqlen = T.ivector('c_seqlen')
        r_seqlen = T.ivector('r_seqlen')
        embeddings = theano.shared(U, name='embeddings', borrow=True)
        zero_vec_tensor = T.fvector()
        self.zero_vec = np.zeros(img_w, dtype=theano.config.floatX)
        self.set_zero = theano.function([zero_vec_tensor], updates=[(embeddings, T.set_subtensor(embeddings[0,:], zero_vec_tensor))])
        self.M = theano.shared(np.eye(hidden_size).astype(theano.config.floatX), borrow=True)

        c_input = embeddings[c.flatten()].reshape((batch_size, img_h, img_w))
        r_input = embeddings[r.flatten()].reshape((batch_size, img_h, img_w))

        l_in = lasagne.layers.InputLayer(shape=(batch_size, img_h, img_w))

        if is_bidirectional:
            if encoder.find('lstm') > -1:
                prev_fwd, prev_bck = l_in, l_in
                for _ in xrange(n_recurrent_layers):
                    l_fwd = lasagne.layers.LSTMLayer(prev_fwd,
                                                     hidden_size,
                                                     backwards=False,
                                                     learn_init=True,
                                                     peepholes=True)

                    l_bck = lasagne.layers.LSTMLayer(prev_bck,
                                                     hidden_size,
                                                     backwards=True,
                                                     learn_init=True,
                                                     peepholes=True)
                    prev_fwd, prev_bck = l_fwd, l_bck
            else:
                prev_fwd, prev_bck = l_in, l_in
                for _ in xrange(n_recurrent_layers):
                    l_fwd = RecurrentLayer(prev_fwd,
                                           hidden_size,
                                           nonlinearity=lasagne.nonlinearities.tanh,
                                           W_hid_to_hid=lasagne.init.Orthogonal(),
                                           W_in_to_hid=lasagne.init.Orthogonal(),
                                           external_memory_size=self.external_memory_size,
                                           hid_init=lasagne.init.Uniform(1.),
                                           backwards=False,
                                           learn_init=True
                                           )

                    l_bck = RecurrentLayer(prev_bck,
                                           hidden_size,
                                           nonlinearity=lasagne.nonlinearities.tanh,
                                           W_hid_to_hid=lasagne.init.Orthogonal(),
                                           W_in_to_hid=lasagne.init.Orthogonal(),
                                           external_memory_size=self.external_memory_size,
                                           hid_init=lasagne.init.Uniform(1.),
                                           backwards=True,
                                           learn_init=True
                                           )
                    prev_fwd, prev_bck = l_fwd, l_bck

            l_recurrent = lasagne.layers.ConcatLayer([l_fwd, l_bck])
        else:
            prev_fwd = l_in
            if encoder.find('lstm') > -1:
                for _ in xrange(n_recurrent_layers):
                    l_recurrent = lasagne.layers.LSTMLayer(prev_fwd,
                                                           hidden_size,
                                                           backwards=False,
                                                           learn_init=True,
                                                           peepholes=True)
                    prev_fwd = l_recurrent
            else:
                for _ in xrange(n_recurrent_layers):
                    l_recurrent = RecurrentLayer(prev_fwd,
                                                 hidden_size,
                                                 nonlinearity=lasagne.nonlinearities.tanh,
                                                 W_hid_to_hid=lasagne.init.Orthogonal(),
                                                 W_in_to_hid=lasagne.init.Orthogonal(),
                                                 external_memory_size=self.external_memory_size,
                                                 hid_init=lasagne.init.Uniform(1.),
                                                 backwards=False,
                                                 learn_init=True
                                                 )
                    prev_fwd = l_recurrent

        l_out = l_recurrent

        input_stacked = T.concatenate([c_input, r_input], axis=0)
        mask_stacked = T.concatenate([c_mask, r_mask], axis=0)
        e_context_response = lasagne.layers.helper.get_output(l_out, input_stacked, mask=mask_stacked, deterministic=False)
        e_context = e_context_response[:batch_size][T.arange(batch_size), c_seqlen].reshape((batch_size, hidden_size))
        e_response = e_context_response[batch_size:][T.arange(batch_size), r_seqlen].reshape((batch_size, hidden_size))

        dp = T.batched_dot(e_context, T.dot(e_response, self.M.T))
        #dp = pp('dp')(dp)
        o = T.nnet.sigmoid(dp)
        o = T.clip(o, 1e-7, 1.0-1e-7)

        self.shared_data = {}
        for key in ['c', 'r']:
            self.shared_data[key] = theano.shared(np.zeros((batch_size, img_h), dtype=np.int32))
        for key in ['c_mask', 'r_mask']:
            self.shared_data[key] = theano.shared(np.zeros((batch_size, img_h), dtype=theano.config.floatX))
        for key in ['y', 'c_seqlen', 'r_seqlen']:
            self.shared_data[key] = theano.shared(np.zeros((batch_size,), dtype=np.int32))

        self.probas = T.concatenate([(1-o).reshape((-1,1)), o.reshape((-1,1))], axis=1)
        self.pred = T.argmax(self.probas, axis=1)
        self.errors = T.sum(T.neq(self.pred, y))
        self.cost = T.nnet.binary_crossentropy(o, y).mean()
        self.l_out = l_out
        self.l_recurrent = l_recurrent
        self.embeddings = embeddings
        self.c = c
        self.r = r
        self.y = y
        self.c_seqlen = c_seqlen
        self.r_seqlen = r_seqlen
        self.c_mask = c_mask
        self.r_mask = r_mask

        self.update_params()

    def update_params(self):
        params = lasagne.layers.get_all_params(self.l_out)
        if self.fine_tune_W:
            params += [self.embeddings]
        if self.fine_tune_M:
            params += [self.M]

        total_params = sum([p.get_value().size for p in params])
        print "total_params: ", total_params

        if 'adam' == self.optimizer:
            updates = lasagne.updates.adam(self.cost, params, learning_rate=self.lr)
        elif 'adadelta' == self.optimizer:
            updates = lasagne.updates.adadelta(self.cost, params, learning_rate=1.0, rho=self.lr_decay)
        else:
            raise 'Unsupported optimizer: %s' % self.optimizer

        givens = {
            self.c: self.shared_data['c'],
            self.r: self.shared_data['r'],
            self.y: self.shared_data['y'],
            self.c_seqlen: self.shared_data['c_seqlen'],
            self.r_seqlen: self.shared_data['r_seqlen'],
            self.c_mask: self.shared_data['c_mask'],
            self.r_mask: self.shared_data['r_mask']
        }
        self.train_model = theano.function([], self.cost, updates=updates, givens=givens, on_unused_input='warn')
        self.get_loss = theano.function([], self.errors, givens=givens, on_unused_input='warn')
        self.get_probas = theano.function([], self.probas, givens=givens, on_unused_input='warn')

    def get_batch(self, dataset, index, max_l):
        seqlen = np.zeros((self.batch_size,), dtype=np.int32)
        mask = np.zeros((self.batch_size,max_l), dtype=theano.config.floatX)
        batch = np.zeros((self.batch_size, max_l), dtype=np.int32)
        data = dataset[index*self.batch_size:(index+1)*self.batch_size]
        for i,row in enumerate(data):
            row = row[:max_l]
            batch[i,0:len(row)] = row
            seqlen[i] = len(row)-1
            mask[i,0:len(row)] = 1
        return batch, seqlen, mask

    def set_shared_variables(self, dataset, index):
        c, c_seqlen, c_mask = self.get_batch(dataset['c'], index, self.img_h)
        r, r_seqlen, r_mask = self.get_batch(dataset['r'], index, self.img_h)
        y = np.array(dataset['y'][index*self.batch_size:(index+1)*self.batch_size], dtype=np.int32)
        self.shared_data['c'].set_value(c)
        self.shared_data['r'].set_value(r)
        self.shared_data['y'].set_value(y)
        self.shared_data['c_seqlen'].set_value(c_seqlen)
        self.shared_data['r_seqlen'].set_value(r_seqlen)
        self.shared_data['c_mask'].set_value(c_mask)
        self.shared_data['r_mask'].set_value(r_mask)

    def compute_loss(self, dataset, index):
        self.set_shared_variables(dataset, index)
        return self.get_loss()

    def compute_probas(self, dataset, index):
        self.set_shared_variables(dataset, index)
        return self.get_probas()[:,1]

    def train(self, n_epochs=100, shuffle_batch=False):
        epoch = 0
        best_val_perf = 0
        best_val_rk1 = 0
        test_perf = 0
        cost_epoch = 0

        n_train_batches = len(self.data['train']['y']) // self.batch_size
        n_val_batches = len(self.data['val']['y']) // self.batch_size
        n_test_batches = len(self.data['test']['y']) // self.batch_size

        while (epoch < n_epochs):
            epoch += 1
            indices = range(n_train_batches)
            if shuffle_batch:
                indices = np.random.permutation(indices)
            bar = pyprind.ProgBar(len(indices), monitor=True)
            total_cost = 0
            start_time = time.time()
            for minibatch_index in indices:
                self.set_shared_variables(self.data['train'], minibatch_index)
                cost_epoch = self.train_model()
                total_cost += cost_epoch
                self.set_zero(self.zero_vec)
                bar.update()
            end_time = time.time()
            print "cost: ", (total_cost / len(indices)), " took: %d(s)" % (end_time - start_time)
            train_losses = [self.compute_loss(self.data['train'], i) for i in xrange(n_train_batches)]
            train_perf = 1 - np.sum(train_losses) / len(self.data['train']['y'])
            val_losses = [self.compute_loss(self.data['val'], i) for i in xrange(n_val_batches)]
            val_perf = 1 - np.sum(val_losses) / len(self.data['val']['y'])
            print 'epoch %i, train_perf %f, val_perf %f' % (epoch, train_perf*100, val_perf*100)

            val_probas = np.concatenate([self.compute_probas(self.data['val'], i) for i in xrange(n_val_batches)])
            val_recall_k = self.compute_recall_ks(val_probas)

            if val_perf > best_val_perf or val_recall_k[10][1] > best_val_rk1:
                best_val_perf = val_perf
                best_val_rk1 = val_recall_k[10][1]
                test_losses = [self.compute_loss(self.data['test'], i) for i in xrange(n_test_batches)]
                test_perf = 1 - np.sum(test_losses) / len(self.data['test']['y'])
                print 'test_perf: %f' % (test_perf*100)
                test_probas = np.concatenate([self.compute_probas(self.data['test'], i) for i in xrange(n_test_batches)])
                self.compute_recall_ks(test_probas)
            else:
                if not self.fine_tune_W:
                    self.fine_tune_W = True # try fine-tuning W
                else:
                    if not self.fine_tune_M:
                        self.fine_tune_M = True # try fine-tuning M
                    else:
                        break
                self.update_params()
        return test_perf

    def compute_recall_ks(self, probas):
      recall_k = {}
      for group_size in [2, 10]:
          recall_k[group_size] = {}
          print 'group_size: %d' % group_size
          for k in [1, 2, 5]:
              if k < group_size:
                  recall_k[group_size][k] = self.recall(probas, k, group_size)
                  print 'recall@%d' % k, recall_k[group_size][k]
      return recall_k

    def recall(self, probas, k, group_size):
        test_size = 10
        n_batches = len(probas) // test_size
        n_correct = 0
        for i in xrange(n_batches):
            batch = np.array(probas[i*test_size:(i+1)*test_size])[:group_size]
            #p = np.random.permutation(len(batch))
            #indices = p[np.argpartition(batch[p], -k)[-k:]]
            indices = np.argpartition(batch, -k)[-k:]
            if 0 in indices:
                n_correct += 1
        return n_correct / (len(probas) / test_size)

def pad_to_batch_size(X, batch_size):
    n_seqs = len(X)
    n_batches_out = np.ceil(float(n_seqs) / batch_size)
    n_seqs_out = batch_size * n_batches_out

    to_pad = n_seqs % batch_size
    if to_pad > 0:
        X += X[:batch_size-to_pad]
    return X

def get_nrows(fname):
    with open(fname, 'rb') as f:
        nrows = 0
        for _ in f:
            nrows += 1
        return nrows

def load_pv_vecs(fname, ndims):
    nrows = get_nrows(fname)
    with open(fname, "rb") as f:
        X = np.zeros((nrows+1, ndims))
        for i,line in enumerate(f):
            L = line.strip().split()
            X[i+1] = np.array(L[1:], dtype='float32')
        return X

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

def main():
  parser = argparse.ArgumentParser()
  parser.register('type','bool',str2bool)
  parser.add_argument('--encoder', type=str, default='rnn', help='Encoder')
  parser.add_argument('--hidden_size', type=int, default=200, help='Hidden size')
  parser.add_argument('--fine_tune_W', type='bool', default=False, help='Whether to fine-tune W')
  parser.add_argument('--fine_tune_M', type='bool', default=False, help='Whether to fine-tune M')
  parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
  parser.add_argument('--shuffle_batch', type='bool', default=False, help='Shuffle batch')
  parser.add_argument('--is_bidirectional', type='bool', default=False, help='Bidirectional RNN/LSTM')
  parser.add_argument('--n_epochs', type=int, default=100, help='Num epochs')
  parser.add_argument('--lr_decay', type=float, default=0.95, help='Learning rate decay')
  parser.add_argument('--sqr_norm_lim', type=float, default=1, help='Squared norm limit')
  parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
  parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer')
  parser.add_argument('--suffix', type=str, default='', help='Suffix for pkl files')
  parser.add_argument('--use_pv', type='bool', default=False, help='Use PV')
  parser.add_argument('--pv_ndims', type=int, default=100, help='PV ndims')
  parser.add_argument('--max_seqlen', type=int, default=160, help='Max seqlen')
  parser.add_argument('--n_recurrent_layers', type=int, default=1, help='Num recurrent layers')
  parser.add_argument('--n_memory_slots', type=int, default=0, help='Num memory slots')
  parser.add_argument('--input_dir', type=str, default='.', help='Input dir')
  parser.add_argument('--save_model', type='bool', default=False, help='Whether to save the model')
  parser.add_argument('--load_model', type=str, default=False, help='Whether to load the model')
  parser.add_argument('--model_fname', type=str, default='model.pkl', help='Model filename')
  parser.add_argument('--dump_probas', type='bool', default=False, help='Dump test probabilities')
  args = parser.parse_args()
  print "args: ", args

  if args.load_model:
    model = cPickle.load(open(args.model_fname))
  else:
    print "loading data...",
    if args.use_pv:
        data = cPickle.load(open('../data/all_pv.pkl'))
        train_data = { 'c': data['c'][:1000000], 'r': data['r'][:1000000], 'y': data['y'][:1000000] }
        val_data = { 'c': data['c'][1000000:1356080], 'r': data['r'][1000000:1356080], 'y': data['y'][1000000:1356080] }
        test_data = { 'c': data['c'][1000000+356080:], 'r': data['r'][1000000+356080:], 'y': data['y'][1000000+356080:] }

        for key in ['c', 'r', 'y']:
            for dataset in [train_data, val_data]:
                dataset[key] = pad_to_batch_size(dataset[key], args.batch_size)

        W = load_pv_vecs('../data/pv_vectors_%dd.txt' % args.pv_ndims, args.pv_ndims)
        args.max_seqlen = 21
    else:
        train_data, val_data, test_data = cPickle.load(open('%s/dataset%s.pkl' % (args.input_dir, args.suffix), 'rb'))
        W, _ = cPickle.load(open('%s/W%s.pkl' % (args.input_dir, args.suffix), 'rb'))
    print "data loaded!"

    data = { 'train' : train_data, 'val': val_data, 'test': test_data }

    model = Model(data,
                  W.astype(theano.config.floatX),
                  img_h=args.max_seqlen,
                  img_w=W.shape[1],
                  hidden_size=args.hidden_size,
                  batch_size=args.batch_size,
                  lr=args.lr,
                  lr_decay=args.lr_decay,
                  sqr_norm_lim=args.sqr_norm_lim,
                  fine_tune_W=args.fine_tune_W,
                  fine_tune_M=args.fine_tune_M,
                  optimizer=args.optimizer,
                  encoder=args.encoder,
                  is_bidirectional=args.is_bidirectional,
                  n_recurrent_layers=args.n_recurrent_layers,
                  n_memory_slots=args.n_memory_slots)

    print model.train(n_epochs=args.n_epochs, shuffle_batch=args.shuffle_batch)

  if args.save_model:
    cPickle.dump(model, open(args.model_fname, 'wb'))
  
  if args.dump_probas:
    n_test_batches = len(model.data['test']['y']) // model.batch_size
    test_probas = np.concatenate([model.compute_probas(model.data['test'], i) for i in xrange(n_test_batches)])
    cPickle.dump(test_probas, open('test_probas.pkl', 'wb'))

if __name__ == '__main__':
  main()
