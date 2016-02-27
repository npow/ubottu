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
sys.setrecursionlimit(10000)

class Model:
    def __init__(self,
                 data,
                 W,
                 max_seqlen=160,
                 hidden_size=100,
                 batch_size=50,
                 lr=0.001,
                 lr_decay=0.95,
                 fine_tune_W=True,
                 optimizer='adam',
                 forget_gate_bias=2,
                 filter_sizes=[3,4,5],
                 num_filters=100,
                 encoder='rnn',
                 penalize_emb_norm=False,
                 penalize_emb_drift=False,
                 penalize_activations=False,
                 emb_penalty=0.001,
                 act_penalty=500,
                 k=4,
                 n_recurrent_layers=1,
                 is_bidirectional=False,
                 **kwargs):
        embedding_size = W.shape[1]
        self.data = data
        self.max_seqlen = max_seqlen
        self.batch_size = batch_size
        self.fine_tune_W = fine_tune_W
        self.lr = lr
        self.lr_decay = lr_decay
        self.optimizer = optimizer
        self.emb_penalty = emb_penalty
        self.penalize_emb_norm = penalize_emb_norm
        self.penalize_emb_drift = penalize_emb_drift
        if penalize_emb_drift:
            self.orig_embeddings = theano.shared(W.copy(), name='orig_embeddings', borrow=True)

        index = T.iscalar()
        c = T.imatrix('c')
        y = T.ivector('y')
        c_mask = T.fmatrix('c_mask')
        r_mask = T.fmatrix('r_mask')
        embeddings = theano.shared(W, name='embeddings', borrow=True)
        zero_vec_tensor = T.fvector()
        self.zero_vec = np.zeros(embedding_size, dtype=theano.config.floatX)
        self.set_zero = theano.function([zero_vec_tensor], updates=[(embeddings, T.set_subtensor(embeddings[0,:], zero_vec_tensor))])

        c_input = embeddings[c.flatten()].reshape((c.shape[0], c.shape[1], embeddings.shape[1]))
        l_in = lasagne.layers.InputLayer(shape=(batch_size, max_seqlen, embedding_size))
        l_mask_in = lasagne.layers.InputLayer(shape=(batch_size, max_seqlen))

        if is_bidirectional:
            if encoder.find('lstm') > -1:
                prev_fwd, prev_bck = l_in, l_in
                for _ in xrange(n_recurrent_layers):
                    l_fwd = lasagne.layers.LSTMLayer(prev_fwd,
                                                     hidden_size,
                                                     mask_input=l_mask_in,
                                                     grad_clipping=10,
                                                     forgetgate=lasagne.layers.Gate(b=lasagne.init.Constant(forget_gate_bias)),
                                                     backwards=False,
                                                     learn_init=True,
                                                     peepholes=True)

                    l_bck = lasagne.layers.LSTMLayer(prev_bck,
                                                     hidden_size,
                                                     mask_input=l_mask_in,
                                                     grad_clipping=10,
                                                     forgetgate=lasagne.layers.Gate(b=lasagne.init.Constant(forget_gate_bias)),
                                                     backwards=True,
                                                     learn_init=True,
                                                     peepholes=True)
                    prev_fwd, prev_bck = l_fwd, l_bck
            else:
                prev_fwd, prev_bck = l_in, l_in
                for _ in xrange(n_recurrent_layers):
                    l_fwd = lasagne.layers.RecurrentLayer(prev_fwd,
                                                          hidden_size,
                                                          mask_input=l_mask_in,
                                                          nonlinearity=lasagne.nonlinearities.tanh,
                                                          W_hid_to_hid=lasagne.init.Orthogonal(),
                                                          W_in_to_hid=lasagne.init.Orthogonal(),
                                                          backwards=False,
                                                          learn_init=True
                                                          )

                    l_bck = lasagne.layers.RecurrentLayer(prev_bck,
                                                          hidden_size,
                                                          mask_input=l_mask_in,
                                                          nonlinearity=lasagne.nonlinearities.tanh,
                                                          W_hid_to_hid=lasagne.init.Orthogonal(),
                                                          W_in_to_hid=lasagne.init.Orthogonal(),
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
                                                           mask_input=l_mask_in,
                                                           grad_clipping=10,
                                                           forgetgate=lasagne.layers.Gate(b=lasagne.init.Constant(forget_gate_bias)),
                                                           backwards=False,
                                                           learn_init=True,
                                                           peepholes=True)
                    prev_fwd = l_recurrent
            elif encoder.find('gru') > -1:
                for _ in xrange(n_recurrent_layers):
                    l_recurrent = lasagne.layers.GRULayer(prev_fwd,
                                                          hidden_size,
                                                          mask_input=l_mask_in,
                                                          grad_clipping=10,
                                                          resetgate=lasagne.layers.Gate(b=lasagne.init.Constant(forget_gate_bias)),
                                                          backwards=False,
                                                          learn_init=True)
                    prev_fwd = l_recurrent
            else:
                for _ in xrange(n_recurrent_layers):
                    l_recurrent = lasagne.layers.RecurrentLayer(prev_fwd,
                                                                hidden_size,
                                                                mask_input=l_mask_in,
                                                                nonlinearity=lasagne.nonlinearities.tanh,
                                                                W_hid_to_hid=lasagne.init.Orthogonal(),
                                                                W_in_to_hid=lasagne.init.Orthogonal(),
                                                                backwards=False,
                                                                learn_init=True
                                                                )
                    prev_fwd = l_recurrent

        l_recurrent = lasagne.layers.SliceLayer(l_recurrent, indices=-1, axis=1)
        l_recurrent = lasagne.layers.ReshapeLayer(l_recurrent, (batch_size, hidden_size))
        l_out = lasagne.layers.DenseLayer(l_recurrent, 1, nonlinearity=lasagne.nonlinearities.sigmoid)
        l_out = lasagne.layers.ReshapeLayer(l_out, (batch_size, 1))
        o = lasagne.layers.helper.get_output(l_out, { l_in: c_input, l_mask_in: c_mask })
        o = T.clip(o, 1e-7, 1.0-1e-7)

        self.shared_data = {}
        for key in ['c']:
            self.shared_data[key] = theano.shared(np.zeros((batch_size, max_seqlen), dtype=np.int32))
        for key in ['c_mask']:
            self.shared_data[key] = theano.shared(np.zeros((batch_size, max_seqlen), dtype=theano.config.floatX))
        for key in ['y']:
            self.shared_data[key] = theano.shared(np.zeros((batch_size,), dtype=np.int32))

        self.probas = T.concatenate([(1-o).reshape((-1,1)), o.reshape((-1,1))], axis=1)
        self.pred = T.argmax(self.probas, axis=1)
        self.errors = T.sum(T.neq(self.pred, y))
        self.cost = T.nnet.binary_crossentropy(o, y).mean()

        if self.penalize_emb_norm:
            self.cost += self.emb_penalty * (embeddings ** 2).sum()

        if self.penalize_emb_drift:
            self.cost += self.emb_penalty * ((embeddings - self.orig_embeddings) ** 2).sum()

        if penalize_activations:
            self.cost += act_penalty * T.stack([((h_context[:,i] - h_context[:,i+1]) ** 2).sum(axis=1).mean() for i in xrange(max_seqlen-1)]).mean()
            self.cost += act_penalty * T.stack([((h_response[:,i] - h_response[:,i+1]) ** 2).sum(axis=1).mean() for i in xrange(max_seqlen-1)]).mean()

        self.l_out = l_out
        self.embeddings = embeddings
        self.c = c
        self.y = y
        self.c_mask = c_mask

        self.update_params()

    def update_params(self):
        params = lasagne.layers.get_all_params(self.l_out)
        if self.fine_tune_W:
            params += [self.embeddings]

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
            self.y: self.shared_data['y'],
            self.c_mask: self.shared_data['c_mask'],
        }
        self.train_model = theano.function([], self.cost, updates=updates, givens=givens, on_unused_input='warn')
        self.get_loss = theano.function([], self.errors, givens=givens, on_unused_input='warn')
        self.get_probas = theano.function([], self.probas, givens=givens, on_unused_input='warn')

    def get_batch(self, dataset_c, dataset_r, index, max_l):
        seqlen = np.zeros((self.batch_size,), dtype=np.int32)
        mask = np.zeros((self.batch_size,max_l), dtype=theano.config.floatX)
        batch = np.zeros((self.batch_size, max_l), dtype=np.int32)
        data_c = dataset_c[index*self.batch_size:(index+1)*self.batch_size]
        data_r = dataset_r[index*self.batch_size:(index+1)*self.batch_size]
        for i, (row_c, row_r) in enumerate(zip(data_c, data_r)):
            row_r = row_r[:max_l]
            row_c = row_c[:max_l-len(row_r)]
            row = row_c + row_r
            row = row[:max_l]
            batch[i,0:len(row)] = row
            seqlen[i] = len(row)-1
            mask[i,0:len(row)] = 1
        return batch, seqlen, mask

    def set_shared_variables(self, dataset, index):
        c, _, c_mask = self.get_batch(dataset['c'], dataset['r'], index, self.max_seqlen)
        y = np.array(dataset['y'][index*self.batch_size:(index+1)*self.batch_size], dtype=np.int32)
        self.shared_data['c'].set_value(c)
        self.shared_data['y'].set_value(y)
        self.shared_data['c_mask'].set_value(c_mask)

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
        test_probas = None
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
                    break
                self.update_params()
        return test_perf, test_probas

    def compute_recall_ks(self, probas):
      recall_k = {}
      for group_size in [2, 5, 10]:
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

def sort_by_len(dataset):
    c, r, y = dataset['c'], dataset['r'], dataset['y']
    indices = range(len(y))
    indices.sort(key=lambda i: len(c[i]))
    for k in ['c', 'r', 'y']:
        dataset[k] = np.array(dataset[k])[indices]

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

def main():
  parser = argparse.ArgumentParser()
  parser.register('type','bool',str2bool)
  parser.add_argument('--encoder', type=str, default='rnn', help='Encoder')
  parser.add_argument('--hidden_size', type=int, default=200, help='Hidden size')
  parser.add_argument('--fine_tune_W', type='bool', default=False, help='Whether to fine-tune W')
  parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
  parser.add_argument('--shuffle_batch', type='bool', default=False, help='Shuffle batch')
  parser.add_argument('--is_bidirectional', type='bool', default=False, help='Bidirectional RNN/LSTM')
  parser.add_argument('--n_epochs', type=int, default=100, help='Num epochs')
  parser.add_argument('--lr_decay', type=float, default=0.95, help='Learning rate decay')
  parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
  parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer')
  parser.add_argument('--forget_gate_bias', type=float, default=2.0, help='Forget gate bias')
  parser.add_argument('--max_seqlen', type=int, default=160, help='Max seqlen')
  parser.add_argument('--n_recurrent_layers', type=int, default=1, help='Num recurrent layers')
  parser.add_argument('--input_dir', type=str, default='.', help='Input dir')
  parser.add_argument('--save_model', type='bool', default=False, help='Whether to save the model')
  parser.add_argument('--model_fname', type=str, default='model.pkl', help='Model filename')
  parser.add_argument('--dataset_fname', type=str, default='dataset.pkl', help='Dataset filename')
  parser.add_argument('--W_fname', type=str, default='W.pkl', help='W filename')
  parser.add_argument('--sort_by_len', type='bool', default=False, help='Whether to sort contexts by length')
  parser.add_argument('--penalize_emb_norm', type='bool', default=False, help='Whether to penalize norm of embeddings')
  parser.add_argument('--penalize_emb_drift', type='bool', default=False, help='Whether to use re-embedding words penalty')
  parser.add_argument('--penalize_activations', type='bool', default=False, help='Whether to penalize activations')
  parser.add_argument('--emb_penalty', type=float, default=0.001, help='Embedding penalty')
  parser.add_argument('--act_penalty', type=float, default=500, help='Activation penalty')
  parser.add_argument('--seed', type=int, default=42, help='Random seed')
  args = parser.parse_args()
  print 'args:', args
  np.random.seed(args.seed)

  print "loading data...",
  train_data, val_data, test_data = cPickle.load(open('%s/%s' % (args.input_dir, args.dataset_fname), 'rb'))
  W, _ = cPickle.load(open('%s/%s' % (args.input_dir, args.W_fname), 'rb'))
  print "data loaded!"

  args.data = { 'train' : train_data, 'val': val_data, 'test': test_data }
  args.W = W.astype(theano.config.floatX)

  if args.sort_by_len:
      sort_by_len(data['train'])

  model = Model(**args.__dict__)
  _, test_probas = model.train(n_epochs=args.n_epochs, shuffle_batch=args.shuffle_batch)

  if args.save_model:
      cPickle.dump(model, open(args.model_fname, 'wb'))
      cPickle.dump(test_probas, open('probas_%s' % args.model_fname, 'wb'))

if __name__ == '__main__':
  main()
