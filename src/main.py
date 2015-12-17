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
np.random.seed(42)

def split_utterances(utterances, eos_idx):
    return [[int(yy) for yy in y.split()] for y in ' '.join([str(x) for x in utterances]).split(eos_idx)]

class Model(object):
    def __init__(self,
                 data,
                 U,
                 max_seqlen=20,
                 max_sentlen=50,
                 embedding_size=300,
                 hidden_size=100,
                 batch_size=50,
                 lr=0.001,
                 lr_decay=0.95,
                 fine_tune_W=True,
                 fine_tune_M=False,
                 optimizer='adam',
                 encoder='rnn',
                 eos_idx='63346',
                 n_recurrent_layers=1,
                 is_bidirectional=False):
        self.data = data
        self.max_seqlen = max_seqlen
        self.max_sentlen = max_sentlen
        self.batch_size = batch_size
        self.fine_tune_W = fine_tune_W
        self.fine_tune_M = fine_tune_M
        self.lr = lr
        self.lr_decay = lr_decay
        self.optimizer = optimizer
        self.eos_idx = eos_idx

        index = T.iscalar()
        c = T.itensor3('c')
        r = T.itensor3('r')
        y = T.ivector('y')
        c_mask = T.fmatrix('c_mask')
        r_mask = T.fmatrix('r_mask')
        c_seqlen = T.ivector('c_seqlen')
        r_seqlen = T.ivector('r_seqlen')
        c_w_mask = T.itensor3('c_w_mask')
        r_w_mask = T.itensor3('r_w_mask')
        c_w_seqlen = T.imatrix('c_w_seqlen')
        r_w_seqlen = T.imatrix('r_w_seqlen')
        embeddings = theano.shared(U, name='embeddings', borrow=True)
        zero_vec_tensor = T.fvector()
        self.zero_vec = np.zeros(embedding_size, dtype=theano.config.floatX)
        self.set_zero = theano.function([zero_vec_tensor], updates=[(embeddings, T.set_subtensor(embeddings[0,:], zero_vec_tensor))])
        self.M = theano.shared(np.eye(hidden_size).astype(theano.config.floatX), borrow=True)

        c_input = embeddings[c.flatten()].reshape((c.shape[0], c.shape[1]*c.shape[2], embeddings.shape[1]))
        r_input = embeddings[r.flatten()].reshape((r.shape[0], r.shape[1]*r.shape[2], embeddings.shape[1]))

        l_in = lasagne.layers.InputLayer(shape=(batch_size, max_seqlen*max_sentlen, embedding_size))
        l_w_encoder = lasagne.layers.RecurrentLayer(l_in,
                                                    hidden_size,
                                                    nonlinearity=lasagne.nonlinearities.tanh,
                                                    W_hid_to_hid=lasagne.init.Orthogonal(),
                                                    W_in_to_hid=lasagne.init.Orthogonal(),
                                                    backwards=False,
                                                    learn_init=True
                                                    )

        e_w_context = lasagne.layers.helper.get_output(l_w_encoder, c_input, mask=c_w_mask, deterministic=False)
        e_w_context = e_w_context.reshape((c.shape[0], max_seqlen, max_sentlen, hidden_size))
        e_w_context = e_w_context[T.arange(batch_size), :, c_w_seqlen[:, 1]].reshape((c.shape[0], c.shape[1], hidden_size))

        e_w_response = lasagne.layers.helper.get_output(l_w_encoder, r_input, mask=r_w_mask, deterministic=False)
        e_w_response = e_w_response.reshape((r.shape[0], max_seqlen, max_sentlen, hidden_size))
        e_w_response = e_w_response[T.arange(batch_size), :, r_w_seqlen[:, 1]].reshape((r.shape[0], r.shape[1], hidden_size))

        l_s_in = lasagne.layers.InputLayer(shape=(batch_size, max_seqlen, hidden_size))
        l_s_encoder = lasagne.layers.RecurrentLayer(l_s_in,
                                                    hidden_size,
                                                    nonlinearity=lasagne.nonlinearities.tanh,
                                                    W_hid_to_hid=lasagne.init.Orthogonal(),
                                                    W_in_to_hid=lasagne.init.Orthogonal(),
                                                    backwards=False,
                                                    learn_init=True
                                                    )

        e_context = lasagne.layers.helper.get_output(l_s_encoder, e_w_context, mask=c_mask, deterministic=False)[T.arange(batch_size), c_seqlen].reshape((c.shape[0], hidden_size))
        e_response = lasagne.layers.helper.get_output(l_s_encoder, e_w_response, mask=r_mask, deterministic=False)[T.arange(batch_size), r_seqlen].reshape((r.shape[0], hidden_size))

        dp = T.batched_dot(e_context, T.dot(e_response, self.M.T))
        #dp = pp('dp')(dp)
        o = T.nnet.sigmoid(dp)
        o = T.clip(o, 1e-7, 1.0-1e-7)

        self.shared_data = {}
        for key in ['c', 'r', 'c_w_mask', 'r_w_mask']:
            self.shared_data[key] = theano.shared(np.zeros((batch_size, max_seqlen, max_sentlen), dtype=np.int32))
        for key in ['c_mask', 'r_mask']:
            self.shared_data[key] = theano.shared(np.zeros((batch_size, max_seqlen), dtype=theano.config.floatX))
        for key in ['c_w_seqlen', 'r_w_seqlen']:
            self.shared_data[key] = theano.shared(np.zeros((batch_size, max_seqlen), dtype=np.int32))
        for key in ['y', 'c_seqlen', 'r_seqlen']:
            self.shared_data[key] = theano.shared(np.zeros((batch_size,), dtype=np.int32))

        self.probas = T.concatenate([(1-o).reshape((-1,1)), o.reshape((-1,1))], axis=1)
        self.pred = T.argmax(self.probas, axis=1)
        self.errors = T.sum(T.neq(self.pred, y))
        self.cost = T.nnet.binary_crossentropy(o, y).mean()
        self.l_w_encoder = l_w_encoder
        self.l_s_encoder = l_s_encoder
        self.embeddings = embeddings
        self.c = c
        self.r = r
        self.y = y
        self.c_seqlen = c_seqlen
        self.r_seqlen = r_seqlen
        self.c_mask = c_mask
        self.r_mask = r_mask
        self.c_w_seqlen = c_w_seqlen
        self.r_w_seqlen = r_w_seqlen
        self.c_w_mask = c_w_mask
        self.r_w_mask = r_w_mask

        self.update_params()

    def update_params(self):
        params = lasagne.layers.get_all_params(self.l_w_encoder)
        params += lasagne.layers.get_all_params(self.l_s_encoder)
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
            self.r_mask: self.shared_data['r_mask'],
            self.c_w_seqlen: self.shared_data['c_w_seqlen'],
            self.r_w_seqlen: self.shared_data['r_w_seqlen'],
            self.c_w_mask: self.shared_data['c_w_mask'],
            self.r_w_mask: self.shared_data['r_w_mask'],
        }
        self.train_model = theano.function([], self.cost, updates=updates, givens=givens, on_unused_input='warn')
        self.get_loss = theano.function([], self.errors, givens=givens, on_unused_input='warn')
        self.get_probas = theano.function([], self.probas, givens=givens, on_unused_input='warn')

    def get_batch(self, dataset, index, max_seqlen, max_sentlen):
        w_seqlen = np.zeros((self.batch_size, max_seqlen,), dtype=np.int32)
        w_mask = np.zeros((self.batch_size, max_seqlen, max_sentlen), dtype=np.int32)
        seqlen = np.zeros((self.batch_size,), dtype=np.int32)
        mask = np.zeros((self.batch_size, max_seqlen), dtype=theano.config.floatX)
        batch = np.zeros((self.batch_size, max_seqlen, max_sentlen), dtype=np.int32)
        data = dataset[index*self.batch_size:(index+1)*self.batch_size]
        for i, row in enumerate(data):
            utterances = split_utterances(row, self.eos_idx)[:max_seqlen]
            for ii, utterance in enumerate(utterances):
                word_indices = utterance[:max_sentlen]
                w_seqlen[i, ii] = len(word_indices) - 1
                w_mask[i, ii, :len(word_indices)] = 1
                word_indices += [0] * (max_sentlen - len(word_indices))
                batch[i, ii, :] = word_indices

            seqlen[i] = len(utterances) - 1
            mask[i, 0:len(utterances)] = 1
        return batch, seqlen, mask, w_seqlen, w_mask

    def set_shared_variables(self, dataset, index):
        c, c_seqlen, c_mask, c_w_seqlen, c_w_mask = self.get_batch(dataset['c'], index, self.max_seqlen, self.max_sentlen)
        r, r_seqlen, r_mask, r_w_seqlen, r_w_mask = self.get_batch(dataset['r'], index, self.max_seqlen, self.max_sentlen)
        y = np.array(dataset['y'][index*self.batch_size:(index+1)*self.batch_size], dtype=np.int32)
        self.shared_data['c'].set_value(c)
        self.shared_data['r'].set_value(r)
        self.shared_data['y'].set_value(y)
        self.shared_data['c_seqlen'].set_value(c_seqlen)
        self.shared_data['r_seqlen'].set_value(r_seqlen)
        self.shared_data['c_mask'].set_value(c_mask)
        self.shared_data['r_mask'].set_value(r_mask)
        self.shared_data['c_w_seqlen'].set_value(c_w_seqlen)
        self.shared_data['r_w_seqlen'].set_value(r_w_seqlen)
        self.shared_data['c_w_mask'].set_value(c_w_mask)
        self.shared_data['r_w_mask'].set_value(r_w_mask)

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
  parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
  parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer')
  parser.add_argument('--suffix', type=str, default='', help='Suffix for pkl files')
  parser.add_argument('--max_seqlen', type=int, default=20, help='Max seqlen')
  parser.add_argument('--max_sentlen', type=int, default=50, help='Max sentlen')
  parser.add_argument('--n_recurrent_layers', type=int, default=1, help='Num recurrent layers')
  parser.add_argument('--input_dir', type=str, default='.', help='Input dir')
  parser.add_argument('--save_model', type='bool', default=False, help='Whether to save the model')
  parser.add_argument('--model_fname', type=str, default='model.pkl', help='Model filename')
  args = parser.parse_args()
  print "args: ", args

  print "loading data...",
  train_data, val_data, test_data = cPickle.load(open('%s/dataset%s.pkl' % (args.input_dir, args.suffix), 'rb'))
  W, _ = cPickle.load(open('%s/W%s.pkl' % (args.input_dir, args.suffix), 'rb'))
  print "data loaded!"

  data = { 'train' : train_data, 'val': val_data, 'test': test_data }

  model = Model(data,
                W.astype(theano.config.floatX),
                max_seqlen=args.max_seqlen,
                max_sentlen=args.max_sentlen,
                embedding_size=W.shape[1],
                hidden_size=args.hidden_size,
                batch_size=args.batch_size,
                lr=args.lr,
                lr_decay=args.lr_decay,
                fine_tune_W=args.fine_tune_W,
                fine_tune_M=args.fine_tune_M,
                optimizer=args.optimizer,
                encoder=args.encoder,
                is_bidirectional=args.is_bidirectional,
                n_recurrent_layers=args.n_recurrent_layers)

  print model.train(n_epochs=args.n_epochs, shuffle_batch=args.shuffle_batch)
  if args.save_model:
      cPickle.dump(model, open(args.model_fname, 'wb'))

if __name__ == '__main__':
  main()
