from __future__ import division
import cPickle
import lasagne
import numpy as np
import pyprind
import re
import sys
import theano
import theano.tensor as T
from collections import defaultdict, OrderedDict
from theano.ifelse import ifelse
from theano.printing import Print as pp

class GradClip(theano.compile.ViewOp):

    def __init__(self, clip_lower_bound, clip_upper_bound):
        self.clip_lower_bound = clip_lower_bound
        self.clip_upper_bound = clip_upper_bound
        assert(self.clip_upper_bound >= self.clip_lower_bound)

    def grad(self, args, g_outs):
        def pgrad(g_out):
            g_out = T.clip(g_out, self.clip_lower_bound, self.clip_upper_bound)
            g_out = ifelse(T.any(T.isnan(g_out)), T.ones_like(g_out)*0.00001, g_out)
            return g_out
        return [pgrad(g_out) for g_out in g_outs]

gradient_clipper = GradClip(-10.0, 10.0)
#T.opt.register_canonicalize(theano.gof.OpRemove(gradient_clipper), name='gradient_clipper')

def adam(loss, all_params, learning_rate=0.001, b1=0.9, b2=0.999, e=1e-8,
         gamma=1-1e-8):
    """
    ADAM update rules
    Default values are taken from [Kingma2014]

    References:
    [Kingma2014] Kingma, Diederik, and Jimmy Ba.
    "Adam: A Method for Stochastic Optimization."
    arXiv preprint arXiv:1412.6980 (2014).
    http://arxiv.org/pdf/1412.6980v4.pdf

    """
    updates = []
    all_grads = theano.grad(gradient_clipper(loss), all_params)
    alpha = learning_rate
    t = theano.shared(np.float32(1))
    b1_t = b1*gamma**(t-1)   #(Decay the first moment running average coefficient)
 
    for theta_previous, g in zip(all_params, all_grads):
        m_previous = theano.shared(np.zeros(theta_previous.get_value().shape,
                                            dtype=theano.config.floatX))
        v_previous = theano.shared(np.zeros(theta_previous.get_value().shape,
                                            dtype=theano.config.floatX))
 
        m = b1_t*m_previous + (1 - b1_t)*g                             # (Update biased first moment estimate)
        v = b2*v_previous + (1 - b2)*g**2                              # (Update biased second raw moment estimate)
        m_hat = m / (1-b1**t)                                          # (Compute bias-corrected first moment estimate)
        v_hat = v / (1-b2**t)                                          # (Compute bias-corrected second raw moment estimate)
        theta = theta_previous - (alpha * m_hat) / (T.sqrt(v_hat) + e) #(Update parameters)
 
        updates.append((m_previous, m))
        updates.append((v_previous, v))
        updates.append((theta_previous, theta) )
    updates.append((t, t + 1.))
    return updates

class RNN(object):
    def __init__(self,
                 data,
                 U,
                 img_w=300,
                 hidden_size=100,
                 batch_size=50,
                 lr_decay=0.95,
                 sqr_norm_lim=9,
                 non_static=True,
                 filter_sizes=[3,4,5],
                 num_filters=100,
                 use_conv=False,
                 use_lstm=True,
                 is_bidirectional=False):
        self.data = data
        self.batch_size = batch_size
        
        img_h = data['train']['r'].shape[1]
        
        index = T.iscalar()
        c = T.imatrix('c')
        r = T.imatrix('r')
        y = T.ivector('y')
        c_mask = T.bmatrix('c_mask')
        r_mask = T.bmatrix('r_mask')
        c_seqlen = T.ivector('c_seqlen')
        r_seqlen = T.ivector('r_seqlen')
        embeddings = theano.shared(U, name='embeddings', borrow=True)
        zero_vec_tensor = T.fvector()
        self.zero_vec = np.zeros(img_w, dtype=theano.config.floatX)
        self.set_zero = theano.function([zero_vec_tensor], updates=[(embeddings, T.set_subtensor(embeddings[0,:], zero_vec_tensor))])
        self.M = theano.shared(np.eye(hidden_size).astype(theano.config.floatX), borrow=True)
        
        c_input = embeddings[c.flatten()].reshape((c.shape[0], c.shape[1], embeddings.shape[1]))
        r_input = embeddings[r.flatten()].reshape((r.shape[0], r.shape[1], embeddings.shape[1]))                

        l_in = lasagne.layers.InputLayer(shape=(batch_size, img_h, img_w))
        
        if is_bidirectional:
            pass
        else:
            if use_lstm:
                l_recurrent = lasagne.layers.LSTMLayer(l_in,
                                                       hidden_size,
                                                       backwards=False,
                                                       learn_init=False,
                                                       peepholes=True)
            else:
                if False:
                    l_recurrent_in = lasagne.layers.InputLayer(shape=(batch_size, img_w))

                    l_input_to_hidden = lasagne.layers.DenseLayer(l_recurrent_in,
                                                                  hidden_size,
                                                                  nonlinearity=None,
                                                                  W=lasagne.init.Orthogonal())

                    l_recurrent_hid = lasagne.layers.InputLayer(shape=(batch_size, hidden_size))

                    l_hidden_to_hidden_1 = lasagne.layers.DenseLayer(l_recurrent_hid,
                                                                     hidden_size,
                                                                     nonlinearity=None,
                                                                     W=lasagne.init.Orthogonal())

                    """
                    l_hidden_to_hidden_2 = lasagne.layers.DenseLayer(l_hidden_to_hidden_1,
                                                                     hidden_size, nonlinearity=None)
                    """

                    l_recurrent = lasagne.layers.RecurrentLayer(l_in,
                                                                l_input_to_hidden,
                                                                l_hidden_to_hidden_1,
                                                                nonlinearity=lasagne.nonlinearities.tanh,
                                                                learn_init=True,
                                                                backwards=False
                                                                )
                else:
                    """
                    l_fwd = lasagne.layers.RecurrentLayer(l_in,
                                                          hidden_size,
                                                          nonlinearity=lasagne.nonlinearities.tanh,
                                                          W_hid_to_hid=lasagne.init.Orthogonal(),
                                                          W_in_to_hid=lasagne.init.Orthogonal(),
                                                          backwards=False,
                                                          learn_init=True
                                                          )
                    
                    l_bck = lasagne.layers.RecurrentLayer(l_in,
                                                          hidden_size,
                                                          nonlinearity=lasagne.nonlinearities.tanh,
                                                          W_hid_to_hid=lasagne.init.Orthogonal(),
                                                          W_in_to_hid=lasagne.init.Orthogonal(),
                                                          backwards=True,
                                                          learn_init=True
                                                          )
                    
                    l_recurrent = lasagne.layers.ElemwiseSumLayer([l_fwd, l_bck])  
                    """
                    
                    l_recurrent = lasagne.layers.RecurrentLayer(l_in,
                                                          hidden_size,
                                                          nonlinearity=lasagne.nonlinearities.tanh,
                                                          W_hid_to_hid=lasagne.init.Orthogonal(),
                                                          W_in_to_hid=lasagne.init.Orthogonal(),
                                                          backwards=False,
                                                          learn_init=True
                                                          )                    
        
        if use_conv:
            l_recurrent = lasagne.layers.ReshapeLayer(l_recurrent, (batch_size, 1, img_h, hidden_size))
            #l_recurrent = lasagne.layers.DropoutLayer(l_recurrent, p=0.5)
            conv_layers = []
            for filter_size in filter_sizes:
                conv_layer = lasagne.layers.Conv2DLayer(
                        l_recurrent,
                        num_filters=num_filters,
                        filter_size=(filter_size, hidden_size),
                        strides=(1,1),
                        nonlinearity=lasagne.nonlinearities.rectify,
                        border_mode='valid'
                        )
                pool_layer = lasagne.layers.MaxPool2DLayer(
                        conv_layer,
                        ds=(img_h-filter_size+1, 1)
                        )
                conv_layers.append(pool_layer)

            #hidden_size = len(conv_layers) * num_filters
            #hidden_size = num_filters
            l_hidden1 = lasagne.layers.ConcatLayer(conv_layers)
            l_hidden2 = lasagne.layers.DenseLayer(l_hidden1, num_units=hidden_size, nonlinearity=lasagne.nonlinearities.tanh)
            l_out = l_hidden2
        else:
            l_out = l_recurrent
        
        if use_conv:
            e_context = l_out.get_output(c_input, deterministic=False)
            e_response = l_out.get_output(r_input, deterministic=False)
        else:         
            e_context = l_out.get_output(c_input, c_mask, deterministic=False)[T.arange(batch_size), c_seqlen].reshape((c.shape[0], hidden_size))   
            e_response = l_out.get_output(r_input, r_mask, deterministic=False)[T.arange(batch_size), r_seqlen].reshape((r.shape[0], hidden_size))
            
        dp = T.batched_dot(e_context, T.dot(e_response, self.M.T))
        #dp = pp('dp')(dp)
        o = T.nnet.sigmoid(dp)
        o = T.clip(o, 1e-7, 1.0-1e-7)

        self.train_set_c = theano.shared(data['train']['c'], borrow=True)
        self.train_set_r = theano.shared(data['train']['r'], borrow=True)
        self.train_set_y = theano.shared(data['train']['y'], borrow=True)
        self.train_set_c_mask = theano.shared(data['train']['c_mask'], borrow=True)
        self.train_set_r_mask = theano.shared(data['train']['r_mask'], borrow=True)
        self.train_set_c_seqlen = theano.shared(data['train']['c_seqlen'], borrow=True)
        self.train_set_r_seqlen = theano.shared(data['train']['r_seqlen'], borrow=True)        
        
        self.val_set_c = theano.shared(data['val']['c'], borrow=True)
        self.val_set_r = theano.shared(data['val']['r'], borrow=True)
        self.val_set_y = theano.shared(data['val']['y'], borrow=True)
        self.val_set_c_mask = theano.shared(data['val']['c_mask'], borrow=True)
        self.val_set_r_mask = theano.shared(data['val']['r_mask'], borrow=True)
        self.val_set_c_seqlen = theano.shared(data['val']['c_seqlen'], borrow=True)
        self.val_set_r_seqlen = theano.shared(data['val']['r_seqlen'], borrow=True)
        
        self.test_set_c = theano.shared(data['test']['c'], borrow=True)
        self.test_set_r = theano.shared(data['test']['r'], borrow=True)
        self.test_set_y = theano.shared(data['test']['y'], borrow=True)
        self.test_set_c_mask = theano.shared(data['test']['c_mask'], borrow=True)
        self.test_set_r_mask = theano.shared(data['test']['r_mask'], borrow=True)
        self.test_set_c_seqlen = theano.shared(data['test']['c_seqlen'], borrow=True)
        self.test_set_r_seqlen = theano.shared(data['test']['r_seqlen'], borrow=True)          
        
        probas = T.concatenate([(1-o).reshape((-1,1)), o.reshape((-1,1))], axis=1)
        pred = T.argmax(probas, axis=1)
        errors = T.sum(T.neq(pred, y))    

        cost = T.nnet.binary_crossentropy(o, y).mean()
        params = lasagne.layers.get_all_params(l_out) + [self.M]
        if non_static:
            params += [embeddings]
            
        total_params = sum([p.get_value().size for p in params])
        print "total_params: ", total_params
#        updates = lasagne.updates.adadelta(cost, params, learning_rate=1.0, rho=lr_decay)
#        updates = sgd_updates_adadelta(cost, params, lr_decay, 1e-6, sqr_norm_lim)
#        updates = lasagne.updates.nesterov_momentum(cost, params, learning_rate=0.1)
        updates = adam(cost, params)
                        
        self.train_model = theano.function([index], cost, updates=updates,
              givens={
                c: self.train_set_c[index*batch_size:(index+1)*batch_size],
                r: self.train_set_r[index*batch_size:(index+1)*batch_size],
                y: self.train_set_y[index*batch_size:(index+1)*batch_size],
                c_mask: self.train_set_c_mask[index*batch_size:(index+1)*batch_size],
                r_mask: self.train_set_r_mask[index*batch_size:(index+1)*batch_size],
                c_seqlen: self.train_set_c_seqlen[index*batch_size:(index+1)*batch_size],
                r_seqlen: self.train_set_r_seqlen[index*batch_size:(index+1)*batch_size],
              },
              on_unused_input='warn')         

        self.train_loss = theano.function([index], errors,
                 givens={
                    c: self.train_set_c[index * batch_size: (index + 1) * batch_size],
                    r: self.train_set_r[index * batch_size: (index + 1) * batch_size],
                    y: self.train_set_y[index * batch_size: (index + 1) * batch_size],
                    c_mask: self.train_set_c_mask[index * batch_size: (index + 1) * batch_size],
                    r_mask: self.train_set_r_mask[index * batch_size: (index + 1) * batch_size],
                    c_seqlen: self.train_set_c_seqlen[index * batch_size: (index + 1) * batch_size],
                    r_seqlen: self.train_set_r_seqlen[index * batch_size: (index + 1) * batch_size],
                 },
                 on_unused_input='warn')
        
        self.val_loss = theano.function([index], errors,
             givens={
                c: self.val_set_c[index * batch_size: (index + 1) * batch_size],
                r: self.val_set_r[index * batch_size: (index + 1) * batch_size],
                y: self.val_set_y[index * batch_size: (index + 1) * batch_size],
                c_mask: self.val_set_c_mask[index * batch_size: (index + 1) * batch_size],
                r_mask: self.val_set_r_mask[index * batch_size: (index + 1) * batch_size],
                c_seqlen: self.val_set_c_seqlen[index * batch_size: (index + 1) * batch_size],
                r_seqlen: self.val_set_r_seqlen[index * batch_size: (index + 1) * batch_size],
             },
             on_unused_input='warn')

        self.test_loss = theano.function([index], errors,
             givens={
                c: self.test_set_c[index * batch_size: (index + 1) * batch_size],
                r: self.test_set_r[index * batch_size: (index + 1) * batch_size],
                y: self.test_set_y[index * batch_size: (index + 1) * batch_size],
                c_mask: self.test_set_c_mask[index * batch_size: (index + 1) * batch_size],
                r_mask: self.test_set_r_mask[index * batch_size: (index + 1) * batch_size],
                c_seqlen: self.test_set_c_seqlen[index * batch_size: (index + 1) * batch_size],
                r_seqlen: self.test_set_r_seqlen[index * batch_size: (index + 1) * batch_size],
             },
             on_unused_input='warn')

    def train(self, n_epochs=100, shuffle_batch=True):
        epoch = 0
        best_val_perf = 0
        val_perf = 0
        test_perf = 0
        cost_epoch = 0
        
        n_train_batches = self.data['train']['y'].shape[0] // self.batch_size
        n_val_batches = self.data['val']['y'].shape[0] // self.batch_size
        n_test_batches = self.data['test']['y'].shape[0] // self.batch_size

        while (epoch < n_epochs):
            epoch += 1
            indices = range(n_train_batches)
            if shuffle_batch:
                indices = np.random.permutation(indices)
            bar = pyprind.ProgBar(len(indices), monitor=True)
            total_cost = 0
            for minibatch_index in indices:
                cost_epoch = self.train_model(minibatch_index)
                total_cost += cost_epoch
                self.set_zero(self.zero_vec)
                bar.update()
            print "cost: ", (total_cost / len(indices))
            train_losses = [self.train_loss(i) for i in xrange(n_train_batches)]
            train_perf = 1 - np.sum(train_losses) / self.data['train']['y'].shape[0]
            val_losses = [self.val_loss(i) for i in xrange(n_val_batches)]
            val_perf = 1 - np.sum(val_losses) / self.data['val']['y'].shape[0]
            print 'epoch %i, train_perf %f, val_perf %f' % (epoch, train_perf*100, val_perf*100)
            if val_perf >= best_val_perf:
                best_val_perf = val_perf
                test_losses = [self.test_loss(i) for i in xrange(n_test_batches)]
                test_perf = 1 - np.sum(test_losses) / self.data['test']['y'].shape[0]
                print 'test_perf %f' % (test_perf*100)
        return test_perf

def as_floatX(variable):
    if isinstance(variable, float):
        return np.cast[theano.config.floatX](variable)

    if isinstance(variable, np.ndarray):
        return np.cast[theano.config.floatX](variable)
    return theano.tensor.cast(variable, theano.config.floatX)

def sgd_updates_adadelta(cost, params, rho=0.95, epsilon=1e-6, norm_lim=9, word_vec_name='embeddings'):
    updates = OrderedDict({})
    exp_sqr_grads = OrderedDict({})
    exp_sqr_ups = OrderedDict({})
    gparams = []
    for param in params:
        empty = np.zeros_like(param.get_value())
        exp_sqr_grads[param] = theano.shared(value=as_floatX(empty),name="exp_grad_%s" % param.name)
        gp = T.grad(cost, param)
        exp_sqr_ups[param] = theano.shared(value=as_floatX(empty), name="exp_grad_%s" % param.name)
        gparams.append(gp)
    for param, gp in zip(params, gparams):
        exp_sg = exp_sqr_grads[param]
        exp_su = exp_sqr_ups[param]
        up_exp_sg = rho * exp_sg + (1 - rho) * T.sqr(gp)
        updates[exp_sg] = up_exp_sg
        step =  -(T.sqrt(exp_su + epsilon) / T.sqrt(up_exp_sg + epsilon)) * gp
        updates[exp_su] = rho * exp_su + (1 - rho) * T.sqr(step)
        stepped_param = param + step
        if (param.get_value(borrow=True).ndim == 2) and (param.name != word_vec_name):
            col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
            desired_norms = T.clip(col_norms, 0, T.sqrt(norm_lim))
            scale = desired_norms / (1e-7 + col_norms)
            updates[param] = stepped_param * scale
        else:
            updates[param] = stepped_param      
    return updates 
 
def get_idx_from_sent(sent, word_idx_map, max_l, k):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    words = sent.split()
    for word in words[:max_l]:
        if word in word_idx_map:
            x.append(word_idx_map[word])
    while len(x) < max_l:
        x.append(0)
    mask = np.zeros(max_l, dtype=np.bool)
    mask[:len(words)] = 1
    return x, mask, len(words) if len(words) < max_l-1 else max_l-1

def make_idx_data(dataset, word_idx_map, max_l=152, k=300):
    """
    Transforms sentences into a 2-d matrix.
    """
    for i in xrange(len(dataset['y'])):
        dataset['c'][i], dataset['c_mask'][i], dataset['c_seqlen'][i] = get_idx_from_sent(dataset['c'][i], word_idx_map, max_l, k)
        dataset['r'][i], dataset['r_mask'][i], dataset['r_seqlen'][i] = get_idx_from_sent(dataset['r'][i], word_idx_map, max_l, k)
    for col in ['c', 'r', 'y', 'c_seqlen', 'r_seqlen']:
        dataset[col] = np.array(dataset[col], dtype=np.int32)
    for col in ['c_mask', 'r_mask']:
        dataset[col] = np.array(dataset[col], dtype=np.int8)

def pad_to_batch_size(X, batch_size):
    n_seqs = X.shape[0]
    seq_length = X.shape[1] if X.ndim > 1 else None
    n_batches_out = np.ceil(float(n_seqs) / batch_size)
    n_seqs_out = batch_size * n_batches_out

    if X.ndim > 1:
        X_out = np.zeros((n_seqs_out, seq_length), dtype=X.dtype)
    else:
        X_out = np.zeros((n_seqs_out), dtype=X.dtype)        
    X_out[:n_seqs, ] = X
    to_pad = n_seqs % batch_size
    if to_pad > 0:
        X_out[n_seqs:] = X[:batch_size-to_pad]
    
    return X_out

print "loading data...",
data = cPickle.load(open('data.pkl', 'rb'))
train_data, val_data, test_data, W, W2, word_idx_map, vocab = data
print "data loaded!"

for key in ['c_mask', 'r_mask', 'c_seqlen', 'r_seqlen']:
    for dataset in [train_data, val_data, test_data]:
        dataset[key] = [0] * len(dataset['y'])

make_idx_data(train_data, word_idx_map)
make_idx_data(val_data, word_idx_map)
make_idx_data(test_data, word_idx_map)

BATCH_SIZE = 256

for key in ['c', 'r', 'y', 'c_mask', 'r_mask', 'c_seqlen', 'r_seqlen']:
    print key
    for dataset in [train_data, val_data, test_data]:
        dataset[key] = pad_to_batch_size(dataset[key], BATCH_SIZE)
        print dataset[key].shape

data = { 'train' : train_data, 'val': val_data, 'test': test_data }

rnn = RNN(data,
          W.astype(theano.config.floatX),
          img_w=300,
          hidden_size=100,
          batch_size=BATCH_SIZE,
          lr_decay=0.95,
          sqr_norm_lim=1,
          non_static=True,
          use_lstm=True,
          use_conv=True)
print rnn.train(n_epochs=100, shuffle_batch=True)
