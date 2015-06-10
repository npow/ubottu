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

MAX_LEN = 160
BATCH_SIZE = 256

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

class MultiplicativeGatingLayer(nn.layers.MergeLayer):
    """
    Generic layer that combines its 3 inputs t, h1, h2 as follows:
    y = t * h1 + (1 - t) * h2
    """
    def __init__(self, gate, input1, input2, **kwargs):
        incomings = [gate, input1, input2]
        super(MultiplicativeGatingLayer, self).__init__(incomings, **kwargs)
        assert gate.output_shape == input1.output_shape == input2.output_shape

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, inputs, **kwargs):
        return inputs[0] * inputs[1] + (1 - inputs[0]) * inputs[2]

def highway_dense(incoming, Wh=nn.init.Orthogonal(), bh=nn.init.Constant(0.0),
                  Wt=nn.init.Orthogonal(), bt=nn.init.Constant(-4.0),
                  nonlinearity=nn.nonlinearities.rectify, **kwargs):
    num_inputs = int(np.prod(incoming.output_shape[1:]))
    # regular layer
    l_h = nn.layers.DenseLayer(incoming, num_units=num_inputs, W=Wh, b=bh,
                               nonlinearity=nonlinearity)
    # gate layer
    l_t = nn.layers.DenseLayer(incoming, num_units=num_inputs, W=Wt, b=bt,
                               nonlinearity=T.nnet.sigmoid)

    return MultiplicativeGatingLayer(gate=l_t, input1=l_h, input2=incoming)

class CustomRecurrentLayer(Layer):
    '''
    A layer which implements a recurrent connection.

    Expects inputs of shape
        (n_batch, n_time_steps, n_features_1, n_features_2, ...)
    '''
    def __init__(self, input_layer, input_to_hidden, hidden_to_hidden,
                 nonlinearity=nonlinearities.rectify,
                 hid_init=init.Constant(0.), backwards=False,
                 learn_init=False, gradient_steps=-1):
        '''
        Create a recurrent layer.

        :parameters:
            - input_layer : nntools.layers.Layer
                Input to the recurrent layer
            - input_to_hidden : nntools.layers.Layer
                Layer which connects input to the hidden state
            - hidden_to_hidden : nntools.layers.Layer
                Layer which connects the previous hidden state to the new state
            - nonlinearity : function or theano.tensor.elemwise.Elemwise
                Nonlinearity to apply when computing new state
            - hid_init : function or np.ndarray or theano.shared
                Initial hidden state
            - backwards : boolean
                If True, process the sequence backwards
            - learn_init : boolean
                If True, initial hidden values are learned
            - gradient_steps : int
                Number of timesteps to include in backpropagated gradient
                If -1, backpropagate through the entire sequence
        '''
        super(CustomRecurrentLayer, self).__init__(input_layer)

        self.input_to_hidden = input_to_hidden
        self.hidden_to_hidden = hidden_to_hidden
        self.learn_init = learn_init
        self.backwards = backwards
        self.gradient_steps = gradient_steps

        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        # Get the batch size and number of units based on the expected output
        # of the input-to-hidden layer
        (n_batch, self.num_units) = self.input_to_hidden.get_output_shape()

        # Initialize hidden state
        self.hid_init = self.create_param(hid_init, (n_batch, self.num_units))

    def get_params(self):
        '''
        Get all parameters of this layer.

        :returns:
            - params : list of theano.shared
                List of all parameters
        '''
        params = (helper.get_all_params(self.input_to_hidden) +
                  helper.get_all_params(self.hidden_to_hidden))

        if self.learn_init:
            return params + self.get_init_params()
        else:
            return params

    def get_init_params(self):
        '''
        Get all initital parameters of this layer.
        :returns:
            - init_params : list of theano.shared
                List of all initial parameters
        '''
        return [self.hid_init]

    def get_bias_params(self):
        '''
        Get all bias parameters of this layer.

        :returns:
            - bias_params : list of theano.shared
                List of all bias parameters
        '''
        return (helper.get_all_bias_params(self.input_to_hidden) +
                helper.get_all_bias_params(self.hidden_to_hidden))

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1], self.num_units)

    def get_output_for(self, input, mask=None, *args, **kwargs):
        '''
        Compute this layer's output function given a symbolic input variable

        :parameters:
            - input : theano.TensorType
                Symbolic input variable
            - mask : theano.TensorType
                Theano variable denoting whether each time step in each
                sequence in the batch is part of the sequence or not.  If None,
                then it assumed that all sequences are of the same length.  If
                not all sequences are of the same length, then it must be
                supplied as a matrix of shape (n_batch, n_time_steps) where
                `mask[i, j] = 1` when `j <= (length of sequence i)` and
                `mask[i, j] = 0` when `j > (length of sequence i)`.

        :returns:
            - layer_output : theano.TensorType
                Symbolic output variable
        '''
        if input.ndim > 3:
            input = input.reshape((input.shape[0], input.shape[1],
                                   T.prod(input.shape[2:])))

        # Input should be provided as (n_batch, n_time_steps, n_features)
        # but scan requires the iterable dimension to be first
        # So, we need to dimshuffle to (n_time_steps, n_batch, n_features)
        input = input.dimshuffle(1, 0, 2)

        # Create single recurrent computation step function
        def step(layer_input, hid_previous):
            return self.nonlinearity(
                self.input_to_hidden.get_output(layer_input) +
                self.hidden_to_hidden.get_output(hid_previous))

        def step_masked(layer_input, mask, hid_previous):
            # If mask is 0, use previous state until mask = 1 is found.
            # This propagates the layer initial state when moving backwards
            # until the end of the sequence is found.
            hid = (step(layer_input, hid_previous)*mask
                   + hid_previous*(1 - mask))
            return [hid]

        if self.backwards and mask is not None:
            mask = mask.dimshuffle(1, 0, 'x')
            sequences = [input, mask]
            step_fun = step_masked
        else:
            sequences = input
            step_fun = step

        output = theano.scan(step_fun, sequences=sequences,
                             go_backwards=self.backwards,
                             outputs_info=[self.hid_init],
                             truncate_gradient=self.gradient_steps)[0]

        # Now, dimshuffle back to (n_batch, n_time_steps, n_features))
        output = output.dimshuffle(1, 0, 2)

        if self.backwards:
            output = output[:, ::-1, :]

        return output


class RecurrentLayer(CustomRecurrentLayer):
    '''
    A "vanilla" RNN layer, which has dense input-to-hidden and
    hidden-to-hidden connections.

    Expects inputs of shape
        (n_batch, n_time_steps, n_features_1, n_features_2, ...)
    '''
    def __init__(self, input_layer, num_units, W_in_to_hid=init.Uniform(),
                 W_hid_to_hid=init.Uniform(), b=init.Constant(0.),
                 nonlinearity=nonlinearities.rectify,
                 hid_init=init.Constant(0.), backwards=False,
                 learn_init=False, gradient_steps=-1):
        '''
        Create a recurrent layer.

        :parameters:
            - input_layer : nntools.layers.Layer
                Input to the recurrent layer
            - num_units : int
                Number of hidden units in the layer
            - W_in_to_hid : function or np.ndarray or theano.shared
                Initializer for input-to-hidden weight matrix
            - W_hid_to_hid : function or np.ndarray or theano.shared
                Initializer for hidden-to-hidden weight matrix
            - b : function or np.ndarray or theano.shared
                Initializer for bias vector
            - nonlinearity : function or theano.tensor.elemwise.Elemwise
                Nonlinearity to apply when computing new state
            - hid_init : function or np.ndarray or theano.shared
                Initial hidden state
            - backwards : boolean
                If True, process the sequence backwards
            - learn_init : boolean
                If True, initial hidden values are learned
            - gradient_steps : int
                Number of timesteps to include in backpropagated gradient
                If -1, backpropagate through the entire sequence
        '''

        input_shape = input_layer.get_output_shape()
        n_batch = input_shape[0]
        # We will be passing the input at each time step to the dense layer,
        # so we need to remove the second dimension (the time dimension)
        in_to_hid = DenseLayer(InputLayer((n_batch,) + input_shape[2:]),
                               num_units, W=W_in_to_hid, b=b,
                               nonlinearity=None)
        in_to_hid = highway_dense(in_to_hid)

        # The hidden-to-hidden layer expects its inputs to have num_units
        # features because it recycles the previous hidden state
        hid_to_hid = DenseLayer(InputLayer((n_batch, num_units)),
                                num_units, W=W_hid_to_hid, b=None,
                                nonlinearity=None)
        hid_to_hid = highway_dense(hid_to_hid)

        super(RecurrentLayer, self).__init__(
            input_layer, in_to_hid, hid_to_hid, nonlinearity=nonlinearity,
            hid_init=hid_init, backwards=backwards, learn_init=learn_init,
            gradient_steps=gradient_steps)

class RNN(object):
    def __init__(self,
                 data,
                 U,
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
                 conv_attn=False,
                 encoder='rnn',
                 elemwise_sum=True,
                 penalize_corr=False,
                 is_bidirectional=False):
        self.data = data
        self.batch_size = batch_size
        self.fine_tune_W = fine_tune_W
        self.fine_tune_M = fine_tune_M
        self.lr = lr
        self.lr_decay = lr_decay
        self.optimizer = optimizer
        self.sqr_norm_lim = sqr_norm_lim
        self.conv_attn = conv_attn

        img_h = MAX_LEN

        index = T.iscalar()
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
        if encoder.find('cnn') > -1 and (encoder.find('rnn') > -1 or encoder.find('lstm') > -1) and not elemwise_sum:
            self.M = theano.shared(np.eye(2*hidden_size).astype(theano.config.floatX), borrow=True)
        else:
            self.M = theano.shared(np.eye(hidden_size).astype(theano.config.floatX), borrow=True)

        c_input = embeddings[c.flatten()].reshape((c.shape[0], c.shape[1], embeddings.shape[1]))
        r_input = embeddings[r.flatten()].reshape((r.shape[0], r.shape[1], embeddings.shape[1]))

        l_in = lasagne.layers.InputLayer(shape=(batch_size, img_h, img_w))

        if encoder.find('cnn') > -1:
            l_conv_in = lasagne.layers.ReshapeLayer(l_in, shape=(batch_size, 1, img_h, img_w))
            conv_layers = []
            for filter_size in filter_sizes:
                conv_layer = lasagne.layers.Conv2DLayer(
                        l_conv_in,
                        num_filters=num_filters,
                        filter_size=(filter_size, img_w),
                        stride=(1,1),
                        nonlinearity=lasagne.nonlinearities.rectify,
                        border_mode='valid'
                        )
                pool_layer = lasagne.layers.MaxPool2DLayer(
                        conv_layer,
                        pool_size=(img_h-filter_size+1, 1)
                        )
                conv_layers.append(pool_layer)

            l_conv = lasagne.layers.ConcatLayer(conv_layers)
            l_conv = lasagne.layers.DenseLayer(l_conv, num_units=hidden_size, nonlinearity=lasagne.nonlinearities.tanh)

        if is_bidirectional:
            if encoder.find('lstm') > -1:
                l_fwd = lasagne.layers.LSTMLayer(l_in,
                                                 hidden_size,
                                                 backwards=False,
                                                 learn_init=True,
                                                 peepholes=True)

                l_bck = lasagne.layers.LSTMLayer(l_in,
                                                 hidden_size,
                                                 backwards=True,
                                                 learn_init=True,
                                                 peepholes=True)
            else:
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

            l_recurrent = lasagne.layers.ConcatLayer([l_fwd, l_bck])
        else:
            if encoder.find('lstm') > -1:
                l_recurrent = lasagne.layers.LSTMLayer(l_in,
                                                       hidden_size,
                                                       backwards=False,
                                                       learn_init=True,
                                                       peepholes=True)
            else:
                l_recurrent = lasagne.layers.RecurrentLayer(l_in,
                                                            hidden_size,
                                                            nonlinearity=lasagne.nonlinearities.tanh,
                                                            W_hid_to_hid=lasagne.init.Orthogonal(),
                                                            W_in_to_hid=lasagne.init.Orthogonal(),
                                                            backwards=False,
                                                            learn_init=True
                                                            )

        recurrent_size = hidden_size * 2 if is_bidirectional else hidden_size
        if conv_attn:
            l_rconv_in = lasagne.layers.InputLayer(shape=(batch_size, img_h, recurrent_size))
            l_rconv_in = lasagne.layers.ReshapeLayer(l_rconv_in, shape=(batch_size, 1, img_h, recurrent_size))
            conv_layers = []
            for filter_size in filter_sizes:
                conv_layer = lasagne.layers.Conv2DLayer(
                        l_rconv_in,
                        num_filters=num_filters,
                        filter_size=(filter_size, recurrent_size),
                        stride=(1,1),
                        nonlinearity=lasagne.nonlinearities.rectify,
                        border_mode='valid'
                        )
                pool_layer = lasagne.layers.MaxPool2DLayer(
                        conv_layer,
                        pool_size=(img_h-filter_size+1, 1)
                        )
                conv_layers.append(pool_layer)

            l_hidden1 = lasagne.layers.ConcatLayer(conv_layers)
            l_hidden2 = lasagne.layers.DenseLayer(l_hidden1, num_units=hidden_size, nonlinearity=lasagne.nonlinearities.tanh)
            l_out = l_hidden2
        else:
            l_out = l_recurrent

        if conv_attn:
            e_context = l_recurrent.get_output(c_input, mask=c_mask, deterministic=False)
            e_response = l_recurrent.get_output(r_input, mask=r_mask, deterministic=False)
            def step_fn(row_t, mask_t):
                return row_t * mask_t.reshape((-1, 1))
            if is_bidirectional:
                e_context, _ = theano.scan(step_fn, outputs_info=None, sequences=[e_context, T.concatenate([c_mask, c_mask], axis=1)])
                e_response, _ = theano.scan(step_fn, outputs_info=None, sequences=[e_response, T.concatenate([r_mask, r_mask], axis=1)])
            else:
                e_context, _ = theano.scan(step_fn, outputs_info=None, sequences=[e_context, c_mask])
                e_response, _ = theano.scan(step_fn, outputs_info=None, sequences=[e_response, r_mask])

            e_context = l_out.get_output(e_context, mask=c_mask, deterministic=False)
            e_response = l_out.get_output(e_response, mask=r_mask, deterministic=False)
        else:
            e_context = l_out.get_output(c_input, mask=c_mask, deterministic=False)[T.arange(batch_size), c_seqlen].reshape((c.shape[0], hidden_size))
            e_response = l_out.get_output(r_input, mask=r_mask, deterministic=False)[T.arange(batch_size), r_seqlen].reshape((r.shape[0], hidden_size))

        if encoder.find('cnn') > -1:
            e_conv_context = l_conv.get_output(c_input, deterministic=False)
            e_conv_response = l_conv.get_output(r_input, deterministic=False)
            if encoder.find('rnn') > -1 or encoder.find('lstm') > -1:
                if elemwise_sum:
                    e_context = e_context + e_conv_context
                    e_response = e_response + e_conv_response
                else:
                    e_context = T.concatenate([e_context, e_conv_context], axis=1)
                    e_response = T.concatenate([e_response, e_conv_response], axis=1)

                # penalize correlation
                if penalize_corr:
                    cor = []
                    for i in range(hidden_size if elemwise_sum else 2*hidden_size):
                        y1, y2 = e_context, e_response
                        x1 = y1[:,i] - (np.ones(batch_size)*(T.sum(y1[:,i])/batch_size))
                        x2 = y2[:,i] - (np.ones(batch_size)*(T.sum(y2[:,i])/batch_size))
                        nr = T.sum(x1 * x2) / (T.sqrt(T.sum(x1 * x1))*T.sqrt(T.sum(x2 * x2)))
                        cor.append(-nr)
            else:
                e_context = e_conv_context
                e_response = e_conv_response

        dp = T.batched_dot(e_context, T.dot(e_response, self.M.T))
        #dp = pp('dp')(dp)
        o = T.nnet.sigmoid(dp)
        o = T.clip(o, 1e-7, 1.0-1e-7)

        self.shared_data = {}
        for key in ['c', 'r']:
            self.shared_data[key] = theano.shared(np.zeros((batch_size, MAX_LEN), dtype=np.int32))
        for key in ['c_mask', 'r_mask']:
            self.shared_data[key] = theano.shared(np.zeros((batch_size, MAX_LEN), dtype=theano.config.floatX))
        for key in ['y', 'c_seqlen', 'r_seqlen']:
            self.shared_data[key] = theano.shared(np.zeros((batch_size,), dtype=np.int32))

        self.probas = T.concatenate([(1-o).reshape((-1,1)), o.reshape((-1,1))], axis=1)
        self.pred = T.argmax(self.probas, axis=1)
        self.errors = T.sum(T.neq(self.pred, y))
        self.cost = T.nnet.binary_crossentropy(o, y).mean()
        if penalize_corr and encoder.find('cnn') > -1 and (encoder.find('rnn') > -1 or encoder.find('lstm') > -1):
            self.cost += 4 * T.sum(cor)
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
        if self.conv_attn:
            params += lasagne.layers.get_all_params(self.l_recurrent)
        if self.fine_tune_W:
            params += [self.embeddings]
        if self.fine_tune_M:
            params += [self.M]

        total_params = sum([p.get_value().size for p in params])
        print "total_params: ", total_params

        if 'adam' == self.optimizer:
            updates = adam(self.cost, params, learning_rate=self.lr)
        elif 'adadelta' == self.optimizer:
            updates = sgd_updates_adadelta(self.cost, params, self.lr_decay, 1e-6, self.sqr_norm_lim)
#            updates = lasagne.updates.adadelta(self.cost, params, learning_rate=1.0, rho=self.lr_decay)
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

    def get_batch(self, dataset, index, max_l=MAX_LEN):
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
        c, c_seqlen, c_mask = self.get_batch(dataset['c'], index)
        r, r_seqlen, r_mask = self.get_batch(dataset['r'], index)
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

def pad_to_batch_size(X, batch_size):
    n_seqs = len(X)
    n_batches_out = np.ceil(float(n_seqs) / batch_size)
    n_seqs_out = batch_size * n_batches_out

    to_pad = n_seqs % batch_size
    if to_pad > 0:
        X += X[:batch_size-to_pad]
    return X

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

def main():
  parser = argparse.ArgumentParser()
  parser.register('type','bool',str2bool)
  parser.add_argument('--conv_attn', type='bool', default=False, help='Use convolutional attention')
  parser.add_argument('--encoder', type=str, default='rnn', help='Encoder')
  parser.add_argument('--hidden_size', type=int, default=200, help='Hidden size')
  parser.add_argument('--fine_tune_W', type='bool', default=False, help='Whether to fine-tune W')
  parser.add_argument('--fine_tune_M', type='bool', default=False, help='Whether to fine-tune M')
  parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size')
  parser.add_argument('--shuffle_batch', type='bool', default=False, help='Shuffle batch')
  parser.add_argument('--is_bidirectional', type='bool', default=False, help='Bidirectional RNN/LSTM')
  parser.add_argument('--n_epochs', type=int, default=100, help='Num epochs')
  parser.add_argument('--lr_decay', type=float, default=0.95, help='Learning rate decay')
  parser.add_argument('--sqr_norm_lim', type=float, default=1, help='Squared norm limit')
  parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
  parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer')
  parser.add_argument('--suffix', type=str, default='', help='Suffix for pkl files')
  parser.add_argument('--use_pv', type='bool', default=False, help='Use PV')
  args = parser.parse_args()
  print "args: ", args

  print "loading data...",
  if args.use_pv:
      data = cPickle.load(open('../data/all_pv.pkl'))
      train_data = { 'c': data['c'][:1000000], 'r': data['r'][:1000000], 'y': data['y'][:1000000] }
      val_data = { 'c': data['c'][1000000:1356080], 'r': data['r'][1000000:1356080], 'y': data['y'][1000000:1356080] }
      test_data = { 'c': data['c'][1000000+356080:], 'r': data['r'][1000000+356080:], 'y': data['y'][1000000+356080:] }

      for key in ['c', 'r', 'y']:
          for dataset in [train_data, val_data]:
              dataset[key] = pad_to_batch_size(dataset[key], BATCH_SIZE)

      W = cPickle.load(open('../data/pv_vectors_10d.txt.pkl', 'rb'))
  else:
      train_data, val_data, test_data = cPickle.load(open('dataset%s.pkl' % args.suffix, 'rb'))
      W, _ = cPickle.load(open('W%s.pkl' % args.suffix, 'rb'))
  print "data loaded!"

  data = { 'train' : train_data, 'val': val_data, 'test': test_data }

  rnn = RNN(data,
            W.astype(theano.config.floatX),
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
            conv_attn=args.conv_attn)

  print rnn.train(n_epochs=args.n_epochs, shuffle_batch=args.shuffle_batch)

if __name__ == '__main__':
  main()
