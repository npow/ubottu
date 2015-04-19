import cPickle
import lasagne
import numpy as np
import re
import sys
import theano
import theano.tensor as T
from collections import defaultdict, OrderedDict
from theano.printing import Print as pp

class CNN(object):
    def __init__(self,
                 data,
                 U,
                 img_w=300,
                 filter_sizes=[3,4,5],
                 num_filters=100,
                 batch_size=50,
                 lr_decay=0.95,
                 sqr_norm_lim=9,
                 non_static=True):
        self.data = data
        self.batch_size = batch_size
        img_h = data['train']['r'].shape[1]
        print img_h
        conv_layers = []
        for filter_size in filter_sizes:
            l_in = lasagne.layers.InputLayer(shape=(None, 1, img_h, img_w))
            conv_layer = lasagne.layers.Conv2DLayer(
                    l_in,
                    num_filters=num_filters,
                    filter_size=(filter_size, img_w),
                    strides=(1,1),
                    nonlinearity=lasagne.nonlinearities.rectify
                    )
            pool_layer = lasagne.layers.MaxPool2DLayer(
                    conv_layer,
                    ds=(img_h-filter_size+1, 1)
                    )
            conv_layers.append(pool_layer)
            
        hidden_size = len(conv_layers) * num_filters
        hidden_size = num_filters
        l_hidden1 = lasagne.layers.ConcatLayer(conv_layers)
        l_hidden2 = lasagne.layers.DenseLayer(l_hidden1, num_units=hidden_size, nonlinearity=lasagne.nonlinearities.tanh)
#        l_hidden2 = lasagne.layers.DropoutLayer(l_hidden2, 0.5)
        #self.M = theano.shared(np.random.randn(hidden_size, hidden_size).astype(theano.config.floatX))
        self.M = theano.shared(np.eye(hidden_size).astype(theano.config.floatX))

        index = T.iscalar()
        c = T.imatrix('c')
        r = T.imatrix('r')
        y = T.ivector('y')
        embeddings = theano.shared(U, name='embeddings')
        zero_vec_tensor = T.fvector()
        self.zero_vec = np.zeros(img_w, dtype=theano.config.floatX)
        self.set_zero = theano.function([zero_vec_tensor], updates=[(embeddings, T.set_subtensor(embeddings[0,:], zero_vec_tensor))])

        c_input = embeddings[c.flatten()].reshape((c.shape[0], 1, c.shape[1], embeddings.shape[1]))
        r_input = embeddings[r.flatten()].reshape((r.shape[0], 1, r.shape[1], embeddings.shape[1]))

        e_context = l_hidden2.get_output(c_input, deterministic=True)
        e_response = l_hidden2.get_output(r_input, deterministic=True)

        o = T.nnet.sigmoid(T.batched_dot(e_context, T.dot(e_response, self.M.T)))
        o = T.clip(o, 1e-7, 1.0-1e-7)

        cost = T.nnet.binary_crossentropy(o, y).mean()
        params = lasagne.layers.get_all_params(l_hidden2) + [self.M]
        if non_static:
            params += [embeddings]
        #updates = lasagne.updates.adadelta(cost, params, learning_rate=1.0, rho=lr_decay)
        updates = sgd_updates_adadelta(cost, params, lr_decay, 1e-6, sqr_norm_lim)
#        updates = lasagne.updates.adagrad(cost, params, learning_rate=0.01)
        
        self.train_set_c = theano.shared(data['train']['c'], borrow=True)
        self.train_set_r = theano.shared(data['train']['r'], borrow=True)
        self.train_set_y = theano.shared(data['train']['y'], borrow=True)
        self.val_set_c = theano.shared(data['val']['c'], borrow=True)
        self.val_set_r = theano.shared(data['val']['r'], borrow=True)
        self.val_set_y = theano.shared(data['val']['y'], borrow=True)
        
        probas = T.concatenate([(1-o).reshape((-1,1)), o.reshape((-1,1))], axis=1)
        pred = T.argmax(probas, axis=1)
        errors = T.mean(T.neq(pred, y))
        
        self.train_model = theano.function([index], cost, updates=updates,
              givens={
                c: self.train_set_c[index*batch_size:(index+1)*batch_size],
                r: self.train_set_r[index*batch_size:(index+1)*batch_size],
                y: self.train_set_y[index*batch_size:(index+1)*batch_size]})         
        
        self.val_model = theano.function([index], errors,
             givens={
                c: self.val_set_c[index * batch_size: (index + 1) * batch_size],
                r: self.val_set_r[index * batch_size: (index + 1) * batch_size],
                y: self.val_set_y[index * batch_size: (index + 1) * batch_size]})

        self.test_model = theano.function([index], errors,
                 givens={
                    c: self.train_set_c[index * batch_size: (index + 1) * batch_size],
                    r: self.train_set_r[index * batch_size: (index + 1) * batch_size],
                    y: self.train_set_y[index * batch_size: (index + 1) * batch_size]})    

        test_error = T.mean(T.neq(pred, y))
        self.test_model_all = theano.function([c, r, y], test_error)   
    
    def train(self, n_epochs=100, shuffle_batch=True):
        epoch = 0
        best_val_perf = 0
        val_perf = 0
        test_perf = 0
        cost_epoch = 0
        
        n_train_batches = self.data['train']['y'].shape[0] // self.batch_size
        n_val_batches = self.data['val']['y'].shape[0] // self.batch_size
        
        test_set_c, test_set_r, test_set_y = self.data['test']['c'], self.data['test']['r'], self.data['test']['y']
        while (epoch < n_epochs):
            epoch += 1
            indices = range(n_train_batches)
            if shuffle_batch:
                indices = np.random.permutation(indices)
            for minibatch_index in indices:
                cost_epoch = self.train_model(minibatch_index)
                self.set_zero(self.zero_vec)
            train_losses = [self.test_model(i) for i in xrange(n_train_batches)]
            train_perf = 1 - np.mean(train_losses)
            val_losses = [self.val_model(i) for i in xrange(n_val_batches)]
            val_perf = 1 - np.mean(val_losses)                        
            print 'epoch %i, train perf %f %%, val perf %f' % (epoch, train_perf * 100., val_perf*100.)
            if val_perf >= best_val_perf:
                best_val_perf = val_perf
#                test_loss = self.test_model_all(test_set_c, test_set_r, test_set_y)        
#                test_perf = 1 - test_loss         
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
 
def get_idx_from_sent(sent, word_idx_map, max_l, k, filter_h):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    pad = filter_h - 1
    for i in xrange(pad):
        x.append(0)
    words = sent.split()
    for word in words[:max_l]:
        if word in word_idx_map:
            x.append(word_idx_map[word])
    while len(x) < max_l+2*pad:
        x.append(0)
    return x

def make_idx_data(dataset, word_idx_map, max_l=152, k=300, filter_h=5):
    """
    Transforms sentences into a 2-d matrix.
    """
    for i in xrange(len(dataset['y'])):
        dataset['c'][i] = get_idx_from_sent(dataset['c'][i], word_idx_map, max_l, k, filter_h)
        dataset['r'][i] = get_idx_from_sent(dataset['r'][i], word_idx_map, max_l, k, filter_h)
    for col in ['c', 'r', 'y']:
        dataset[col] = np.array(dataset[col], dtype=np.int32)

def main():
    print "loading data...",
    data = cPickle.load(open('data.pkl', 'rb'))
    train_data, val_data, test_data, W, W2, word_idx_map, vocab = data
    print "data loaded!"

    make_idx_data(train_data, word_idx_map)
    make_idx_data(val_data, word_idx_map)
    make_idx_data(test_data, word_idx_map)
    for key in ['c', 'r', 'y']:
        print train_data[key].shape
        print val_data[key].shape
        print test_data[key].shape

    data = { 'train' : train_data, 'val': val_data, 'test': test_data }

    cnn = CNN(data,
              W.astype(theano.config.floatX),
              img_w=300,
              filter_sizes=[3,4,5],
              num_filters=100,
              batch_size=64,
              lr_decay=0.95,
              sqr_norm_lim=9,
              non_static=True)
    print cnn.train(n_epochs=100, shuffle_batch=True)

if __name__ == '__main__':
    main()
