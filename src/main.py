import itertools
import joblib
import logging
import pprint
import math
import numpy as np
import os
import operator
import theano
import theano.tensor as T
from collections import OrderedDict
from theano.printing import Print as pp

from fuel.streams import DataStream
from fuel.datasets import TextFile
from fuel.schemes import ShuffledScheme
from fuel.transformers import Mapping, Batch, Merge, Padding, Filter, ForceFloatX
from fuel.datasets import OneBillionWord, TextFile
from fuel.schemes import ConstantScheme

from six.moves import input

from blocks import initialization
from blocks.bricks import Tanh, Initializable, Identity
from blocks.bricks.base import application
from blocks.bricks.lookup import LookupTable
from blocks.bricks.recurrent import SimpleRecurrent, Bidirectional, BaseRecurrent, recurrent
from blocks.bricks.attention import SequenceContentAttention
from blocks.bricks.parallel import Fork
from blocks.bricks.sequence_generators import (SequenceGenerator, LinearReadout, SoftmaxEmitter, TrivialEmitter, LookupFeedback)
from blocks.config_parser import config
from blocks.graph import ComputationGraph
from blocks.algorithms import (GradientDescent, Scale, StepClipping, CompositeRule)
from blocks.dump import load_parameter_values
from blocks.extensions import FinishAfter, Printing, Timing
from blocks.extensions.saveload import Checkpoint
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.extensions.plot import Plot
from blocks.filter import VariableFilter
from blocks.initialization import Orthogonal, IsotropicGaussian, Constant
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.monitoring import aggregation
from blocks.search import BeamSearch
from blocks.utils import named_copy, dict_union, equizip

EN_FILE = '../data/X1.txt'
FR_FILE = '../data/X2.txt'
LABEL_FILE = '../data/Y.txt'

EN_DICT = joblib.load('../blobs/vocab.en.pkl')
FR_DICT = joblib.load('../blobs/vocab.fr.pkl')

EMBEDDING_SIZE = 1
HIDDEN_SIZE = EMBEDDING_SIZE
NUM_EPOCHS = 1
VOCAB_SIZE = len(EN_DICT)
MARGIN = 1
ALPHA = 0.1

def parse(L):
    english = L[0]
    french = L[1]
    label = L[2]
    return [parse_sentence(english), parse_sentence(french), label]

def parse_sentence(x):
    x = x[1:-1]
    arr = []
    inside = False
    curr = []
    for i in x:
        if i == 0:
            if not inside:
                inside = True
            else:
                inside = False
                curr = np.array(curr, dtype=np.int32)
                curr.resize((1,5))
                arr.append(curr)
                curr = []
        else:
            curr.append(i)
    arr = np.concatenate(arr, axis=0)
    return arr

class model(object):
    def __init__(self, nh, ne, de):
        '''
        nh :: dimension of the hidden layer
        ne :: number of word embeddings in the vocabulary
        de :: dimension of the word embeddings
        '''
        # parameters of the model
        self.emb = theano.shared(1000.2 * np.random.uniform(-1.0, 1.0,\
                   (ne+1, de)).astype(theano.config.floatX)) # add one for PADDING at the end

        # first recurrence
        self.Wx  = theano.shared(0.2 * np.random.uniform(-1.0, 1.0,\
                   (de, nh)).astype(theano.config.floatX))
        self.Wh  = theano.shared(0.2 * np.random.uniform(-1.0, 1.0,\
                   (nh, nh)).astype(theano.config.floatX))
        self.bh  = theano.shared(np.zeros(nh, dtype=theano.config.floatX))
        self.h0  = theano.shared(np.zeros(nh, dtype=theano.config.floatX))

        # second recurrence
        self.Wx_1 = theano.shared(0.2 * np.random.uniform(-1.0, 1.0,\
                   (de, nh)).astype(theano.config.floatX))
        self.Wh_1 = theano.shared(0.2 * np.random.uniform(-1.0, 1.0,\
                   (nh, nh)).astype(theano.config.floatX))
        self.bh_1 = theano.shared(np.zeros(nh, dtype=theano.config.floatX))
        self.h0_1 = theano.shared(np.zeros(nh, dtype=theano.config.floatX))

        # bundle
        # FIXME: Why is h0 a parameter?
        # TODO: Use pretrained embeddings
        self.params = [self.emb, self.Wx, self.Wh, self.bh, self.h0, self.Wx_1, self.Wh_1, self.bh_1, self.h0_1]
        self.names  = ['embeddings', 'Wx', 'Wh', 'bh', 'h0', 'Wx_1', 'Wh_1', 'bh_1', 'h0_1']

        x1 = T.imatrix()
        x2 = T.imatrix()
        y = T.iscalar()

        def sentence_recurrence(x_t, h_tm1):
            s_h, _ = theano.scan(fn=word_recurrence, sequences=x_t, outputs_info=self.h0_1, n_steps=T.shape(x_t)[0])
            return T.nnet.sigmoid(T.dot(s_h[-1], self.Wx) + T.dot(h_tm1, self.Wh) + self.bh)

        def word_recurrence(idx, h_tm1):
            x_t = self.emb[idx]
            return T.nnet.sigmoid(T.dot(x_t, self.Wx_1) + T.dot(h_tm1, self.Wh_1) + self.bh_1)

        h_x1, _ = theano.scan(fn=sentence_recurrence, sequences=x1, outputs_info=self.h0, n_steps=x1.shape[0])
        h_x2, _ = theano.scan(fn=sentence_recurrence, sequences=x2, outputs_info=self.h0, n_steps=x2.shape[0])
        e_x1 = h_x1[-1]
        e_x2 = h_x2[-1]

        # cost and gradients and learning rate
        lr = T.scalar('lr')
        cost = T.switch(T.eq(y, 1), T.sum(T.abs_(e_x1-e_x2)), T.maximum(0, MARGIN-T.sum(T.abs_(e_x1-e_x2))))
        gradients = T.grad(cost, self.params)
        updates = OrderedDict((p, p-lr*g) for p, g in zip(self.params, gradients))

        self.train = theano.function(inputs=[x1, x2, y, lr], outputs=[cost, e_x1, e_x2], updates=updates)

    def save(self, folder):   
        for param, name in zip(self.params, self.names):
            np.save(os.path.join(folder, name + '.npy'), param.get_value())

import scipy
def main():
    en_data = TextFile(files=[EN_FILE], bos_token='<s>', eos_token='</s>', unk_token='UNK', dictionary=EN_DICT)
    fr_data = TextFile(files=[FR_FILE], bos_token='<s>', eos_token='</s>', unk_token='UNK', dictionary=FR_DICT)
    labels = TextFile(files=[LABEL_FILE], bos_token=None, eos_token=None, unk_token='UNK', dictionary={'0':0, '1':1, 'UNK':2})
    streams = (en_data.get_example_stream(), fr_data.get_example_stream(), labels.get_example_stream())
    merged_stream = Merge(streams, ('english', 'french', 'labels'))
    merged_stream = ForceFloatX(merged_stream)
    merged_stream = Mapping(merged_stream, parse)

    rnn = model(nh=HIDDEN_SIZE, ne=VOCAB_SIZE, de=EMBEDDING_SIZE)
    i = 0
    for e in xrange(100):
        for line in merged_stream.get_epoch_iterator():
            english, french, label = line[0], line[1], line[2][0]
            i += 1
            if i > 50:
                print "HERE"
                label = 0
            cost, e_x1, e_x2 = rnn.train(english, french, label, ALPHA)

            print cost, e_x1, e_x2, np.linalg.norm(e_x1-e_x2, 1)

if __name__ == '__main__':
    main()
