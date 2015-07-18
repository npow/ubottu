from __future__ import division
import numpy as np
import theano
import theano.tensor as T
from lasagne import init
from lasagne import nonlinearities
from lasagne.layers import Layer, InputLayer, DenseLayer, helper
from lasagne.utils import unroll_scan
from theano.printing import Print as pp

def norm(x):
    axis = None if x.ndim == 1 else 1
    return T.sqrt(T.sum(T.sqr(x), axis=axis))

def cos_matrix_multiplication(matrix, vector):
    matrix = matrix.T
    dotted = T.dot(matrix, vector)
    matrix_norms = norm(matrix)
    vector_norms = norm(vector)
    matrix_vector_norms = matrix_norms * vector_norms
    neighbors = dotted / matrix_vector_norms
    return 1 - neighbors

A = theano.shared(np.array([[7, 5, 8, 1, 9], [6, 6, 4, 0, 8]], dtype=np.float32).T)
B = theano.shared(np.array([1, 2, 3, 4, 5], dtype=np.float32))
print cos_matrix_multiplication(A, B).eval()

