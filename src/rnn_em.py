from __future__ import division
import numpy as np
import theano
import theano.tensor as T
from lasagne import init
from lasagne import nonlinearities
from lasagne.layers import helper, Gate, InputLayer, Layer
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

class CustomRecurrentLayer(Layer):
    """
    lasagne.layers.recurrent.CustomRecurrentLayer(incoming, input_to_hidden,
    hidden_to_hidden, nonlinearity=lasagne.nonlinearities.rectify,
    hid_init=lasagne.init.Constant(0.), backwards=False,
    learn_init=False, gradient_steps=-1, grad_clipping=False,
    unroll_scan=False, precompute_input=True, **kwargs)

    A layer which implements a recurrent connection.

    This layer allows you to specify custom input-to-hidden and
    hidden-to-hidden connections by instantiating layer instances and passing
    them on initialization.  The output shape for the provided layers must be
    the same.  If you are looking for a standard, densely-connected recurrent
    layer, please see :class:`RecurrentLayer`.  The output is computed
    by

    .. math ::
        h_t = \sigma(f_i(x_t) + f_h(h_{t-1}))

    Parameters
    ----------
    incoming : a :class:`lasagne.layers.Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape.
    input_to_hidden : :class:`lasagne.layers.Layer`
        Layer which connects input to the hidden state (:math:`f_i`).
    hidden_to_hidden : :class:`lasagne.layers.Layer`
        Layer which connects the previous hidden state to the new state
        (:math:`f_h`).
    nonlinearity : callable or None
        Nonlinearity to apply when computing new state (:math:`\sigma`). If
        None is provided, no nonlinearity will be applied.
    hid_init : callable, np.ndarray, theano.shared or TensorVariable
        Passing in a TensorVariable allows the user to specify
        the value of `hid_init` (:math:`h_0`). In this mode, `learn_init` is
        ignored.
    backwards : bool
        If True, process the sequence backwards and then reverse the
        output again such that the output from the layer is always
        from :math:`x_1` to :math:`x_n`.
    learn_init : bool
        If True, initial hidden values are learned. If `hid_init` is a
        TensorVariable then the TensorVariable is used and
        `learn_init` is ignored.
    gradient_steps : int
        Number of timesteps to include in the backpropagated gradient.
        If -1, backpropagate through the entire sequence.
    grad_clipping : False or float
        If a float is provided, the gradient messages are clipped during the
        backward pass.  If False, the gradients will not be clipped.  See [1]_
        (p. 6) for further explanation.
    unroll_scan : bool
        If True the recursion is unrolled instead of using scan. For some
        graphs this gives a significant speed up but it might also consume
        more memory. When `unroll_scan` is true then the `gradient_steps`
        setting is ignored. The input sequence length cannot be specified as
        None when `unroll_scan` is True.
    precompute_input : bool
        If True, precompute input_to_hid before iterating through
        the sequence. This can result in a speedup at the expense of
        an increase in memory usage.

    References
    ----------
    .. [1] Graves, Alex: "Generating sequences with recurrent neural networks."
           arXiv preprint arXiv:1308.0850 (2013).
    """
    def __init__(self, incoming, input_to_hidden, hidden_to_hidden,
                 nonlinearity=nonlinearities.rectify,
                 hid_init=init.Constant(0.),
                 backwards=False,
                 learn_init=False,
                 gradient_steps=-1,
                 grad_clipping=False,
                 unroll_scan=False,
                 precompute_input=True,
                 **kwargs):

        super(CustomRecurrentLayer, self).__init__(incoming, **kwargs)

        self.input_to_hidden = input_to_hidden
        self.hidden_to_hidden = hidden_to_hidden
        self.learn_init = learn_init
        self.backwards = backwards
        self.gradient_steps = gradient_steps
        self.grad_clipping = grad_clipping
        self.unroll_scan = unroll_scan
        self.precompute_input = precompute_input

        if unroll_scan and gradient_steps != -1:
            raise ValueError(
                "Gradient steps must be -1 when unroll_scan is true.")

        if unroll_scan and self.input_shape[1] is None:
            raise ValueError("Input sequence length cannot be specified as "
                             "None when unroll_scan is True")

        # Check that output shapes match
        if input_to_hidden.output_shape != hidden_to_hidden.output_shape:
            raise ValueError("The output shape for input_to_hidden and "
                             "input_to_hidden must be equal, but "
                             "input_to_hidden.output_shape={} and "
                             "hidden_to_hidden.output_shape={}".format(
                                 input_to_hidden.output_shape,
                                 hidden_to_hidden.output_shape))

        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        # Get the input dimensionality and number of units based on the
        # expected output of the input-to-hidden layer
        self.num_inputs = np.prod(self.input_shape[2:])
        self.num_units = input_to_hidden.output_shape[-1]

        # Initialize hidden state
        if isinstance(hid_init, T.TensorVariable):
            if hid_init.ndim != 2:
                raise ValueError(
                    "When hid_init is provided as a TensorVariable, it should "
                    "have 2 dimensions and have shape (num_batch, num_units)")
            self.hid_init = hid_init
        else:
            self.hid_init = self.add_param(
                hid_init, (1, self.num_units), name="hid_init",
                trainable=learn_init, regularizable=False)

    def get_params(self, **tags):
        # Get all parameters from this layer, the master layer
        params = super(CustomRecurrentLayer, self).get_params(**tags)
        # Combine with all parameters from the child layers
        params += helper.get_all_params(self.input_to_hidden, **tags)
        params += helper.get_all_params(self.hidden_to_hidden, **tags)
        return params

    def get_output_shape_for(self, input_shape):
        return input_shape[0], input_shape[1], self.num_units

    def get_output_for(self, input, mask=None, **kwargs):
        """
        Compute this layer's output function given a symbolic input variable.

        Parameters
        ----------
        input : theano.TensorType
            Symbolic input variable.
        mask : theano.TensorType
            Theano variable denoting whether each time step in each
            sequence in the batch is part of the sequence or not.  If ``None``,
            then it is assumed that all sequences are of the same length.  If
            not all sequences are of the same length, then it must be
            supplied as a matrix of shape ``(n_batch, n_time_steps)`` where
            ``mask[i, j] = 1`` when ``j <= (length of sequence i)`` and
            ``mask[i, j] = 0`` when ``j > (length of sequence i)``.

        Returns
        -------
        layer_output : theano.TensorType
            Symbolic output variable.
        """

        # Treat all dimensions after the second as flattened feature dimensions
        if input.ndim > 3:
            input = input.reshape((input.shape[0], input.shape[1],
                                   T.prod(input.shape[2:])))

        # Input should be provided as (n_batch, n_time_steps, n_features)
        # but scan requires the iterable dimension to be first
        # So, we need to dimshuffle to (n_time_steps, n_batch, n_features)
        input = input.dimshuffle(1, 0, 2)
        seq_len, num_batch, _ = input.shape

        if self.precompute_input:
            # Because the input is given for all time steps, we can precompute
            # the inputs to hidden before scanning. First we need to reshape
            # from (seq_len, batch_size, num_inputs) to
            # (seq_len*batch_size, num_inputs)
            input = T.reshape(input,
                              (seq_len*num_batch, -1))
            input = helper.get_output(
                self.input_to_hidden, input, **kwargs)

            # Reshape back to (seq_len, batch_size, num_units)
            input = T.reshape(input, (seq_len, num_batch, -1))

        # We will always pass the hidden-to-hidden layer params to step
        non_seqs = helper.get_all_params(self.hidden_to_hidden)
        # When we are not precomputing the input, we also need to pass the
        # input-to-hidden parameters to step
        if not self.precompute_input:
            non_seqs += helper.get_all_params(self.input_to_hidden)

        # Create single recurrent computation step function
        def step(input_n, hid_previous, *args):
            # Compute the hidden-to-hidden activation
            hid_pre = helper.get_output(self.hidden_to_hidden, hid_previous)

            # If the dot product is precomputed then add it, otherwise
            # calculate the input_to_hidden values and add them
            if self.precompute_input:
                hid_pre += input_n
            else:
                hid_pre += helper.get_output(self.input_to_hidden, input_n)

            # Clip gradients
            if self.grad_clipping is not False:
                hid_pre = theano.gradient.grad_clip(
                    hid_pre, -self.grad_clipping, self.grad_clipping)

            return self.nonlinearity(hid_pre)

        def step_masked(input_n, mask_n, hid_previous, *args):
            # If mask is 0, use previous state until mask = 1 is found.
            # This propagates the layer initial state when moving backwards
            # until the end of the sequence is found.
            hid = step(input_n, hid_previous, *args)
            hid_out = hid*mask_n + hid_previous*(1 - mask_n)
            return [hid_out]

        if mask is not None:
            mask = mask.dimshuffle(1, 0, 'x')
            sequences = [input, mask]
            step_fun = step_masked
        else:
            sequences = input
            step_fun = step

        # When hid_init is provided as a TensorVariable, use it as-is
        if isinstance(self.hid_init, T.TensorVariable):
            hid_init = self.hid_init
        else:
            # Dot against a 1s vector to repeat to shape (num_batch, num_units)
            hid_init = T.dot(T.ones((num_batch, 1)), self.hid_init)

        if self.unroll_scan:
            # Explicitly unroll the recurrence instead of using scan
            hid_out = unroll_scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=[hid_init],
                go_backwards=self.backwards,
                non_sequences=non_seqs,
                n_steps=self.input_shape[1])[0]
        else:
            # Scan op iterates over first dimension of input and repeatedly
            # applies the step function
            hid_out = theano.scan(
                fn=step_fun,
                sequences=sequences,
                go_backwards=self.backwards,
                outputs_info=[hid_init],
                non_sequences=non_seqs,
                truncate_gradient=self.gradient_steps,
                strict=True)[0]

        # dimshuffle back to (n_batch, n_time_steps, n_features))
        hid_out = hid_out.dimshuffle(1, 0, 2)

        # if scan is backward reverse the output
        if self.backwards:
            hid_out = hid_out[:, ::-1, :]

        self.hid_out = hid_out
        return hid_out


class RecurrentLayer(CustomRecurrentLayer):
    """
    lasagne.layers.recurrent.RecurrentLayer(incoming, num_units,
    W_in_to_hid=lasagne.init.Uniform(), W_hid_to_hid=lasagne.init.Uniform(),
    b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.rectify,
    hid_init=lasagne.init.Constant(0.), backwards=False, learn_init=False,
    gradient_steps=-1, grad_clipping=False, unroll_scan=False,
    precompute_input=True, **kwargs)

    Dense recurrent neural network (RNN) layer

    A "vanilla" RNN layer, which has dense input-to-hidden and
    hidden-to-hidden connections.  The output is computed as

    .. math ::
        h_t = \sigma(W_x x_t + W_h h_{t-1} + b)

    Parameters
    ----------
    incoming : a :class:`lasagne.layers.Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape.
    num_units : int
        Number of hidden units in the layer.
    W_in_to_hid : Theano shared variable, numpy array or callable
        Initializer for input-to-hidden weight matrix (:math:`W_x`).
    W_hid_to_hid : Theano shared variable, numpy array or callable
        Initializer for hidden-to-hidden weight matrix (:math:`W_h`).
    b : Theano shared variable, numpy array, callable or None
        Initializer for bias vector (:math:`b`). If None is provided there will
        be no bias.
    nonlinearity : callable or None
        Nonlinearity to apply when computing new state (:math:`\sigma`). If
        None is provided, no nonlinearity will be applied.
    hid_init : callable, np.ndarray, theano.shared or TensorVariable
        Passing in a TensorVariable allows the user to specify
        the value of `hid_init` (:math:`h_0`). In this mode `learn_init` is
        ignored.
    backwards : bool
        If True, process the sequence backwards and then reverse the
        output again such that the output from the layer is always
        from :math:`x_1` to :math:`x_n`.
    learn_init : bool
        If True, initial hidden values are learned. If `hid_init` is a
        TensorVariable then `learn_init` is ignored.
    gradient_steps : int
        Number of timesteps to include in the backpropagated gradient.
        If -1, backpropagate through the entire sequence.
    grad_clipping : False or float
        If a float is provided, the gradient messages are clipped during the
        backward pass.  If False, the gradients will not be clipped.  See [1]_
        (p. 6) for further explanation.
    unroll_scan : bool
        If True the recursion is unrolled instead of using scan. For some
        graphs this gives a significant speed up but it might also consume
        more memory. When `unroll_scan` is true then the `gradient_steps`
        setting is ignored. The input sequence length cannot be specified as
        None when `unroll_scan` is True.
    precompute_input : bool
        If True, precompute input_to_hid before iterating through
        the sequence. This can result in a speedup at the expense of
        an increase in memory usage.

    References
    ----------
    .. [1] Graves, Alex: "Generating sequences with recurrent neural networks."
           arXiv preprint arXiv:1308.0850 (2013).
    """
    def __init__(self, incoming, num_units,
                 W_in_to_hid=init.Uniform(),
                 W_hid_to_hid=init.Uniform(),
                 b=init.Constant(0.),
                 nonlinearity=nonlinearities.rectify,
                 hid_init=init.Constant(0.),
                 backwards=False,
                 learn_init=False,
                 gradient_steps=-1,
                 grad_clipping=False,
                 unroll_scan=False,
                 precompute_input=True,
                 **kwargs):
        input_shape = helper.get_output_shape(incoming)
        num_batch = input_shape[0]
        # We will be passing the input at each time step to the dense layer,
        # so we need to remove the second dimension (the time dimension)
        in_to_hid = DenseLayer(InputLayer((num_batch,) + input_shape[2:]),
                               num_units, W=W_in_to_hid, b=b,
                               nonlinearity=None, **kwargs)
        # The hidden-to-hidden layer expects its inputs to have num_units
        # features because it recycles the previous hidden state
        hid_to_hid = DenseLayer(InputLayer((num_batch, num_units)),
                                num_units, W=W_hid_to_hid, b=None,
                                nonlinearity=None, **kwargs)

        # Make child layer parameters intuitively accessible
        self.W_in_to_hid = in_to_hid.W
        self.W_hid_to_hid = hid_to_hid.W
        self.b = in_to_hid.b

        # Just use the CustomRecurrentLayer with the DenseLayers we created
        super(RecurrentLayer, self).__init__(
            incoming, in_to_hid, hid_to_hid, nonlinearity=nonlinearity,
            hid_init=hid_init, backwards=backwards, learn_init=learn_init,
            gradient_steps=gradient_steps,
            grad_clipping=grad_clipping, unroll_scan=unroll_scan,
            precompute_input=precompute_input, **kwargs)
