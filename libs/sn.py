# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Taken and adapted from https://github.com/minhnhat93/tf-SNDCGAN.

If you find any errors in this implementation, please file an issue on Github.
"""
import warnings

import tensorflow as tf
from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.contrib.layers.python.layers import utils as layer_utils
from tensorflow.python.layers import convolutional as convolutional_layers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_math_ops

NO_OPS = 'NO_OPS'
# All spectral norm update ops are put under the `SPECTRAL_NORM_OPS` by default.
SPECTRAL_NORM_OPS=None


# For the original chainer implementation
# see https://github.com/pfnet-research/chainer-gan-lib/commit/7861e6ce2a188f97198e56132e0c4c619086deba
# Here's a good blog: https://christiancosgrove.com/blog/2018/01/04/spectral-normalization-explained.html
def _spectral_normed_weight(W, num_iters=1, update_collection=None, with_sigma=False):
  """Wraps around a conv2d weight to perform spectral normalization on it.

  :param W: Weight to be normalized.
  :param num_iters: Number of iterations for power iteration method (Yoshida & Miyato, 2017)
  :param update_collection: Optional string specifying the collection to add the update op to. If not specified the
    update is performed whenever W is used.
  :param with_sigma: If true, returns sigma in addition to the normalized weight.
  :return: The normalized weight `W_bar`, or (W_bar, sigma) if `with_sigma` is true.
  """
  # (FYI in the chainer implementation it is (output_channel_num, input_channels_num, kernel_h, kernel_w), which
  # corresponds to Function: [in_channels * h * w] -> out_channels. You can substitute the W in tensorflow with
  #  Transpose(T) in chainer.)
  # y=Wx. W shape = [h, w, in_channels, out_channels].
  W_shape = W.shape.as_list()
  # W_reshaped shape = [h * w * in_channels, out_channels]. It maps h * w * in_channels => out_channels.
  W_reshaped = tf.reshape(W, [-1, W_shape[-1]])
  # u shape: [1, out_channels]
  u = tf.get_variable("u", [1, W_shape[-1]], dtype=tf.float32, initializer=tf.truncated_normal_initializer(),
                      trainable=False, collections=tf.GraphKeys.MODEL_VARIABLES)
  u_casted = tf.cast(u, dtype=W.dtype, name='u_casted')
  v_casted=None

  def power_iteration(i, u_i, v_i):
    # v shape: [1, h * w * in_channels]
    v_ip1 = tf.nn.l2_normalize(tf.matmul(u_i, tf.transpose(W_reshaped)))
    # u shape: [1, out_channels]
    u_ip1 = tf.nn.l2_normalize(tf.matmul(v_ip1, W_reshaped))
    return i + 1, u_ip1, v_ip1

  # Usually num_iters = 1 will be enough.
  # _, u_final, v_final = tf.while_loop(
  #   cond=lambda i, _1, _2: i < num_iters,
  #   body=power_iteration,
  #   loop_vars=(tf.constant(0, dtype=tf.int32), u_casted, v_casted)
  # )

  for i in range(num_iters):
    _, u_casted, v_casted = power_iteration(i, u_casted, v_casted)
  u_final, v_final = u_casted, v_casted

  if update_collection is None:
    warnings.warn(
      'Setting update_collection to None will make u being updated every W execution. This maybe undesirable'
      '. Please consider using a update collection instead.')
    # sigma shape: [1, 1]
    sigma = tf.matmul(tf.matmul(v_final, W_reshaped), tf.transpose(u_final), name='spectral_norm_sigma')
    sigma = sigma[0, 0]
    W_bar = W_reshaped / sigma
    with tf.control_dependencies([tf.assign(u, u_final, name='spectral_norm_power_iter')]):
      W_bar = tf.reshape(W_bar, W.shape)
  else:
    sigma = tf.matmul(tf.matmul(v_final, W_reshaped), tf.transpose(u_final), name='spectral_norm_sigma')
    sigma = sigma[0, 0]
    W_bar = W_reshaped / sigma
    W_bar = tf.reshape(W_bar, W.shape)
    # Put NO_OPS to not update any collection.
    if update_collection != NO_OPS:
      u_final_casted_back = tf.cast(u_final, dtype=u.dtype)
      tf.add_to_collection(update_collection, tf.assign(u, u_final_casted_back, name='spectral_norm_power_iter'))
  if with_sigma:
    return W_bar, sigma
  else:
    return W_bar


#########################
# Spectral Norm classes.#
#########################
class SpecNormConv2d(convolutional_layers.Convolution2D):
  """Custom spectral norm conv2d class. The only change is to add `_spectral_normed_weight` when called."""
  def call(self, inputs):
    # Do spectral norm on the kernel.
    outputs = self._convolution_op(inputs, _spectral_normed_weight(self.kernel, update_collection=SPECTRAL_NORM_OPS))

    if self.use_bias:
      if self.data_format == 'channels_first':
        if self.rank == 1:
          # nn.bias_add does not accept a 1D input tensor.
          bias = array_ops.reshape(self.bias, (1, self.filters, 1))
          outputs += bias
        if self.rank == 2:
          outputs = tf.nn.bias_add(outputs, self.bias, data_format='NCHW')
        if self.rank == 3:
          # As of Mar 2017, direct addition is significantly slower than
          # bias_add when computing gradients. To use bias_add, we collapse Z
          # and Y into a single dimension to obtain a 4D input tensor.
          outputs_shape = outputs.shape.as_list()
          if outputs_shape[0] is None:
            outputs_shape[0] = -1
          outputs_4d = array_ops.reshape(outputs,
                                         [outputs_shape[0], outputs_shape[1],
                                          outputs_shape[2] * outputs_shape[3],
                                          outputs_shape[4]])
          outputs_4d = tf.nn.bias_add(outputs_4d, self.bias, data_format='NCHW')
          outputs = array_ops.reshape(outputs_4d, outputs_shape)
      else:
        outputs = tf.nn.bias_add(outputs, self.bias, data_format='NHWC')

    if self.activation is not None:
      return self.activation(outputs)
    return outputs


class SpectralNormedDense(layers.core_layers.Dense):
  """Spectral normalized densely-connected layer class. The only change is to add `_spectral_normed_weight`."""
  def call(self, inputs):
    inputs = tf.convert_to_tensor(inputs, dtype=self.dtype)
    shape = inputs.get_shape().as_list()
    if len(shape) > 2:
      # Broadcasting is required for the inputs.
      outputs = tf.tensordot(inputs, _spectral_normed_weight(self.kernel, update_collection=SPECTRAL_NORM_OPS),
                             [[len(shape) - 1], [0]])
      # Reshape the output back to the original ndim of the input.
      if not tf.executing_eagerly():
        output_shape = shape[:-1] + [self.units]
        outputs.set_shape(output_shape)
    else:
      outputs = gen_math_ops.mat_mul(inputs, self.kernel)
    if self.use_bias:
      outputs = tf.nn.bias_add(outputs, self.bias)
    if self.activation is not None:
      return self.activation(outputs)  # pylint: disable=not-callable
    return outputs


#########################
# Spectral Norm layers. #
#########################

# Copied from layers but adds do_spec_norm option.
# Spectral norm can also be implemented through kernel and bias constraints, but in tf 1.8 the constraints are not
# support by all getters.
@add_arg_scope
def convolution(inputs,
                num_outputs,
                kernel_size,
                stride=1,
                padding='SAME',
                data_format=None,
                rate=1,
                activation_fn=tf.nn.relu,
                normalizer_fn=None,
                normalizer_params=None,
                weights_initializer=initializers.xavier_initializer(),
                weights_regularizer=None,
                biases_initializer=tf.zeros_initializer(),
                biases_regularizer=None,
                do_spec_norm=False,
                reuse=None,
                variables_collections=None,
                outputs_collections=None,
                trainable=True,
                scope=None):
  """Adds support for spectral normalization following https://arxiv.org/abs/1802.05957.

  For non-spectral normed convolution, See tensorflow.contrib.layers.python.layers.convolution for doc.
  """
  if data_format not in [None, 'NWC', 'NCW', 'NHWC', 'NCHW', 'NDHWC', 'NCDHW']:
    raise ValueError('Invalid data_format: %r' % (data_format,))

  layer_variable_getter = layers._build_variable_getter(
    {'bias': 'biases', 'kernel': 'weights'})

  with tf.variable_scope(
      scope, 'Conv', [inputs], reuse=reuse,
      custom_getter=layer_variable_getter) as sc:
    inputs = tf.convert_to_tensor(inputs)
    input_rank = inputs.get_shape().ndims

    # ***Modified section***
    if input_rank == 3:
      layer_class = convolutional_layers.Convolution1D
      if do_spec_norm:
        raise NotImplementedError('only supports 2d conv for spectral norm.')
    elif input_rank == 4:
      layer_class = convolutional_layers.Convolution2D
      if do_spec_norm:
        layer_class = SpecNormConv2d
    elif input_rank == 5:
      layer_class = convolutional_layers.Convolution3D
      if do_spec_norm:
        raise NotImplementedError('only supports 2d conv for spectral norm.')
    else:
      raise ValueError('Convolution not supported for input with rank',
                       input_rank)
  # ***Modified section ends***

    df = ('channels_first' if data_format and data_format.startswith('NC')
          else 'channels_last')
    layer = layer_class(filters=num_outputs,
                        kernel_size=kernel_size,
                        strides=stride,
                        padding=padding,
                        data_format=df,
                        dilation_rate=rate,
                        activation=None,
                        use_bias=not normalizer_fn and biases_initializer,
                        kernel_initializer=weights_initializer,
                        bias_initializer=biases_initializer,
                        kernel_regularizer=weights_regularizer,
                        bias_regularizer=biases_regularizer,
                        activity_regularizer=None,
                        trainable=trainable,
                        name=sc.name,
                        dtype=inputs.dtype.base_dtype,
                        _scope=sc,
                        _reuse=reuse)
    outputs = layer.apply(inputs)

    # Add variables to collections.
    layers._add_variable_to_collections(layer.kernel, variables_collections, 'weights')
    if layer.use_bias:
      layers._add_variable_to_collections(layer.bias, variables_collections, 'biases')

    if normalizer_fn is not None:
      normalizer_params = normalizer_params or {}
      outputs = normalizer_fn(outputs, **normalizer_params)

    if activation_fn is not None:
      outputs = activation_fn(outputs)
    return layer_utils.collect_named_outputs(outputs_collections, sc.name, outputs)


@add_arg_scope
def fully_connected(inputs,
                    num_outputs,
                    activation_fn=tf.nn.relu,
                    normalizer_fn=None,
                    normalizer_params=None,
                    weights_initializer=initializers.xavier_initializer(),
                    weights_regularizer=None,
                    biases_initializer=tf.zeros_initializer(),
                    biases_regularizer=None,
                    do_spec_norm=False,
                    reuse=None,
                    variables_collections=None,
                    outputs_collections=None,
                    trainable=True,
                    scope=None):
  """Adds support for spectral normalization following https://arxiv.org/abs/1802.05957.

  For non-spectral normed fc layer, See tensorflow.contrib.layers.python.layers.fully_connected for doc.
  """
  # ***Added section***
  layer_class = layers.core_layers.Dense
  if do_spec_norm:
    layer_class = SpectralNormedDense
  # ***Added section ends***

  if not isinstance(num_outputs, layers.six.integer_types):
    raise ValueError(
      'num_outputs should be int or long, got %s.' % (num_outputs,))

  layer_variable_getter = layers._build_variable_getter({'bias': 'biases',
                                                         'kernel': 'weights'})

  with tf.variable_scope(
      scope, 'fully_connected', [inputs],
      reuse=reuse, custom_getter=layer_variable_getter) as sc:
    inputs = tf.convert_to_tensor(inputs)
    layer = layer_class(
      units=num_outputs,
      activation=None,
      use_bias=not normalizer_fn and biases_initializer,
      kernel_initializer=weights_initializer,
      bias_initializer=biases_initializer,
      kernel_regularizer=weights_regularizer,
      bias_regularizer=biases_regularizer,
      activity_regularizer=None,
      trainable=trainable,
      name=sc.name,
      dtype=inputs.dtype.base_dtype,
      _scope=sc,
      _reuse=reuse)
    outputs = layer.apply(inputs)

    # Add variables to collections.
    layers._add_variable_to_collections(layer.kernel, variables_collections, 'weights')
    if layer.bias is not None:
      layers._add_variable_to_collections(layer.bias, variables_collections, 'biases')

    # Apply normalizer function / layer.
    if normalizer_fn is not None:
      if not normalizer_params:
        normalizer_params = {}
      outputs = normalizer_fn(outputs, **normalizer_params)

    if activation_fn is not None:
      outputs = activation_fn(outputs)

    return layer_utils.collect_named_outputs(outputs_collections, sc.name, outputs)

