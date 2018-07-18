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
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import add_arg_scope

from tensorflow.contrib.framework.python.ops import variables
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.framework import ops
from batch_norm import get_conditional_batch_norm_param


DATA_FORMAT_NCHW = 'NCHW'
DATA_FORMAT_NHWC = 'NHWC'
DATA_FORMAT_NCDHW = 'NCDHW'
DATA_FORMAT_NDHWC = 'NDHWC'


# Copied and modified from layers.instance_norm to support the optional `conditional_layer` and `var_scope_postfix`.
@add_arg_scope
def instance_norm(inputs,
                  conditional_layer,
                  var_scope_postfix='',
                  center=True,
                  scale=True,
                  epsilon=1e-6,
                  activation_fn=None,
                  param_initializers=None,
                  reuse=None,
                  variables_collections=None,
                  outputs_collections=None,
                  trainable=True,
                  data_format=DATA_FORMAT_NHWC,
                  scope=None):
  """Custom implementation of instance norm  to support the optional `conditional_layer` and `var_scope_postfix`.
  For comments on the other parameters, see tensorflow.contrib.layers.python.layers.batch_norm, where this is copied
  from (tf 1.5 version).

  Args:
    conditional_layer: A tensor with 2 dimensions [batch, channels]. If not None, the beta and gamma parameters will
      be conditioned on the `conditional_layer`.
    var_scope_postfix: A string. Append it to the var scopes of all variables other than the weight and bias. e.g.
      var scope of the `gamma` variable becomes `'gamma' + var_scope_postfix`.
  """
  inputs = ops.convert_to_tensor(inputs)
  inputs_shape = inputs.shape
  inputs_rank = inputs.shape.ndims

  if inputs_rank is None:
    raise ValueError('Inputs %s has undefined rank.' % inputs.name)
  if data_format not in (DATA_FORMAT_NCHW, DATA_FORMAT_NHWC):
    raise ValueError('data_format has to be either NCHW or NHWC.')

  with tf.variable_scope(
      scope, 'InstanceNorm', [inputs], reuse=reuse) as sc:
    if data_format == DATA_FORMAT_NCHW:
      reduction_axis = 1
      # For NCHW format, rather than relying on implicit broadcasting, we
      # explicitly reshape the params to params_shape_broadcast when computing
      # the moments and the batch normalization.
      params_shape_broadcast = list(
          [1, inputs_shape[1].value] + [1 for _ in range(2, inputs_rank)])
    else:
      reduction_axis = inputs_rank - 1
      params_shape_broadcast = None
    moments_axes = list(range(inputs_rank))
    del moments_axes[reduction_axis]
    del moments_axes[0]
    params_shape = inputs_shape[reduction_axis:reduction_axis + 1]
    if not params_shape.is_fully_defined():
      raise ValueError('Inputs %s has undefined channels dimension %s.' % (
          inputs.name, params_shape))

    # Allocate parameters for the beta and gamma of the normalization.
    beta, gamma = None, None
    dtype = inputs.dtype.base_dtype
    if param_initializers is None:
      param_initializers = {}

    if center:
      beta_scope = 'beta' + var_scope_postfix
      if conditional_layer is not None:
        beta = get_conditional_batch_norm_param(conditional_layer, int(params_shape[-1]), scope=beta_scope)
      else:
        # Behaves like normal batch norm.
        beta_collections = utils.get_variable_collections(variables_collections,
                                                          beta_scope)
        beta_initializer = param_initializers.get(beta_scope,
                                                  tf.zeros_initializer())
        beta = variables.model_variable(beta_scope,
                                        shape=params_shape,
                                        dtype=dtype,
                                        initializer=beta_initializer,
                                        collections=beta_collections,
                                        trainable=trainable)
      if params_shape_broadcast:
        beta = tf.reshape(beta, params_shape_broadcast)

    if scale:
      gamma_scope = 'gamma' + var_scope_postfix
      if conditional_layer is not None:
        # Per https://arxiv.org/pdf/1707.03017.pdf.
        delta_gamma = get_conditional_batch_norm_param(conditional_layer, int(params_shape[-1]), scope=gamma_scope)
        gamma = tf.constant(1.0, dtype=dtype, ) + delta_gamma
      else:
        gamma_collections = utils.get_variable_collections(variables_collections,
                                                           gamma_scope)
        gamma_initializer = param_initializers.get(gamma_scope,
                                                   tf.ones_initializer())
        gamma = variables.model_variable(gamma_scope,
                                         shape=params_shape,
                                         dtype=dtype,
                                         initializer=gamma_initializer,
                                         collections=gamma_collections,
                                         trainable=trainable)
      if params_shape_broadcast:
        gamma = tf.reshape(gamma, params_shape_broadcast)

    # Calculate the moments (instance activations).
    mean, variance = tf.nn.moments(inputs, moments_axes, keep_dims=True)

    # Compute instance normalization.
    outputs = tf.nn.batch_normalization(
        inputs, mean, variance, beta, gamma, epsilon, name='instancenorm')
    if activation_fn is not None:
      outputs = activation_fn(outputs)
    return utils.collect_named_outputs(outputs_collections, sc.name, outputs)