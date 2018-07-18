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
"""Custom implementation of batch norm function with support for batch renorm and conditional batch norm."""

import tensorflow as tf
from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.training import moving_averages

SPECTRAL_NORM_OPS = 'spectral_norm_scope'  # tf.GraphKeys.UPDATE_OPS
DATA_FORMAT_NCHW = 'NCHW'
DATA_FORMAT_NHWC = 'NHWC'
DATA_FORMAT_NCDHW = 'NCDHW'
DATA_FORMAT_NDHWC = 'NDHWC'


def get_conditional_batch_norm_param(conditional_layer, output_dim, scope='gamma', activation_fn=None):
  """Outputs the batch norm parameter transformed from the `conditional_layer` using a fully connected layer."""
  if conditional_layer is None:
    raise ValueError('`conditional_layer` must not be None.')
  return layers.fully_connected(conditional_layer, output_dim, scope=scope, activation_fn=activation_fn)


@add_arg_scope
def conditional_batch_norm(inputs,
                           conditional_layer,
                           var_scope_postfix='',
                           decay=0.999,
                           center=True,
                           scale=False,
                           epsilon=0.001,
                           activation_fn=None,
                           param_initializers=None,
                           param_regularizers=None,
                           updates_collections=tf.GraphKeys.UPDATE_OPS,
                           is_training=True,
                           reuse=None,
                           variables_collections=None,
                           outputs_collections=None,
                           trainable=True,
                           data_format=DATA_FORMAT_NHWC,
                           zero_debias_moving_mean=False,
                           renorm=False,
                           renorm_clipping=None,
                           renorm_momentum=0.99,
                           scope=None):
  """Custom implementation of batch norm  to support the optional `conditional_layer` and `var_scope_postfix`.
  For comments on the other parameters, see tensorflow.contrib.layers.python.layers.batch_norm, where this is copied
  from (tf 1.5 version).

  Args:
    conditional_layer: A tensor with 2 dimensions [batch, channels]. If not None, the beta and gamma parameters will
      be conditioned on the `conditional_layer`.
    var_scope_postfix: A string. Append it to the var scopes of all variables other than the weight and bias. e.g.
      var scope of the `gamma` variable becomes `'gamma' + var_scope_postfix`.
  """

  if data_format not in (DATA_FORMAT_NCHW, DATA_FORMAT_NHWC):
    raise ValueError('data_format has to be either NCHW or NHWC.')
  if inputs.dtype != tf.float32:
    raise NotImplementedError('This implementation may not be compatible with mixed precision training.')
  with tf.variable_scope(
      scope, 'BatchNorm', [inputs], reuse=reuse) as sc:

    if conditional_layer is not None:
      conditional_layer = tf.convert_to_tensor(conditional_layer)
      # Normalizing the conditional layer seems to stabilize training a little.
      conditional_layer = tf.nn.l2_normalize(conditional_layer, dim=1, name='normalized_conditional_layer')
      conditional_layer_shape = conditional_layer.get_shape()
      conditional_layer_rank = conditional_layer_shape.ndims
      if conditional_layer_rank is None:
        raise ValueError('Conditional layer %s has undefined rank' % conditional_layer.name)
      elif conditional_layer_rank != 2:
        raise ValueError('Conditional layer %s is not rank 2.' % conditional_layer.name)

    inputs = tf.convert_to_tensor(inputs)
    original_shape = inputs.get_shape()
    original_inputs = inputs
    original_rank = original_shape.ndims
    if original_rank is None:
      raise ValueError('Inputs %s has undefined rank' % inputs.name)
    elif original_rank not in [2, 4]:
      raise ValueError('Inputs %s has unsupported rank.'
                       ' Expected 2 or 4 but got %d' % (
                         inputs.name, original_rank))
    if original_rank == 2:
      channels = inputs.get_shape()[-1].value
      if channels is None:
        raise ValueError('`C` dimension must be known but is None')
      new_shape = [-1, 1, 1, channels]
      if data_format == DATA_FORMAT_NCHW:
        new_shape = [-1, channels, 1, 1]
      inputs = tf.reshape(inputs, new_shape)
    inputs_shape = inputs.get_shape()
    if data_format == DATA_FORMAT_NHWC:
      params_shape = inputs_shape[-1:]
    else:
      params_shape = inputs_shape[1:2]
    if not params_shape.is_fully_defined():
      raise ValueError('Inputs %s has undefined `C` dimension %s.' %
                       (inputs.name, params_shape))

    # Allocate parameters for the beta and gamma of the normalization.
    beta_collections = utils.get_variable_collections(variables_collections,
                                                      'beta')
    variable_dtype = inputs.dtype
    if not param_initializers:
      param_initializers = {}
    if not param_regularizers:
      param_regularizers = {}

    if center:
      beta_scope = 'beta' + var_scope_postfix
      if conditional_layer is not None:
        assert not param_initializers, 'param_initializers are not supported with conditional layer.'
        assert not param_regularizers, 'param_initializers are not supported with conditional layer.'
        beta = get_conditional_batch_norm_param(conditional_layer, int(params_shape[-1]), scope=beta_scope)
      else:
        # Behaves like normal batch norm.
        beta_collections = utils.get_variable_collections(variables_collections,
                                                          beta_scope)
        beta_initializer = param_initializers.get(beta_scope,
                                                  tf.zeros_initializer())
        beta_regularizer = param_regularizers.get('beta')
        beta = variables.model_variable(beta_scope,
                                        shape=params_shape,
                                        dtype=variable_dtype,
                                        initializer=beta_initializer,
                                        regularizer=beta_regularizer,
                                        collections=beta_collections,
                                        trainable=trainable)
    else:
      beta = array_ops.constant(0.0, dtype=variable_dtype, shape=params_shape)

    if scale:
      gamma_scope = 'gamma' + var_scope_postfix
      if conditional_layer is not None:
        assert not param_initializers, 'param_initializers are not supported with conditional layer.'
        assert not param_regularizers, 'param_initializers are not supported with conditional layer.'
        delta_gamma = get_conditional_batch_norm_param(conditional_layer, int(params_shape[-1]), scope=gamma_scope)
        # Per https://arxiv.org/pdf/1707.03017.pdf.
        gamma = tf.constant(1.0, dtype=variable_dtype, ) + delta_gamma
      else:
        gamma_collections = utils.get_variable_collections(variables_collections,
                                                           gamma_scope)
        gamma_initializer = param_initializers.get(gamma_scope,
                                                   tf.ones_initializer())
        gamma_regularizer = param_regularizers.get('gamma')
        gamma = variables.model_variable(gamma_scope,
                                         shape=params_shape,
                                         dtype=variable_dtype,
                                         initializer=gamma_initializer,
                                         regularizer=gamma_regularizer,
                                         collections=gamma_collections,
                                         trainable=trainable)
    else:
      gamma = tf.constant(1.0, dtype=variable_dtype, shape=params_shape)

    # Create moving_mean and moving_variance variables and add them to the
    # appropriate collections. We disable variable partitioning while creating
    # them, because assign_moving_average is not yet supported for partitioned
    # variables (this needs to be handled carefully, as it may break
    # the checkpoint backward compatibility).
    with tf.variable_scope(
        tf.get_variable_scope()) as local_scope:
      local_scope.set_partitioner(None)
      moving_mean_scope = 'moving_mean' + var_scope_postfix
      moving_mean_collections = utils.get_variable_collections(
        variables_collections, moving_mean_scope)
      moving_mean_initializer = param_initializers.get(
        moving_mean_scope, tf.zeros_initializer())
      moving_mean = variables.model_variable(
        moving_mean_scope,
        shape=params_shape,
        dtype=tf.float32,
        initializer=moving_mean_initializer,
        trainable=False,
        collections=moving_mean_collections)
      moving_variance_scope = 'moving_variance' + var_scope_postfix
      moving_variance_collections = utils.get_variable_collections(
        variables_collections, moving_variance_scope)
      moving_variance_initializer = param_initializers.get(
        moving_variance_scope, tf.ones_initializer())
      moving_variance = variables.model_variable(
        moving_variance_scope,
        shape=params_shape,
        dtype=tf.float32,
        initializer=moving_variance_initializer,
        trainable=False,
        collections=moving_variance_collections)

      if renorm:
        renorm_clipping = renorm_clipping or {}
        keys = ['rmax', 'rmin', 'dmax']
        if set(renorm_clipping) - set(keys):
          raise ValueError('renorm_clipping %s contains keys not in %s' %
                           (renorm_clipping, keys))

        # Create variables to maintain the moving mean and standard deviation.
        # These are used in training and thus are different from the moving
        # averages above. The renorm variables are colocated with moving_mean
        # and moving_variance.
        # NOTE: below, the outer `with device` block causes the current device
        # stack to be cleared. The nested ones use a `lambda` to set the desired
        # device and ignore any devices that may be set by the custom getter.
        def _renorm_variable(name, shape):
          var = variables.model_variable(
            name=name,  # renorm variable should be dependent on var_scope_postfix.
            shape=shape,
            dtype=tf.float32,
            initializer=param_initializers.get(name, tf.zeros_initializer()),
            trainable=False)
          return var

        with ops.device(None):
          device = ((lambda _: moving_mean.device)
                    if context.executing_eagerly() else moving_mean.device)
          with ops.device(device):
            renorm_mean = _renorm_variable('renorm_mean' + var_scope_postfix, params_shape)
            renorm_mean_weight = _renorm_variable('renorm_mean_weight' + var_scope_postfix, ())
          # We initialize renorm_stddev to 0, and maintain the (0-initialized)
          # renorm_stddev_weight. This allows us to (1) mix the average
          # stddev with the minibatch stddev early in training, and (2) compute
          # the unbiased average stddev by dividing renorm_stddev by the weight.
          device = ((lambda _: moving_variance.device)
                    if context.executing_eagerly() else moving_variance.device)
          with ops.device(device):
            renorm_stddev = _renorm_variable('renorm_stddev' + var_scope_postfix, params_shape)
            renorm_stddev_weight = _renorm_variable('renorm_stddev_weight' + var_scope_postfix, ())

        class dotdict(dict):
          """dot.notation access to dictionary attributes"""
          __getattr__ = dict.get
          __setattr__ = dict.__setitem__
          __delattr__ = dict.__delitem__

        renorm_params = dotdict({'renorm_mean': renorm_mean, 'renorm_mean_weight': renorm_mean_weight,
                                 'renorm_stddev': renorm_stddev, 'renorm_stddev_weight': renorm_stddev_weight,
                                 'renorm_clipping': renorm_clipping, 'renorm_momentum': renorm_momentum,
                                 'moving_mean': moving_mean, 'moving_variance': moving_variance, 'epsilon': epsilon})
      else:
        renorm_params = None

    def _batch_norm_training():
      # return tf.nn.fused_batch_norm(
      return _batch_norm_aux(
        inputs, gamma, beta, epsilon=epsilon, data_format=data_format, renorm=renorm, renorm_params=renorm_params)

    def _batch_norm_inference():
      # return tf.nn.fused_batch_norm(
      return _batch_norm_aux(
        inputs,
        gamma,
        beta,
        mean=tf.cast(moving_mean, dtype=variable_dtype),
        variance=tf.cast(moving_variance, dtype=variable_dtype),
        epsilon=epsilon,
        is_training=False,
        data_format=data_format,
        renorm=renorm,
        renorm_params=renorm_params)

    outputs, mean, variance = utils.smart_cond(is_training,
                                               _batch_norm_training,
                                               _batch_norm_inference)

    # If `is_training` doesn't have a constant value, because it is a `Tensor`,
    # a `Variable` or `Placeholder` then is_training_value will be None and
    # `need_updates` will be true.
    is_training_value = utils.constant_value(is_training)
    need_updates = is_training_value is None or is_training_value
    if need_updates:
      if updates_collections is None:
        no_updates = lambda: outputs

        def _force_updates():
          """Internal function forces updates moving_vars if is_training."""
          update_moving_mean = moving_averages.assign_moving_average(
            moving_mean, mean, decay, zero_debias=zero_debias_moving_mean)
          update_moving_variance = moving_averages.assign_moving_average(
            moving_variance, variance, decay, zero_debias=False)
          with tf.control_dependencies(
              [update_moving_mean, update_moving_variance]):
            return tf.identity(outputs)

        outputs = utils.smart_cond(is_training, _force_updates, no_updates)
      else:
        moving_vars_fn = lambda: (moving_mean, moving_variance)

        def _delay_updates():
          """Internal function that delay updates moving_vars if is_training."""
          update_moving_mean = moving_averages.assign_moving_average(
            moving_mean, tf.cast(mean, dtype=moving_mean.dtype), decay, zero_debias=zero_debias_moving_mean)
          update_moving_variance = moving_averages.assign_moving_average(
            moving_variance, tf.cast(variance, dtype=moving_variance.dtype), decay, zero_debias=False)
          return update_moving_mean, update_moving_variance

        update_mean, update_variance = utils.smart_cond(is_training,
                                                        _delay_updates,
                                                        moving_vars_fn)
        ops.add_to_collections(updates_collections, update_mean)
        ops.add_to_collections(updates_collections, update_variance)

    outputs.set_shape(inputs_shape)
    if original_shape.ndims == 2:
      outputs = array_ops.reshape(outputs, array_ops.shape(original_inputs))
    if activation_fn is not None:
      outputs = activation_fn(outputs)
    return utils.collect_named_outputs(outputs_collections, sc.name, outputs)


def _renorm_correction_and_moments(renorm_params, mean, variance, training, ):
  """Returns the correction and update values for renorm."""
  stddev = tf.sqrt(variance + renorm_params.epsilon)
  # Compute the average mean and standard deviation, as if they were
  # initialized with this batch's moments.
  mixed_renorm_mean = (renorm_params.renorm_mean +
                       (1. - renorm_params.renorm_mean_weight) * mean)
  mixed_renorm_stddev = (renorm_params.renorm_stddev +
                         (1. - renorm_params.renorm_stddev_weight) * stddev)
  # Compute the corrections for batch renorm.
  r = stddev / mixed_renorm_stddev
  d = (mean - mixed_renorm_mean) / mixed_renorm_stddev
  # Ensure the corrections use pre-update moving averages.
  with ops.control_dependencies([r, d]):
    mean = array_ops.identity(mean)
    stddev = array_ops.identity(stddev)
  rmin, rmax, dmax = [renorm_params.renorm_clipping.get(key)
                      for key in ['rmin', 'rmax', 'dmax']]
  if rmin is not None:
    r = tf.maximum(r, rmin)
  if rmax is not None:
    r = tf.minimum(r, rmax)
  if dmax is not None:
    d = tf.maximum(d, -dmax)
    d = tf.minimum(d, dmax)
  # When not training, use r=1, d=0.
  r = utils.smart_cond(training, lambda: r, lambda: array_ops.ones_like(r))
  d = utils.smart_cond(training, lambda: d, lambda: array_ops.zeros_like(d))

  def _update_renorm_variable(var, weight, value):
    """Updates a moving average and weight, returns the unbiased value."""
    value = array_ops.identity(value)

    def _do_update():
      # Update the variables without zero debiasing. The debiasing will be
      # accomplished by dividing the exponential moving average by the weight.
      # For example, after a single update, the moving average would be
      # (1-decay) * value. and the weight will be 1-decay, with their ratio
      # giving the value.
      # Make sure the weight is not updated until before r and d computation.
      with ops.control_dependencies([value]):
        weight_value = array_ops.constant(1., dtype=weight.dtype)
      new_var = moving_averages.assign_moving_average(
        var, value, renorm_params.renorm_momentum, zero_debias=False)
      new_weight = moving_averages.assign_moving_average(
        weight, weight_value, renorm_params.renorm_momentum, zero_debias=False)
      return new_var / new_weight

    def _fake_update():
      return array_ops.identity(var)

    return utils.smart_cond(training, _do_update, _fake_update)

  with ops.colocate_with(renorm_params.moving_mean):
    new_mean = _update_renorm_variable(renorm_params.renorm_mean,
                                       renorm_params.renorm_mean_weight,
                                       mean)
  with ops.colocate_with(renorm_params.moving_variance):
    new_stddev = _update_renorm_variable(renorm_params.renorm_stddev,
                                         renorm_params.renorm_stddev_weight,
                                         stddev)
    # Make sqrt(moving_variance + epsilon) = new_stddev.
    new_variance = tf.square(new_stddev) - renorm_params.epsilon

  return (r, d, new_mean, new_variance)


def _batch_norm_aux(
    x,
    scale,
    offset,  # pylint: disable=invalid-name
    mean=None,
    variance=None,
    epsilon=0.001,
    data_format="NHWC",
    renorm=False,
    renorm_params=None,
    is_training=True,
    name=None):
  r"""A batch-renorm friendly version of batch normalization."""
  assert data_format=="NHWC"
  def expand_to_4d(tensor):
    if len(tensor.shape) == 1:  # (num_features)
      tensor = tf.expand_dims(tf.expand_dims(tf.expand_dims(tensor, 0), 0), 0)
    elif len(tensor.shape) == 2:  # (batch_size, num_features)
      tensor = tf.expand_dims(tf.expand_dims(tensor, 1), 1)
    else:
      raise NotImplementedError
    return tensor

  x = ops.convert_to_tensor(x, name="input")
  scale = ops.convert_to_tensor(scale, name="scale")
  offset = ops.convert_to_tensor(offset, name="offset")
  scale = expand_to_4d(scale)
  offset = expand_to_4d(offset)

  if is_training:
    if (mean is not None) or (variance is not None):
      raise ValueError("Both 'mean' and 'variance' must be None "
                       "if is_training is True.")

  moments = tf.nn.moments(x, axes=[0, 1, 2])

  if mean is None:
    mean = moments[0]

  if variance is None:
    variance = moments[1]

  def _compose_transforms(scale, offset, then_scale, then_offset):
    if then_scale is not None:
      scale *= then_scale
      offset *= then_scale
    if then_offset is not None:
      offset += then_offset
    return (scale, offset)

  if is_training:
    if renorm:
      # The input mean and variance will be moving mean and variance for inference.
      r, d, new_mean, new_variance = _renorm_correction_and_moments(renorm_params,
                                                                    mean, variance, is_training, )
      # When training, the normalized values (say, x) will be transformed as
      # x * gamma + beta without renorm, and (x * r + d) * gamma + beta
      # = x * (r * gamma) + (d * gamma + beta) with renorm.
      # In the original implementation under layers.normalization, there is a _broadcast() function around
      # stop_gradient, which is identity for our use case.
      r = array_ops.stop_gradient(r, name='renorm_r')
      d = array_ops.stop_gradient(d, name='renorm_d')
      scale, offset = _compose_transforms(r, d, scale, offset)
    else:
      new_mean, new_variance = mean, variance
  else:
    new_mean, new_variance = mean, variance

  # Set a minimum epsilon to 1.001e-5, which is a requirement by CUDNN to
  # prevent exception (see cudnn.h).
  min_epsilon = 1.001e-5
  epsilon = epsilon if epsilon > min_epsilon else min_epsilon
  epsilon = tf.constant(epsilon, dtype=x.dtype)
  # The mean and variance is used for batch norm. The new_mean and new_variance are used for updating the moving mean/var.
  return tf.nn.batch_normalization(x, mean, variance, offset, scale, epsilon), new_mean, new_variance
