"""
Copyright 2018 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
"""Utility functions for PGGAN."""

import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

import util_misc
from libs import ops

# Flags are defined in pggan.py.
FLAGS = tf.flags.FLAGS

#############
# Constants #
#############

DEFAULT_KERNEL_SIZE = 3
DEFAULT_ACTIVATION_FN = util_misc.fp16_friendly_leaky_relu

# Norm types
BATCH_NORM_TYPE = 'batch_norm'
INSTANCE_NORM_TYPE = 'instance_norm'
BATCH_RENORM_TYPE = 'batch_renorm'
BATCH_RENORM_NATIVE_TYPE = 'batch_renorm_native'
LAYER_NORM_NATIVE_TYPE = 'layer_norm_native'
NO_NORM_TYPE = 'none'

# Note for pggan the global step restarts from 0 for every stage.
BATCH_RENORM_BOUNDARIES = [10000, 20000, 30000]
BATCH_RENORM_RMAX_VALUES = [1.1, 1.5, 2.0, 4.0]
BATCH_RENORM_RMIN_VALUES = [0.9, 0.66, 0.5, 0.25]
BATCH_RENORM_DMAX_VALUES = [0.1, 0.3, 0.5, 1.0]


##############
# Arg scopes #
##############

def pggan_arg_scope(norm_type=None, conditional_layer=None,
                    norm_var_scope_postfix='',
                    weights_init_stddev=0.02,
                    weight_decay=0.0,
                    is_training=False,
                    reuse=None):
  """

  :param norm_type: A string specifying the normalization type. See norm type constants for allowed values.
  :param conditional_layer: A tensor that the norm parameters are conditioned on.
  :param norm_var_scope_postfix: A string. Optional postfix to the normalizer variable scopes. For example if postfix
    is 'pf', then norm variable 'alpha' will become 'alpha_pf'.
  :param weights_init_stddev: A float. Standard deviation of the weight initializer.
  :param weight_decay: A float. Optional decay to apply to weights.
  :param is_training: A boolean. Used by the normalizers such as batch norm.
  :param reuse: A boolean. If set, reuse variables.
  :return:
  """
  normalizer_fn, norm_params = get_normalizer_fn_and_params(
    norm_type,
    conditional_layer=conditional_layer,
    var_scope_postfix=norm_var_scope_postfix,
    is_training=is_training,
    reuse=reuse)
  weights_regularizer = None
  if weight_decay and weight_decay > 0.0:
    weights_regularizer = layers.l2_regularizer(weight_decay)

  if FLAGS.equalized_learning_rate:
    # In equalized learning rate, weights are initialized using N(0,1), then explicitly scaled at runtime.
    weights_init_stddev = 1.0

  with tf.contrib.framework.arg_scope(
      [layers.conv2d, layers.conv2d_transpose, ops.convolution],
      # Note: the activation is added after the normalizer_fn.
      activation_fn=DEFAULT_ACTIVATION_FN,
      # Note: In the PGGAN paper pixel norm is used for generator and no normalization is used for the discriminator.
      normalizer_fn=normalizer_fn,
      normalizer_params=norm_params,
      weights_initializer=tf.random_normal_initializer(0, weights_init_stddev),
      weights_regularizer=weights_regularizer,
      stride=1,
      kernel_size=DEFAULT_KERNEL_SIZE,
      padding='SAME',
      reuse=reuse) as sc:
    return sc


def pggan_generator_arg_scope(norm_type=None,
                              conditional_layer=None,
                              conditional_layer_var_scope_postfix='',
                              weights_init_stddev=0.02,
                              weight_decay=0.0,
                              is_training=False,
                              reuse=None):
  """Wrapper around `pggan_arg_scope` for the generator/encoder. The difference is in the `norm_type`."""
  return pggan_arg_scope(norm_type=norm_type, conditional_layer=conditional_layer,
                         norm_var_scope_postfix=conditional_layer_var_scope_postfix,
                         weights_init_stddev=weights_init_stddev, weight_decay=weight_decay,
                         is_training=is_training, reuse=reuse)


def pggan_discriminator_arg_scope(norm_type=NO_NORM_TYPE,
                                  conditional_layer=None,
                                  conditional_layer_var_scope_postfix='',
                                  weights_init_stddev=0.02,
                                  weight_decay=0.0,
                                  is_training=False,
                                  reuse=None):
  """Wrapper around `pggan_arg_scope` for the discriminator. The difference is in the `norm_type`."""
  return pggan_arg_scope(norm_type=norm_type, conditional_layer=conditional_layer,
                         norm_var_scope_postfix=conditional_layer_var_scope_postfix,
                         weights_init_stddev=weights_init_stddev, weight_decay=weight_decay,
                         is_training=is_training, reuse=reuse)


###############
# Norm Params #
###############

def get_normalizer_fn_and_params(norm_type=None,
                                 conditional_layer=None,
                                 var_scope_postfix='',
                                 is_training=False,
                                 reuse=None):
  """Helper function to return the normalization function specified by `norm_type`."""
  if norm_type is None:
    norm_type = FLAGS.generator_norm_type

  if norm_type == BATCH_NORM_TYPE:
    normalizer_fn = ops.conditional_batch_norm
    norm_params = {
      'center': True,
      'scale': True,
      'is_training': is_training,
      'reuse': reuse,
      'var_scope_postfix': var_scope_postfix,
      'conditional_layer': conditional_layer,
    }
  elif norm_type == INSTANCE_NORM_TYPE:
    normalizer_fn = ops.instance_norm
    norm_params = {
      'center': True,
      'scale': True,
      'reuse': reuse,
      'var_scope_postfix': var_scope_postfix,
      'conditional_layer': conditional_layer,
    }
  elif norm_type == BATCH_RENORM_TYPE:
    normalizer_fn = ops.conditional_batch_norm
    norm_params = {
      'decay': 0.99,  # Set to be the same as renorm decay.
      'center': True,
      'scale': True,
      'is_training': is_training,
      'reuse': reuse,
      'renorm': True,
      'renorm_clipping': get_renorm_clipping_params(),
      'var_scope_postfix': var_scope_postfix,
      'conditional_layer': conditional_layer,
    }
  elif norm_type == BATCH_RENORM_NATIVE_TYPE:
    tf.logging.log_every_n(tf.logging.INFO, 'Using tensorflow native implementation of batch renorm.', 100)
    assert conditional_layer is None, ('Tensorflow implementation does not support `conditional_layer`.')
    normalizer_fn = layers.batch_norm
    norm_params = {
      'decay': 0.99,  # Set to be the same as renorm decay because inference quality is too poor.
      'center': True,
      'scale': True,
      'is_training': is_training,
      'reuse': reuse,
      'renorm': True,
      'renorm_clipping': get_renorm_clipping_params(),
      'scope': var_scope_postfix
    }
  elif norm_type == LAYER_NORM_NATIVE_TYPE:
    tf.logging.log_every_n(tf.logging.INFO, 'Using tensorflow native implementation of layer norm.', 100)
    assert conditional_layer is None, ('Tensorflow implementation does not support `conditional_layer`.')
    normalizer_fn = layers.layer_norm
    norm_params = {
      'center': True,
      'scale': True,
      'reuse': reuse,
      'scope': var_scope_postfix
    }
  elif not norm_type or norm_type == NO_NORM_TYPE:
    normalizer_fn = None
    norm_params = None
  else:
    raise NotImplementedError('unsupported norm type: %s' % norm_type)
  return normalizer_fn, norm_params


def get_renorm_clipping_params():
  """Returns a dictionary containing batch renorm parameters, which are dependent on the global step."""
  global_step = tf.train.get_global_step()
  if global_step is not None:
    # Start with 1.0/0.0 and slowly relax them.
    rmax = tf.train.piecewise_constant(global_step, boundaries=BATCH_RENORM_BOUNDARIES, values=BATCH_RENORM_RMAX_VALUES,
                                       name='rmax')
    rmin = tf.train.piecewise_constant(global_step, boundaries=BATCH_RENORM_BOUNDARIES, values=BATCH_RENORM_RMIN_VALUES,
                                       name='rmin')
    dmax = tf.train.piecewise_constant(global_step, boundaries=BATCH_RENORM_BOUNDARIES, values=BATCH_RENORM_DMAX_VALUES,
                                       name='dmax')
  else:
    tf.logging.warning('Cannot find global step. Falling back to default renorm clipping parameter values.')
    rmax = BATCH_RENORM_RMAX_VALUES[-1]
    rmin = BATCH_RENORM_RMIN_VALUES[-1]
    dmax = BATCH_RENORM_DMAX_VALUES[-1]
  return {'rmax': rmax, 'rmin': rmin, 'dmax': dmax}


#################################################################
# Helper functions for convolutional and fully connected layers #
#################################################################

# Functions used by pggan.py.
def maybe_pixel_norm(layer, do_pixel_norm):
  """Applies pixel norm if `do_pixel_norm` is set. Otherwise acts as identity function."""
  return _pixel_norm(layer) if do_pixel_norm else layer


def maybe_equalized_conv2d(inputs, num_outputs, kernel_size=DEFAULT_KERNEL_SIZE, is_discriminator=False, **kwargs):
  """If `equalized_learning_rate` flag is set, applies equalized lr before conv2d."""
  if FLAGS.equalized_learning_rate:
    in_ch = int(inputs.shape[-1])
    # Parameters from variance_scaling_initializer, MSRA initialization aka. Kaiming Initialization.
    # trunc_stddev = math.sqrt(1.3 * 2.0 / in_ch)
    inv_c = np.sqrt(2.0 / (in_ch * kernel_size ** 2))
    inputs = inv_c * inputs
  return _maybe_spectral_normed_conv(inputs, num_outputs=num_outputs, kernel_size=kernel_size,
                                     is_discriminator=is_discriminator, **kwargs)


def maybe_equalized_fc(inputs, num_outputs, is_discriminator=False, **kwargs):
  """If `equalized_learning_rate` flag is set, applies equalized lr before fc."""
  if FLAGS.equalized_learning_rate:
    in_ch = int(inputs.shape[-1])
    inv_c = np.sqrt(2.0 / in_ch)
    inputs = inv_c * inputs
  return _maybe_spectral_normed_fully_connected(inputs, num_outputs, is_discriminator=is_discriminator, **kwargs)


def maybe_resblock(input_layer, out_channels, conv2d_out, is_discriminator=False):
  """If `use_res_block` flag is set, add a residule layer shortcut to conv2d_out."""
  if FLAGS.use_res_block:
    shortcut = _get_resblock_shortcut(input_layer, out_channels, is_discriminator=is_discriminator)
    ret = shortcut + conv2d_out
  else:
    ret = conv2d_out
  return ret


def maybe_concat_conditional_layer(layer, conditional_layer):
  """If `conditional_layer` is not None, reshape and concatenate it to `layer`."""
  if conditional_layer is not None:
    # Resize conditional layer to the same height and width as layer.
    assert len(conditional_layer.shape) == 4
    resized_conditional_layer = conditional_layer
    # Bilinear is differentiable.
    resized_conditional_layer = tf.image.resize_bilinear(resized_conditional_layer, layer.shape[1:3])
    resized_conditional_layer = tf.cast(resized_conditional_layer, conditional_layer.dtype)
    return tf.concat((layer, resized_conditional_layer), axis=-1)
  else:
    return layer


def maybe_concat_unet_layer(layer, unet_end_points):
  """If `unet_end_points` is not None, finds the corresponding unet layer and concatenate it to `layer`."""
  if unet_end_points is None:
    return layer
  # Assume h = w.
  hw = int(layer.shape[1])
  # If `pggan_unet_max_concat_hw` flag is specified, do not concatenate if hw is larger than that.
  if FLAGS.pggan_unet_max_concat_hw and hw > FLAGS.pggan_unet_max_concat_hw:
    return layer

  max_stage = int(np.log2(hw)) - 2
  num_channels = get_num_channels(max_stage - 1)
  unet_layer_name = 'encoder_block_interpolated_%dx%dx%d' % (hw, hw, num_channels)
  if unet_layer_name not in unet_end_points:
    unet_layer_name = 'encoder_block_%dx%dx%d' % (hw, hw, num_channels)
  if unet_layer_name not in unet_end_points:
    raise ValueError('%s not in unet_end_points' % (unet_layer_name))
  return tf.concat((layer, unet_end_points[unet_layer_name]), axis=-1)


def maybe_add_self_attention(do_self_attention, self_attention_hw, hw, channels, net, end_points):
  """If hw==self_attention_hw, adds a self attention module. See SAGAN for details."""
  if do_self_attention and hw == self_attention_hw:
    scope_name = 'self_attention_%dx%dx%d' % (hw, hw, channels)
    with tf.variable_scope(scope_name):
      net = ops.self_attention_layer(net)
      end_points[scope_name] = net
  return net


# Internal functions

# Although in the spectral norm paper, the spectral norm was applied to the discriminator, because it's main purpose was
# to make the discriminator 1-Lipschitz continuous. However some recent paper like SAGAN found that applying spectral
# norm to generator also helps. Thus the `is_discriminator` variable and the `spectral_norm_in_non_discriminator` flag.
def _maybe_spectral_normed_conv(inputs, num_outputs, kernel_size, is_discriminator=False, **kwargs):
  if FLAGS.spectral_norm and (is_discriminator or FLAGS.spectral_norm_in_non_discriminator):
    return ops.spectral_normed_conv(inputs, num_outputs=num_outputs, kernel_size=kernel_size, **kwargs)
  else:
    return layers.conv2d(inputs, num_outputs=num_outputs, kernel_size=kernel_size, **kwargs)


def _maybe_spectral_normed_fully_connected(inputs, num_outputs, is_discriminator=False, **kwargs):
  if FLAGS.spectral_norm and (is_discriminator or FLAGS.spectral_norm_in_non_discriminator):
    return ops.spectral_normed_fc(inputs, num_outputs=num_outputs, **kwargs)
  else:
    return layers.fully_connected(inputs, num_outputs=num_outputs, **kwargs)


def _pixel_norm(input, eps=1e-6):
  return input / tf.sqrt(tf.reduce_mean(tf.square(input), axis=3, keep_dims=True) + tf.constant(eps, dtype=input.dtype))


def _get_resblock_shortcut(input_layer, out_channels, is_discriminator=False):
  in_channels = int(input_layer.shape[-1])  # Must have a known channel size.
  if out_channels == in_channels:
    shortcut = input_layer
  else:
    shortcut = maybe_equalized_conv2d(
      input_layer, out_channels, is_discriminator=is_discriminator, kernel_size=1, normalizer_fn=None,
      activation_fn=None, scope='shortcut')
  return shortcut


##########################
# Other helper functions #
##########################

def resize_twice_as_big(input_layer):
  return tf.image.resize_nearest_neighbor(input_layer, (input_layer.shape[1] * 2, input_layer.shape[2] * 2))


def minibatch_state_concat(input, averaging='all'):
  # Injects "the across-minibatch standard deviation as an additional feature map at
  # 4 x 4 resolution toward the end of the discriminator as described in Section 3"
  # (from PGGAN paper).
  adjusted_std = lambda x, **kwargs: tf.sqrt(
    tf.reduce_mean((x - tf.reduce_mean(x, **kwargs)) ** 2, **kwargs)
    + tf.constant(1e-8 if x.dtype == tf.float32 else 1e-6, dtype=x.dtype))
  vals = adjusted_std(input, axis=0, keepdims=True)
  if averaging == 'all':
    vals = tf.reduce_mean(vals, keep_dims=True)
  else:
    raise NotImplementedError('averaging method not supported.')
  vals = tf.tile(vals, multiples=[input.shape[0], 4, 4, 1])
  return tf.concat([input, vals], axis=3)


def get_num_channels(stage, max_num_channels=None):
  if max_num_channels is None:
    max_num_channels = FLAGS.pggan_max_num_channels
  return min(1024 / (2 ** stage), max_num_channels)


def get_discriminator_max_num_channels():
  if FLAGS.pggan_max_num_channels_dis:
    max_num_channels = FLAGS.pggan_max_num_channels_dis
  else:
    max_num_channels = FLAGS.pggan_max_num_channels
  return max_num_channels
