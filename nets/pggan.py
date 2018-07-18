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
import numpy as np
import tensorflow as tf

import pggan_utils
import util_misc
from libs import ops

# Normalization and optimization flags.
tf.flags.DEFINE_string('generator_norm_type', 'batch_norm',
                       'The default type of normalization to be used by the genrator. '
                       'Note that this does NOT include pixel norm or spectral norm. '
                       'See pggan_utils.py for allowed values.')
tf.flags.DEFINE_boolean(
  'spectral_norm', False,
  'If true, use spectral_norm as a normalization method in addition to the norm specified in `generator_norm_type`.')
tf.flags.DEFINE_boolean(
  'spectral_norm_in_non_discriminator', False,
  'If true, use spectral_norm in generator and encoders as well.')
tf.flags.DEFINE_boolean(
  'do_pixel_norm', False,
  'If set, a pixelwise feature normalization is applied to the conv layers in the generator/encoder.'
  'Note that this may be incompatible with other normalization methods such as spectral norm. '
  'See PGGAN paper for details.')
tf.flags.DEFINE_boolean(
  'equalized_learning_rate', False,
  'See PGGAN paper for details.')
# Structural flags.
tf.flags.DEFINE_boolean(
  'use_res_block', False,
  'If true, use residual blocks in generator, discriminator, and encoder.'
)
tf.flags.DEFINE_boolean(
  'use_larger_filter_at_rgb_layer', False,
  'If true, instead of the standard 1x1 filter from the PGGAN paper, use a 7x7 filter in the to_rgb layers.'
)
tf.flags.DEFINE_integer(
  'pggan_max_num_channels', 256,
  'Maximum number of channels in the pggan network.')
tf.flags.DEFINE_integer(
  'pggan_max_num_channels_dis', None,
  'Maximum number of channels for the discriminator. If not set, it defaults to the `pggan_max_num_channels` flag.')
tf.flags.DEFINE_integer(
  'pggan_unet_max_concat_hw', None,
  'The maximum hw that the generator will concat encoder layer to. Anything exceeding this will not be concatenated.')
FLAGS = tf.flags.FLAGS

# Inherited function called from outside of this file.
conditional_progressive_gan_generator_arg_scope = pggan_utils.pggan_generator_arg_scope


#############
# Generator #
#############
def generator_three_layer_block(input_layer, out_channels, do_pixel_norm=False, conditional_layer=None,
                                unet_end_points=None):
  # Upsample
  ret = pggan_utils.resize_twice_as_big(input_layer)
  # Concat extra layers.
  ret = pggan_utils.maybe_concat_conditional_layer(ret, conditional_layer)
  ret = pggan_utils.maybe_concat_unet_layer(ret, unet_end_points)
  # Conv
  conv2d_out = ret
  conv2d_out = pggan_utils.maybe_pixel_norm(pggan_utils.maybe_equalized_conv2d(conv2d_out, out_channels, ),
                                            do_pixel_norm=do_pixel_norm)
  conv2d_out = pggan_utils.maybe_pixel_norm(pggan_utils.maybe_equalized_conv2d(conv2d_out, out_channels, ),
                                            do_pixel_norm=do_pixel_norm)
  ret = pggan_utils.maybe_resblock(ret, out_channels, conv2d_out)
  return ret


def get_noise_shape(batch_size=None, max_num_channels=None):
  """Gets the shape of the input noise to the GAN generator"""
  if max_num_channels is None:
    max_num_channels = FLAGS.pggan_max_num_channels
  return tf.TensorShape([batch_size, 1, 1, pggan_utils.get_num_channels(1, max_num_channels=max_num_channels)])


def generator(source=None,
              dtype=tf.float32,
              is_training=False,
              is_growing=False,
              alpha_grow=0.0,
              target_shape=None,
              max_num_channels=None,
              arg_scope_fn=pggan_utils.pggan_generator_arg_scope,
              do_pixel_norm=False,
              do_self_attention=False,
              self_attention_hw=64,
              conditional_layer=None,
              unet_end_points=None,
              ):
  """PGGAN generator.

  :param source: An optional tensor specifying the embedding the generator is conditioned on.
  :param dtype: tf dtype.
  :param is_training: Affects batch norm and update ops.
  :param is_growing: See PGGAN paper for details.
  :param alpha_grow: See PGGAN paper for details.
  :param target_shape: Output shape of format [batch, height, width, channels].
  :param max_num_channels: Maximum number of channels for latent embeddings.
  :param arg_scope_fn: A tf.contrib.framework.arg_scope.
  :param do_pixel_norm: See PGGAN paper for details.
  :param do_self_attention: See SAGAN paper for details.
  :param self_attention_hw: The height and width to start self attention at.
  :param conditional_layer: Generates the image conditioned on this tensor.
  :param unet_end_points: Concatenate UNet to their corresponding layers.
  :return: (Tensor with shape target_shape containing generated images, end_points dictionary)
  """
  if max_num_channels is None:
    max_num_channels = FLAGS.pggan_max_num_channels
  max_stage = int(np.log2(int(target_shape[1]))) - 2  # hw=4->max_stage=0, hw=8->max_stage=1 ...
  assert max_stage >= 0
  end_points = {}
  # Get latent vector the generator is conditioned on.
  with tf.variable_scope('latent_vector'):
    if source is None:
      source = tf.random_normal(get_noise_shape(target_shape[0], max_num_channels), dtype=dtype)
    if len(source.shape) == 2:
      source = tf.expand_dims(tf.expand_dims(source, 1), 2)
    assert len(source.shape) == 4, 'incorrect source shape for generator.'
    if source.shape[1] == 1 and source.shape[2] == 1:
      # Pads to 7x7 so that after the first conv layer with kernel size = 4, the size will be 4x4.
      source = tf.pad(source, paddings=((0, 0), (3, 3), (3, 3), (0, 0)))
  end_points['source'] = source
  net = source
  net_before_growth = None  # To be filled inside the for loop.

  with tf.contrib.framework.arg_scope(arg_scope_fn(is_training=is_training)):
    for stage in range(0, max_stage + 1):
      hw = 2 ** (stage + 2)
      output_channels = pggan_utils.get_num_channels(stage, max_num_channels)
      # 4x4 is a little different.
      if hw == 4:
        scope_name = 'block_%dx%dx%d' % (hw, hw, output_channels)
        with tf.variable_scope(scope_name):
          if source.shape[1] == 7 and source.shape[2] == 7:
            net = pggan_utils.maybe_pixel_norm(
              pggan_utils.maybe_equalized_conv2d(net, output_channels, kernel_size=4, padding='VALID'),
              do_pixel_norm=do_pixel_norm)
          else:
            # When the source is not random noise but is a tensor provided to the generator.
            assert source.shape[1] == 4 and source.shape[2] == 4
            net = pggan_utils.maybe_pixel_norm(
              pggan_utils.maybe_equalized_conv2d(net, output_channels, kernel_size=3, padding='SAME'),
              do_pixel_norm=do_pixel_norm)
          # Concatenate conditional layer before each block.
          net = pggan_utils.maybe_concat_conditional_layer(net, conditional_layer)
          net = pggan_utils.maybe_pixel_norm(pggan_utils.maybe_equalized_conv2d(net, output_channels),
                                             do_pixel_norm=do_pixel_norm)
          assert net.shape[1] == net.shape[2] == hw
          end_points[scope_name] = net
      else:
        # Outputs the image from the previous shape [hw/2, hw/2] to be used later.
        if stage == max_stage and is_growing:
          scope_name = 'generator_to_rgb_%dx%d' % (hw / 2, hw / 2)
          with tf.variable_scope(scope_name):
            if FLAGS.use_larger_filter_at_rgb_layer:
              kernel_size = min(7, hw / 2)
            else:
              kernel_size = 1
            # No pixel norm in to_rgb layers.
            net_before_growth = pggan_utils.maybe_equalized_conv2d(net, target_shape[-1], kernel_size=kernel_size,
                                                                   activation_fn=None)
            net_before_growth = pggan_utils.resize_twice_as_big(net_before_growth)
            end_points[scope_name] = net_before_growth
        # Generator block.
        scope_name = 'block_%dx%dx%d' % (hw, hw, output_channels)
        with tf.variable_scope(scope_name):
          net = generator_three_layer_block(net, output_channels, do_pixel_norm=do_pixel_norm,
                                            conditional_layer=conditional_layer, unet_end_points=unet_end_points)
          end_points[scope_name] = net

      # Adds self attention to the current layer if the `hw` matches `self_attention_hw`.
      net = pggan_utils.maybe_add_self_attention(do_self_attention, self_attention_hw, hw, output_channels, net,
                                                 end_points)

    scope_name = 'generator_to_rgb_%dx%d' % (hw, hw)
    with tf.variable_scope(scope_name):
      if FLAGS.use_larger_filter_at_rgb_layer:
        kernel_size = min(7, hw / 2)
      else:
        kernel_size = 1
      # No pixel norm in to_rgb layers.
      to_rgb_layer = pggan_utils.maybe_equalized_conv2d(net, target_shape[-1], kernel_size=kernel_size,
                                                        activation_fn=None)
      if not is_growing:
        output_layer = to_rgb_layer
      else:
        assert net_before_growth is not None
        output_layer = to_rgb_layer * alpha_grow + (1 - alpha_grow) * net_before_growth
        end_points['alpha_grow'] = alpha_grow

    end_points['output'] = output_layer
    if conditional_layer is not None:
      end_points['conditional_layer'] = conditional_layer
  return end_points['output'], end_points


#################
# Discriminator #
#################
def get_discriminator_max_stage(hw):
  return int(np.log2(hw) - 2)


def discriminator_two_layer_block(input_layer, out_channels, maybe_gdrop_fn):
  input_shape = input_layer.shape
  input_channels = input_shape[3]
  conv2d_out = input_layer
  # The first layer's depth is the same as input.
  conv2d_out = pggan_utils.maybe_equalized_conv2d(maybe_gdrop_fn(conv2d_out), input_channels, is_discriminator=True)
  # The second layer's depth is the output channel.
  conv2d_out = pggan_utils.maybe_equalized_conv2d(maybe_gdrop_fn(conv2d_out), out_channels, kernel_size=3,
                                                  is_discriminator=True)
  ret = pggan_utils.maybe_resblock(input_layer, out_channels, conv2d_out, is_discriminator=True)
  return ret

def discriminator_from_rgb_block(input_layer, out_channels):
  conv2d_out = input_layer
  conv2d_out = pggan_utils.maybe_equalized_conv2d(conv2d_out,
                                                  out_channels,
                                                                kernel_size=1, is_discriminator=True)
  ret = pggan_utils.maybe_resblock(input_layer, out_channels, conv2d_out,
                                                        is_discriminator=True)
  return ret

def discriminator_before_fc(source,
                            maybe_gdrop_fn=tf.identity,
                            is_training=False,
                            is_growing=False,
                            alpha_grow=0.0,
                            conditional_embed=None,
                            do_self_attention=False,
                            self_attention_hw=64,
                            arg_scope_fn=pggan_utils.pggan_discriminator_arg_scope, ):
  """The main body of the PGGAN discriminator. Contains everything before the fully connected prediction layer.

  :param source: Input to discriminate.
  :param maybe_gdrop_fn: optional gdrop function. Default is do not apply gdrop.
  :param is_training: Affects update ops.
  :param is_growing: See PGGAN paper for details.
  :param conditional_embed: Optional conditional embedding of the input source. E.g. 'female, red hair, ...' etc.
  :param alpha_grow: See PGGAN paper for details.
  :param do_self_attention: See SAGAN paper for details.
  :param self_attention_hw: The height and width to start self attention at.
  :param arg_scope_fn: A tf.contrib.framework.arg_scope.
  :return: (A tensor of shape [batch, 1], end_points dictionary)
  """
  # Note: discriminator do not use local response normalization (aka pixel norm in this code).
  max_num_channels = pggan_utils.get_discriminator_max_num_channels()
  max_stage = get_discriminator_max_stage(int(source.shape[1]))
  assert max_stage >= 0
  end_points = {}
  source_shrinked_from_rgb = None
  source_hw = source.shape[1]  # Assume same height and width.
  with tf.contrib.framework.arg_scope(arg_scope_fn(is_training=is_training)):
    # From RGB blocks
    if is_growing:
      source_shrinked_from_rgb = tf.nn.avg_pool(source, (1, 2, 2, 1), (1, 2, 2, 1), 'VALID')
      scope_name = 'from_rgb_%dx%d' % (source_hw / 2, source_hw / 2)
      with tf.variable_scope(scope_name):
        rgb_out = pggan_utils.get_num_channels(max_stage - 1, max_num_channels=max_num_channels)
        source_shrinked_from_rgb = discriminator_from_rgb_block(source_shrinked_from_rgb, rgb_out)
      end_points[scope_name] = source_shrinked_from_rgb

    scope_name = 'from_rgb_%dx%d' % (source_hw, source_hw)
    with tf.variable_scope(scope_name):
      net = discriminator_from_rgb_block(source, pggan_utils.get_num_channels(max_stage,
                                                                              max_num_channels=max_num_channels))
      end_points[scope_name] = net

    # Discriminator blocks.
    # The blocks are down to 8x8. 4x4 is handled outside of the for loop.
    for stage in range(max_stage, 0, -1):
      num_channels = pggan_utils.get_num_channels(stage - 1, max_num_channels=max_num_channels)
      hw_div_by = 2 ** (max_stage - stage)
      current_hw = source_hw / hw_div_by

      # Self atttention. Note that this may not get called if self_attention_hw == min_hw == 4.
      net = pggan_utils.maybe_add_self_attention(do_self_attention, self_attention_hw, current_hw, num_channels, net,
                                                 end_points)
      # Conv.
      scope_name = 'encoder_block_%dx%dx%d' % (current_hw, current_hw, num_channels)
      with tf.variable_scope(scope_name):
        net = discriminator_two_layer_block(net, num_channels, maybe_gdrop_fn=maybe_gdrop_fn)
        end_points[scope_name] = net
      # Down-sample.
      current_hw /= 2
      scope_name = 'downsample_to_%dx%dx%d' % (current_hw, current_hw, num_channels)
      with tf.variable_scope(scope_name):
        net = tf.nn.avg_pool(net, (1, 2, 2, 1), (1, 2, 2, 1), 'VALID')
        end_points[scope_name] = net
      # If it is growing, encode the down-sampled image.
      if stage == max_stage and is_growing:
        assert source_shrinked_from_rgb is not None
        assert current_hw == source_hw / 2
        scope_name = 'encoder_block_interpolated_%dx%dx%d' % (current_hw, current_hw, num_channels)
        with tf.variable_scope(scope_name):
          net = net * alpha_grow + (1 - alpha_grow) * source_shrinked_from_rgb
          end_points[scope_name] = net

    # The final 4x4 block, which is a little bit different from all others.
    if conditional_embed is not None:
      net_h = int(net.shape[1])
      net_w = int(net.shape[2])
      repeated = tf.expand_dims(tf.expand_dims(conditional_embed, axis=1), axis=2)
      repeated = util_misc.tf_repeat(repeated, [1, net_h, net_w, 1])
      net = tf.concat((net, repeated), axis=-1, name='concat_conditional_embed')

    with tf.variable_scope('before_fc_1x1x%d' % max_num_channels):
      # TODO: is this compatible with all normalization and GAN training methods (e.g. Dragan)?
      net = pggan_utils.minibatch_state_concat(net)
      net = pggan_utils.maybe_equalized_conv2d(maybe_gdrop_fn(net), max_num_channels, kernel_size=3,
                                               is_discriminator=True, padding='SAME')
      net = pggan_utils.maybe_equalized_conv2d(maybe_gdrop_fn(net), max_num_channels, kernel_size=4,
                                               is_discriminator=True, padding='VALID')
      end_points['before_fc_1x1x%d' % max_num_channels] = net

  end_points['before_fc'] = net
  return net, end_points


def discriminator(source,
                  conditional_embed=None,
                  do_dgrop=False,
                  gdrop_strength=0.0,
                  is_training=False,
                  is_growing=False,
                  alpha_grow=0.0,
                  do_self_attention=False,
                  self_attention_hw=64,
                  arg_scope_fn=pggan_utils.pggan_discriminator_arg_scope,
                  conditional_layer=None, ):
  """Given a square image tensor, output a prediction tensor. See `discriminator_before_fc` for details."""

  def maybe_gdrop(layer):
    if do_dgrop and is_training and gdrop_strength:
      return ops.gdrop(layer, mode='prop', strength=gdrop_strength)
    else:
      return layer

  net = pggan_utils.maybe_concat_conditional_layer(source, conditional_layer)
  net, end_points = discriminator_before_fc(net, maybe_gdrop, is_training, is_growing, alpha_grow,
                                            conditional_embed=conditional_embed,
                                            do_self_attention=do_self_attention,
                                            self_attention_hw=self_attention_hw,
                                            arg_scope_fn=arg_scope_fn, )
  with tf.variable_scope('prediction'):
    weights_init_stddev = 1.0 if FLAGS.equalized_learning_rate else 0.02
    net = pggan_utils.maybe_equalized_fc(tf.squeeze(net, axis=(1, 2)), 1,
                                         is_discriminator=True,
                                         activation_fn=None,
                                         weights_initializer=tf.random_normal_initializer(0, weights_init_stddev),
                                         weights_regularizer=None,
                                         )
  end_points['prediction'] = net
  if conditional_layer is not None:
    end_points['conditional_layer'] = conditional_layer
    end_points['conditional_layer_w_source'] = source + tf.cast(
      tf.reduce_sum(conditional_layer, axis=-1, keepdims=True), source.dtype)
  return net, end_points


###########
# Encoder #
###########
def encoder_two_layer_block(input_layer, out_channels, do_pixel_norm=False):
  input_shape = input_layer.shape
  input_channels = input_shape[3]
  conv2d_out = input_layer
  # The first layer's depth is the same as input.
  conv2d_out = pggan_utils.maybe_pixel_norm(pggan_utils.maybe_equalized_conv2d(conv2d_out, input_channels, ),
                                            do_pixel_norm=do_pixel_norm)
  # The second layer's depth is the output channel.
  conv2d_out = pggan_utils.maybe_pixel_norm(pggan_utils.maybe_equalized_conv2d(conv2d_out, out_channels, kernel_size=3),
                                            do_pixel_norm=do_pixel_norm)
  ret = pggan_utils.maybe_resblock(input_layer, out_channels, conv2d_out)
  return ret

def encoder_from_rgb_block(input_layer, out_channels, do_pixel_norm=False):
  conv2d_out = pggan_utils.maybe_pixel_norm(
    pggan_utils.maybe_equalized_conv2d(input_layer, out_channels, kernel_size=1), do_pixel_norm=do_pixel_norm)
  ret = pggan_utils.maybe_resblock(input_layer, out_channels, conv2d_out)
  return ret



def encoder_before_classification(source,
                                  output_dim=None,  # Not used.
                                  target_hw=None,
                                  dtype=tf.float32,  # Not used.
                                  is_training=False,
                                  is_growing=False,
                                  alpha_grow=0.0,
                                  target_shape=None,  # Not used.
                                  max_num_channels=None,
                                  arg_scope_fn=pggan_utils.pggan_generator_arg_scope,
                                  do_pixel_norm=False,
                                  do_self_attention=False,
                                  self_attention_hw=64,
                                  conditional_layer=None,
                                  ):
  """The body of the PGGAN-mirrored encoder. It takes a square image and outputs a Bx4x4xC tensor."""
  if max_num_channels is None:
    max_num_channels = FLAGS.pggan_max_num_channels
  if conditional_layer is not None:
    raise NotImplementedError('conditional layer not supported in the encoder.')

  source = pggan_utils.maybe_concat_conditional_layer(source, conditional_layer)
  max_stage = int(np.log2(int(source.shape[1]))) - 2
  assert max_stage >= 0
  end_points = {}
  end_points['source'] = source
  source_shrinked_from_rgb = None
  source_hw = source.shape[1]  # Assume same height and width.
  source_c = source.shape[-1]

  with tf.contrib.framework.arg_scope(arg_scope_fn(is_training=is_training)):
    # From RGB block.
    if is_growing:
      source_shrinked_from_rgb = tf.nn.avg_pool(source, (1, 2, 2, 1), (1, 2, 2, 1), 'VALID')
      scope_name = 'from_rgb_%dx%d' % (source_hw / 2, source_hw / 2)
      with tf.variable_scope(scope_name):
        from_rgb_c = pggan_utils.get_num_channels(max_stage - 1, max_num_channels=max_num_channels)
        source_shrinked_from_rgb = encoder_from_rgb_block(source_shrinked_from_rgb, from_rgb_c, do_pixel_norm=do_pixel_norm)
      end_points[scope_name] = source_shrinked_from_rgb
    scope_name = 'from_rgb_%dx%d' % (source_hw, source_hw)
    with tf.variable_scope(scope_name):

      from_rgb_c = pggan_utils.get_num_channels(max_stage, max_num_channels=max_num_channels)
      net = encoder_from_rgb_block(source, from_rgb_c, do_pixel_norm=do_pixel_norm)
      end_points[scope_name] = net

    # Encoder blocks.
    for stage in range(max_stage, 0, -1):
      num_channels = pggan_utils.get_num_channels(stage - 1, max_num_channels=max_num_channels)
      hw_div_by = 2 ** (max_stage - stage)
      current_hw = source_hw / hw_div_by
      if target_hw is not None and current_hw < target_hw:
        break

      # Self attention (optional).
      net = pggan_utils.maybe_add_self_attention(do_self_attention, self_attention_hw, current_hw, num_channels, net,
                                                 end_points)
      scope_name = 'encoder_block_%dx%dx%d' % (current_hw, current_hw, num_channels)
      with tf.variable_scope(scope_name):
        net = encoder_two_layer_block(net, num_channels, do_pixel_norm=do_pixel_norm)
        end_points[scope_name] = net
      current_hw /= 2
      scope_name = 'downsample_to_%dx%dx%d' % (current_hw, current_hw, num_channels)
      with tf.variable_scope(scope_name):
        # Down sample.
        net = tf.nn.avg_pool(net, (1, 2, 2, 1), (1, 2, 2, 1), 'VALID')
        end_points[scope_name] = net

      if stage == max_stage and is_growing:
        assert source_shrinked_from_rgb is not None
        scope_name = 'encoder_block_interpolated_%dx%dx%d' % (current_hw, current_hw, num_channels)
        with tf.variable_scope(scope_name):
          net = net * alpha_grow + (1 - alpha_grow) * source_shrinked_from_rgb
          end_points[scope_name] = net
    end_points['before_classification'] = net

    return net, end_points


def encoder_classification(source,
                           output_dim=4,
                           is_training=False,
                           arg_scope_fn=pggan_utils.pggan_generator_arg_scope,
                           prediction_scope_name='prediction',
                           **kwargs_unused):
  """Adds some last few layers of convolutions followed by a fully-connected layer. Outputs [batch, 1, 1, channels]."""
  end_points = {}
  net = source
  with tf.contrib.framework.arg_scope(arg_scope_fn(is_training=is_training)):
    before_fc_num_channels = FLAGS.pggan_max_num_channels
    with tf.variable_scope('before_fc_1x1x%d' % before_fc_num_channels):
      net = pggan_utils.maybe_equalized_conv2d(net, before_fc_num_channels, kernel_size=3, padding='SAME')
      net = pggan_utils.maybe_equalized_conv2d(net, before_fc_num_channels, kernel_size=4, padding='VALID')
      end_points['before_fc_1x1x%d' % before_fc_num_channels] = net

    with tf.variable_scope(prediction_scope_name, reuse=tf.AUTO_REUSE):
      weights_init_stddev = 1.0 if FLAGS.equalized_learning_rate else 0.02
      net = pggan_utils.maybe_equalized_fc(tf.squeeze(net, axis=(1, 2)), output_dim,
                                           activation_fn=None,
                                           weights_initializer=tf.random_normal_initializer(0, weights_init_stddev),
                                           weights_regularizer=None,
                                           )
    end_points[prediction_scope_name] = net
    return net, end_points


def encoder(source,
            output_dim=4,
            dtype=tf.float32,
            is_training=False,
            is_growing=False,
            alpha_grow=0.0,
            target_shape=None,
            arg_scope_fn=pggan_utils.pggan_generator_arg_scope,
            do_pixel_norm=False,
            do_self_attention=False,
            prediction_scope_name='prediction',
            conditional_layer=None,
            ):
  """The full encoder that encodes input to tensor of shape [batch, 1, 1, channels]. For most image generation cases
   `encoder_before_classification` is better."""
  net, end_points = encoder_before_classification(source,
                                                  output_dim=output_dim,
                                                  dtype=dtype,
                                                  is_training=is_training,
                                                  is_growing=is_growing,
                                                  alpha_grow=alpha_grow,
                                                  target_shape=target_shape,
                                                  arg_scope_fn=arg_scope_fn,
                                                  do_pixel_norm=do_pixel_norm,
                                                  do_self_attention=do_self_attention,
                                                  conditional_layer=conditional_layer)
  net, classification_endpoints = encoder_classification(net,
                                                         output_dim=output_dim,
                                                         is_training=is_training,
                                                         arg_scope_fn=arg_scope_fn,
                                                         prediction_scope_name=prediction_scope_name, )
  end_points.update(classification_endpoints)
  return net, end_points
