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
"""Defines the CycleGAN generator and discriminator networks."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf
import tensorflow.contrib.layers as layers
from nets import cyclegan



def cyclegan_discriminator_resnet(images,
                                  arg_scope_fn=cyclegan.cyclegan_arg_scope,
                                  num_resnet_blocks=6,
                                  num_filters=64,
                                  upsample_fn=cyclegan.cyclegan_upsample,
                                  kernel_size=3,
                                  num_outputs=3,
                                  tanh_linear_slope=0.0,
                                  is_training=False):
  """Defines the cyclegan resnet network architecture.

  As closely as possible following
  https://github.com/junyanz/CycleGAN/blob/master/models/architectures.lua#L232

  FYI: This network requires input height and width to be divisible by 4 in
  order to generate an output with shape equal to input shape. Assertions will
  catch this if input dimensions are known at graph construction time, but
  there's no protection if unknown at graph construction time (you'll see an
  error).

  Args:
    images: Input image tensor of shape [batch_size, h, w, 3].
    arg_scope_fn: Function to create the global arg_scope for the network.
    num_resnet_blocks: Number of ResNet blocks in the middle of the generator.
    num_filters: Number of filters of the first hidden layer.
    upsample_fn: Upsampling function for the decoder part of the generator.
    kernel_size: Size w or list/tuple [h, w] of the filter kernels for all inner
      layers.
    num_outputs: Number of output layers. Defaults to 3 for RGB.
    tanh_linear_slope: Slope of the linear function to add to the tanh over the
      logits.
    is_training: Whether the network is created in training mode or inference
      only mode. Not actually needed, just for compliance with other generator
      network functions.

  Returns:
    A `Tensor` representing the model output and a dictionary of model end
      points.

  Raises:
    ValueError: If the input height or width is known at graph construction time
      and not a multiple of 4.
  """
  # Neither dropout nor batch norm -> dont need is_training
  del is_training

  end_points = {}

  input_size = images.shape.as_list()
  height, width = input_size[1], input_size[2]
  if height and height % 4 != 0:
    raise ValueError('The input height must be a multiple of 4.')
  if width and width % 4 != 0:
    raise ValueError('The input width must be a multiple of 4.')

  if not isinstance(kernel_size, (list, tuple)):
    kernel_size = [kernel_size, kernel_size]

  kernel_height = kernel_size[0]
  kernel_width = kernel_size[1]
  pad_top = (kernel_height - 1) // 2
  pad_bottom = kernel_height // 2
  pad_left = (kernel_width - 1) // 2
  pad_right = kernel_width // 2
  paddings = np.array(
      [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]],
      dtype=np.int32)
  spatial_pad_3 = np.array([[0, 0], [3, 3], [3, 3], [0, 0]])

  with tf.contrib.framework.arg_scope(arg_scope_fn()):

    ###########
    # Encoder #
    ###########
    with tf.variable_scope('input'):
      # 7x7 input stage
      net = tf.pad(images, spatial_pad_3, 'REFLECT')
      net = layers.conv2d(net, num_filters, kernel_size=[7, 7], padding='VALID')
      end_points['encoder_0'] = net

    with tf.variable_scope('encoder'):
      with tf.contrib.framework.arg_scope(
          [layers.conv2d],
          kernel_size=kernel_size,
          stride=2,
          activation_fn=tf.nn.relu,
          padding='VALID'):

        net = tf.pad(net, paddings, 'REFLECT')
        net = layers.conv2d(net, num_filters * 2)
        end_points['encoder_1'] = net
        net = tf.pad(net, paddings, 'REFLECT')
        net = layers.conv2d(net, num_filters * 4)
        end_points['encoder_2'] = net

    ###################
    # Residual Blocks #
    ###################
    with tf.variable_scope('residual_blocks'):
      with tf.contrib.framework.arg_scope(
          [layers.conv2d],
          kernel_size=kernel_size,
          stride=1,
          activation_fn=tf.nn.relu,
          padding='VALID'):
        for block_id in xrange(num_resnet_blocks):
          with tf.variable_scope('block_{}'.format(block_id)):
            res_net = tf.pad(net, paddings, 'REFLECT')
            res_net = layers.conv2d(res_net, num_filters * 4)
            res_net = tf.pad(res_net, paddings, 'REFLECT')
            res_net = layers.conv2d(res_net, num_filters * 4,
                                    activation_fn=None)
            net += res_net

            end_points['resnet_block_%d' % block_id] = net


    ####
    # FC
    ####
    with tf.variable_scope('prediction'):
      # net = tf.reduce_logsumexp  # Maybe this is better. It's not max but it's still differentiable.
      net = tf.reduce_mean(net, axis=(1,2))
      net = layers.fully_connected(net, 1, activation_fn=None)
    end_points['prediction'] = net


  return end_points['prediction'], end_points
