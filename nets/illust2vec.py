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
# ==============================================================================
"""Contains model definitions for the illust2vec network - A version of the VGG network.

Modified from vgg.py under the same folder.
See https://github.com/rezoo/illustration2vec for more info about illust2vec.

Usage:
  with slim.arg_scope(illust2vec.vgg_arg_scope()):
    outputs, end_points = illust2vec.illust2vec(inputs)

@@illust2vec
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

from datasets import danbooru_2_illust2vec


def vgg_arg_scope(weight_decay=0.0005):
  """Defines the VGG arg scope.

  Args:
    weight_decay: The l2 regularization coefficient.

  Returns:
    An arg_scope.
  """
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      weights_regularizer=slim.l2_regularizer(weight_decay),
                      biases_initializer=tf.zeros_initializer()):
    with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
      return arg_sc


def illust2vec(inputs,
               num_classes=1539,
               is_training=True,  # Unused.
               dropout_keep_prob=0.5,  # Unused.
               spatial_squeeze=True,
               scope='illust2vec',
               prediction_fn=tf.sigmoid,
               fc_conv_padding='VALID',  # Unused.
               global_pool=False  # Unused.
               ):
  """Illust2vec VGG.

  Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 224x224.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes. If 0 or None, the logits layer is
      omitted and the input features to the logits layer are returned instead.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      outputs. Useful to remove unnecessary dimensions for classification.
    scope: Optional scope for the variables.
    fc_conv_padding: the type of padding to use for the fully connected layer
      that is implemented as a convolutional layer. Use 'SAME' padding if you
      are applying the network in a fully convolutional manner and want to
      get a prediction map downsampled by a factor of 32 as an output.
      Otherwise, the output prediction map will be (input / 32) - 6 in case of
      'VALID' padding.
    global_pool: Optional boolean flag. If True, the input to the classification
      layer is avgpooled to size 1x1, for any input size. (This is not part
      of the original VGG architecture.)

  Returns:
    net: the output of the logits layer (if num_classes is a non-zero integer),
      or the input to the logits layer (if num_classes is 0 or None).
    end_points: a dict of tensors with intermediate activations.
  """
  if not num_classes:
    num_classes = danbooru_2_illust2vec.DEFAULT_NUM_CLASSES

  with tf.variable_scope(scope, 'illust2vec', [inputs]) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                        outputs_collections=end_points_collection):
      net = slim.conv2d(inputs, 64, [3, 3], scope='conv1_1')
      net = slim.max_pool2d(net, [2, 2], scope='pool1')
      net = slim.conv2d(net, 128, [3, 3], scope='conv2_1')
      net = slim.max_pool2d(net, [2, 2], scope='pool2')
      net = slim.conv2d(net, 256, [3, 3], scope='conv3_1')
      net = slim.conv2d(net, 256, [3, 3], scope='conv3_2')
      net = slim.max_pool2d(net, [2, 2], scope='pool3')
      net = slim.conv2d(net, 512, [3, 3], scope='conv4_1')
      net = slim.conv2d(net, 512, [3, 3], scope='conv4_2')
      net = slim.max_pool2d(net, [2, 2], scope='pool4')
      net = slim.conv2d(net, 512, [3, 3], scope='conv5_1')
      net = slim.conv2d(net, 512, [3, 3], scope='conv5_2')
      net = slim.max_pool2d(net, [2, 2], scope='pool5')
      net = slim.conv2d(net, 1024, [3, 3], scope='conv6_1')
      net = slim.conv2d(net, 1024, [3, 3], scope='conv6_2')
      net = slim.conv2d(net, 1024, [3, 3], scope='conv6_3')
      net = slim.conv2d(net, num_classes, [3, 3], activation_fn=None, scope='conv6_4')
      net = slim.avg_pool2d(net, [7, 7], scope='pool6')

      end_points = slim.utils.convert_collection_to_dict(end_points_collection)
      if spatial_squeeze:
        net = tf.squeeze(net, [1, 2], name='pool6/squeezed')
      end_points['Logits'] = net
      net = prediction_fn(net)
      end_points['Predictions'] = net
      return net, end_points


illust2vec.default_image_size = 224
