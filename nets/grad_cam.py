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
"""This file implements Grad-CAM as described in the paper by Selvaraju et. al."""

import tensorflow as tf
import util_misc

from preprocessing import preprocessing_util
from preprocessing import danbooru_preprocessing

CNN_LAYER_MAP = {
  'inception_v3': 'Mixed_7c',
  'illust2vec': 'illust2vec/conv6_3',
  'mobilenet_v1': 'Conv2d_13_pointwise',
  'pggan_discriminator_generated': 'discriminator_generated_encoder_block_8x8x512', # 'discriminator_generated_from_rgb_128x128', # 'discriminator_generated_before_fc',
  'pggan_discriminator_real': 'discriminator_real_encoder_block_8x8x512', # 'discriminator_real_from_rgb_128x128' # 'discriminator_real_before_fc',
}

# Format: layer name, keep_positive.
CLASS_LAYER_MAP = {
  'inception_v3': ('Logits', True),
  'illust2vec': ('Logits', True),
  'mobilenet_v1': ('Logits', True),
  'pggan_discriminator_generated': ('discriminator_generated_prediction', False),
  'pggan_discriminator_real': ('discriminator_real_prediction', False),
}


FLAGS = tf.app.flags.FLAGS


def grad_cam(model_name, end_points, class_index, last_cnn=None, class_layer=None, keep_positive=True):
  """For each class, outputs a heat map of size equals to the last cnn layer."""
  if model_name not in CNN_LAYER_MAP and last_cnn is None:
    raise ValueError('parameter `model_name` must be in `CNN_LAYER_MAP`')
  if model_name not in CLASS_LAYER_MAP and class_layer is None:
    raise ValueError('parameter `model_name` must be in `CLASS_LAYER_MAP`')
  if last_cnn is None:
    if CNN_LAYER_MAP[model_name] not in end_points:
      raise ValueError('cnn layer not found. Available layers are: %s' %(str(end_points.keys())))
    last_cnn = end_points[CNN_LAYER_MAP[model_name]]
  if class_layer is None:
    class_layer_name = CLASS_LAYER_MAP[model_name][0]
    class_layer = end_points[class_layer_name]
    keep_positive = CLASS_LAYER_MAP[model_name][1]
  if not keep_positive:  # If keep_positive is false, negate the layer.
    class_layer = -class_layer
  if class_index is None:
    chosen_class_layer = class_layer
  elif isinstance(class_index, int):
    if class_index >= class_layer.shape[-1]:
      raise ValueError('class_index %d is larger than the number of classes in class_layer: %d'
                       %(class_index, int(class_layer.shape[-1])))
    chosen_class_layer = class_layer[:, class_index]
  else:
    raise NotImplementedError('unsupported class_index type.')

  # The following are methods to get the grad_cam for all classes. They are either slow or does not work.
  # each_class_layers = tf.unstack(class_layer, axis=-1)
  # dyda = [tf.gradients(current_class_tensor, last_cnn)[0] for current_class_tensor in each_class_layers]
  # dyda = tf.stack(dyda, axis=-1)

  # class_layer_t = tf.transpose(class_layer, (1, 0))
  # dyda = tf.map_fn(lambda current_class_tensor: tf.gradients(current_class_tensor, last_cnn)[0], class_layer_t)
  # dyda = tf.transpose(dyda, [i for i in range(1, len(dyda.shape))] + [0])
  dyda = tf.gradients(chosen_class_layer, last_cnn)[0]
  alpha = tf.reduce_mean(dyda, axis=(1, 2), keepdims=True)
  l_gradcam = tf.reduce_sum(alpha * last_cnn, axis=-1, keepdims=True)
  l_gradcam = tf.nn.relu(l_gradcam)
  l_gradcam = l_gradcam / (tf.reduce_max(l_gradcam) + 0.00000001)
  # l_gradcam = tf.reduce_sum(alpha * tf.expand_dims(last_cnn, -1), axis=-2,)

  return l_gradcam

def impose_mask_on_image(mask, image, image_alpha=0.5):
  reshaped_mask = tf.cast(tf.image.resize_bilinear(mask, image.shape[-3:-1]), dtype=mask.dtype)
  if reshaped_mask.shape[-1] == 1:
    # reshaped_mask = util_misc.tf_repeat(reshaped_mask, [1 for _ in range(len(image.shape) - 1)] + [3])
    reshaped_mask = util_misc.grayscale_to_heatmap(reshaped_mask, is_bgr=(FLAGS.color_space=='bgr'))
    # reshaped_mask = tf.reshape(reshaped_mask, image.shape)
    if FLAGS.subtract_mean:
      reshaped_mask = tf.multiply(reshaped_mask, tf.constant(255.0, reshaped_mask.dtype))
      # The reshape mask must also be subtracted by the image means.
      reshaped_mask = preprocessing_util.mean_image_subtraction(reshaped_mask, danbooru_preprocessing._MEAN_IMAGE_SUBTRACTION_BGR)
  if reshaped_mask.shape[-1] != 3:
    raise ValueError('Currently only mask of feature size 1 or 3 is supported.')

  masked_image = (image * tf.constant(image_alpha, dtype=image.dtype) +
                  reshaped_mask * tf.constant((1 - image_alpha), dtype=image.dtype))

  return masked_image