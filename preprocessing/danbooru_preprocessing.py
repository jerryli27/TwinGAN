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
"""Provides utilities to preprocess images.

This is the default used for most tasks and can be applied to non-danbooru datasets.
Adapted from inception_preprocessing.py.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

from preprocessing import preprocessing_util

_PADDING = 0
_ALLOWED_COLOR_SPACE = {'rgb', 'yiq', 'bgr', 'gray'}
_ALLOWED_DTYPE = {tf.float32, tf.float16, tf.uint8}
_MEAN_IMAGE_SUBTRACTION_BGR = [164.76139251, 167.47864617, 181.13838569]  # For Danbooru.
_RANDOM_CROP_RATIO = 0.8

def _check_color_space(color_space):
  assert color_space in _ALLOWED_COLOR_SPACE, 'color_space must be one of %s' %(str(_ALLOWED_COLOR_SPACE))

def _check_dtype(dtype):
  assert dtype in _ALLOWED_DTYPE, 'dtype must be one of %s' %(str(_ALLOWED_DTYPE))

def apply_with_random_selector(x, func, num_cases):
  """Computes func(x, sel), with sel sampled from [0...num_cases-1].

  Args:
    x: input Tensor.
    func: Python function to apply.
    num_cases: Python int32, number of cases to sample sel from.

  Returns:
    The result of func(x, sel), where func receives the value of the
    selector as a python integer, but sel is sampled dynamically.
  """
  sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
  # Pass the real x only to one of the func calls.
  return control_flow_ops.merge([
      func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
      for case in range(num_cases)])[0]


def distort_color(image, color_ordering=0, fast_mode=True, scope=None):
  """Distort the color of a Tensor image.

  Each color distortion is non-commutative and thus ordering of the color ops
  matters. Ideally we would randomly permute the ordering of the color ops.
  Rather then adding that level of complication, we select a distinct ordering
  of color ops for each preprocessing thread.

  Args:
    image: 3-D Tensor containing single image in [0, 1].
    color_ordering: Python int, a type of distortion (valid values: 0-3).
    fast_mode: Avoids slower ops (random_hue and random_contrast)
    scope: Optional scope for name_scope.
  Returns:
    3-D Tensor color-distorted image on range [0, 1]
  Raises:
    ValueError: if color_ordering not in [0, 3]
  """
  with tf.name_scope(scope, 'distort_color', [image]):
    if fast_mode:
      if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      else:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
    else:
      if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
      elif color_ordering == 1:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
      elif color_ordering == 2:
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      elif color_ordering == 3:
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
      else:
        raise ValueError('color_ordering must be in [0, 3]')

    # The random_* ops do not necessarily clamp.
    return tf.clip_by_value(image, 0.0, 1.0)


def preprocess_image(image,
                     output_height,
                     output_width,
                     dtype=tf.float32,
                     padding=_PADDING,
                     color_space='rgb',
                     subtract_mean=False,
                     resize_mode=preprocessing_util.RESIZE_MODE_PAD,
                     is_training=False,
                     do_random_cropping=False,
                     random_cropping_ratio=_RANDOM_CROP_RATIO,
                     fast_mode=True,
                     add_image_summaries=True,
                     summary_prefix=''):
  """Preprocesses the given image for training.

  Note that the actual resizing scale is sampled from
    [`resize_size_min`, `resize_size_max`].

  Args:
    image: A `Tensor` or a list of Tensors representing images of arbitrary size.
    output_height: The height of the image after preprocessing.
    output_width: The width of the image after preprocessing.
    dtype: tensorflow dtype. Use tf.float16 to get speed up from Nvidia tensor core.
    padding: The amound of padding before and after each dimension of the image.
    color_space: color space for the output image.
    subtract_mean: a boolean. Subtract _MEAN_IMAGE_SUBTRACTION_BGR from image if set. Used for vgg19.
    resize_mode: See preprocessing_util.resize_image.
    is_training: a boolean. Indicates whether the model is being trained.
    do_random_cropping: a boolean. If True, randomly crop the resized image to `_RANDOM_CROP_RATIO` of its original size
      and resize again.
    random_cropping_ratio: a float.
    fast_mode: Optional boolean, if True avoids slower transformations.
    add_image_summaries: a boolean. If true, enable image summaries.
    summary_prefix: the prefix(s) to the input image(s).

  Returns:
    A preprocessed image.
  """
  _check_dtype(dtype)
  _check_color_space(color_space)
  # These random variables will make sure that if the input is a list of
  # images, the same random action will be applied to all of them.
  flip_random_variable = None
  is_input_list = (isinstance(image, list) or isinstance(image, tuple))
  ret = []
  if not is_input_list:
    image_list = [image]
    prefix_list = [summary_prefix]
  else:
    if (isinstance(summary_prefix, list) or isinstance(summary_prefix, tuple)):
      prefix_list = summary_prefix
    else:
      prefix_list = [summary_prefix for _ in range(len(image))]
    image_list = image

  for i, im in enumerate(image_list):
    if add_image_summaries:
      tf.summary.image(prefix_list[i] + 'image', tf.expand_dims(im, 0))

    if padding > 0:
      im = tf.pad(im, [[padding, padding], [padding, padding], [0, 0]])

    if not subtract_mean:
      # This step converts all images to float32 and their values to [0,1] range.
      # This can make training easier and make the losses all lie on the same scale.
      # (Otherwise l1 loss between images can be 255 times larger than say a cross entropy loss)
      if dtype != tf.uint8 and dtype != im.dtype:
        im = tf.image.convert_image_dtype(im, dtype=dtype)

    # Resizes the image to a square.
    assert output_height == output_width
    # If random cropping is on, first resize to a larger hw, then crop, lastly resize to
    if is_training and do_random_cropping:
      distorted_image = preprocessing_util.resize_image(im, resize_mode, int(output_height / random_cropping_ratio) , )
    else:
      distorted_image = preprocessing_util.resize_image(im, resize_mode, output_height,)


    if subtract_mean:
      # Images are in rgb, so the bgr mean needs to be reversed. Used for vgg19.
      distorted_image = preprocessing_util.mean_image_subtraction(tf.cast(distorted_image, tf.float32),
                                                                  _MEAN_IMAGE_SUBTRACTION_BGR[::-1])

    if is_training:
      if do_random_cropping:
        distorted_image = preprocessing_util.random_crop_image(distorted_image, random_cropping_ratio, resize_hw=output_height)

      # Randomly flip the image horizontally.
      distorted_image, flip_random_variable = preprocessing_util.random_flip_left_right(
        distorted_image, random_var=flip_random_variable, return_random_var=True)

      if color_space != 'gray':
        distorted_image = apply_with_random_selector(
          distorted_image,
          lambda x, ordering: distort_color(x, ordering, fast_mode),
          num_cases=4)

    if add_image_summaries:
      tf.summary.image(prefix_list[i] + 'distorted_image', tf.expand_dims(distorted_image, 0))

    # Transform the image to `dtype`. Used for vgg19 when `subtract_mean` is true.
    if dtype != distorted_image.dtype:
      distorted_image = tf.cast(distorted_image, dtype, name='ToDtype')

    # Transform into other color space.
    if color_space == 'yiq':
      distorted_image = preprocessing_util.rgb_to_yiq(distorted_image)
    elif color_space == 'bgr':
      distorted_image = tf.reverse(distorted_image, axis=[-1])
    ret.append(distorted_image)

  if not is_input_list:
    assert len(ret) == 1
    ret = ret[0]
  return ret


def postprocess_image(image,
                       color_space='rgb',
                       subtract_mean=False,):
  if color_space == 'rgb':
    ret = image
  elif color_space == 'yiq':
    ret = preprocessing_util.yiq_to_rgb(image)
  elif color_space == 'bgr':
    ret = tf.reverse(image, axis=[-1])
  elif color_space == 'gray':
    ret = image
  else:
    raise NotImplementedError
  if subtract_mean:
    # Undo subtract mean.
    neg_mean = [-item for item in _MEAN_IMAGE_SUBTRACTION_BGR[::-1]]
    ret = preprocessing_util.mean_image_subtraction(tf.cast(ret, tf.float32), neg_mean)
    if ret.dtype != tf.uint8:
      ret = ret / tf.constant(255.0, dtype=ret.dtype)
  if ret.shape[-1] > 3:
    ret = tf.reduce_sum(ret, axis=-1, keepdims=True)
  ret = tf.clip_by_value(ret, 0.0, 1.0)
  return ret