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

RESIZE_MODE_NONE = 0
RESIZE_MODE_PAD = 1  # First pad image to a square, then reshape to given height and width.
RESIZE_MODE_CROP = 2  # Crop the center part to given height and width.
RESIZE_MODE_RESHAPE = 3
RESIZE_MODE_RANDOM_CROP = 4  # Crop at a random location.
RESIZE_MODE_RANDOM_CROP_AND_RESHAPE = 5  # See the `random_crop_and_reshape_initial_crop_hw` flag.

tf.flags.DEFINE_integer('random_crop_and_reshape_initial_crop_hw', None,
                        'First randomly crop to this height width, then reshape to a (smaller) desired height width.')
FLAGS = tf.flags.FLAGS

def _ImageDimensions(image, rank):
  """Returns the dimensions of an image tensor.

  Args:
    image: A rank-D Tensor. For 3-D  of shape: `[height, width, channels]`.
    rank: The expected rank of the image

  Returns:
    A list of corresponding to the dimensions of the
    input image.  Dimensions that are statically known are python integers,
    otherwise they are integer scalar tensors.
  """
  if image.get_shape().is_fully_defined():
    return image.get_shape().as_list()
  else:
    static_shape = image.get_shape().with_rank(rank).as_list()
    dynamic_shape = tf.unstack(tf.shape(image), rank)
    return [
        s if s is not None else d for s, d in zip(static_shape, dynamic_shape)
    ]

def resize_mode_str_to_int(resize_mode):
  if resize_mode == 'NONE':
    return RESIZE_MODE_NONE
  elif resize_mode == 'PAD':
    return RESIZE_MODE_PAD
  elif resize_mode == 'CROP':
    return RESIZE_MODE_CROP
  elif resize_mode == 'RANDOM_CROP':
    return RESIZE_MODE_RANDOM_CROP
  elif resize_mode == 'RANDOM_CROP_AND_RESHAPE':
    return RESIZE_MODE_RANDOM_CROP_AND_RESHAPE  # See the `random_crop_and_reshape_initial_crop_hw` flag.
  elif resize_mode == 'RESHAPE':
    return RESIZE_MODE_RESHAPE
  else:
    raise ValueError('resize_mode %s is not in one of the allowed resize modes.' %(resize_mode))

def get_image_height_width(image):
  """Given an image, returns two tensors representing its height and width."""
  if isinstance(image, tf.Tensor):
    dims = _ImageDimensions(image, len(image.shape))
  elif isinstance(image, np.ndarray):
    dims = image.shape
  else:
    raise ValueError('Unsupported image type.')
  if len(image.shape) == 2 or len(image.shape) == 3:
    # Assume format (height, width, num_channels)
    return dims[0], dims[1]
  elif len(image.shape) == 4:
    # Assume format (batch_size, height, width, num_channels)
    return dims[1], dims[2]
  else:
    raise ValueError('\'images\' must have either 2, 3 or 4 dimensions.')

def _random_crop_to_hw(image, new_hw, height=None, width=None):
  if height is None or width is None:
    height, width = get_image_height_width(image)

  min_float = tf.minimum(height, width)
  min_hw = tf.cast(min_float, tf.int32)
  # This is to prevent from cropping to a size larger than the original size of the image.
  dst = tf.cond(tf.greater(new_hw, min_hw),
                lambda: tf.image.resize_images(image, size=[new_hw, new_hw], method=tf.image.ResizeMethod.BILINEAR, ),
                lambda: image)
  dst = tf.random_crop(dst, (new_hw, new_hw, int(dst.shape[2])))
  return dst

def resize_image(image, resize_mode, new_hw):
  if isinstance(resize_mode, str):
    resize_mode = resize_mode_str_to_int(resize_mode)
  assert isinstance(resize_mode, int), 'resize_mode must be an integer.'

  height, width = get_image_height_width(image)

  if resize_mode == RESIZE_MODE_PAD:
    size_float = tf.maximum(height, width)
    size = tf.cast(size_float, tf.int32)
    # pad to correct ratio
    oh = tf.cast((size_float - height) // 2, tf.int32)
    ow = tf.cast((size_float - width) // 2, tf.int32)
    dst = tf.cond(tf.not_equal(height, width), lambda: tf.image.pad_to_bounding_box(image, oh, ow, size, size),
                  lambda: image)
    # dst = tf.image.pad_to_bounding_box(image, oh, ow, size, size)
  elif resize_mode == RESIZE_MODE_CROP:
    # crop to correct ratio
    size_float = tf.minimum(height, width)
    size = tf.cast(size_float, tf.int32)
    oh = tf.cast((height - size_float) // 2, tf.int32)
    ow = tf.cast((width - size_float) // 2, tf.int32)
    dst = tf.cond(tf.not_equal(height, width), lambda: tf.image.crop_to_bounding_box(image, oh, ow, size, size),
                  lambda: image)
  elif resize_mode == RESIZE_MODE_RANDOM_CROP:
    dst = _random_crop_to_hw(image, new_hw, height=height, width=width)
  elif resize_mode == RESIZE_MODE_RANDOM_CROP_AND_RESHAPE:
    assert FLAGS.random_crop_and_reshape_initial_crop_hw > 0
    dst = _random_crop_to_hw(image, FLAGS.random_crop_and_reshape_initial_crop_hw, height=height, width=width)
    # Reshaping is done after the current if clauses.
  elif resize_mode == RESIZE_MODE_RESHAPE:
    dst = image
  elif resize_mode == RESIZE_MODE_NONE:
    dst = image
    return dst
  else:
    raise AttributeError('Resize mode %s not supported.' % (resize_mode))

  # square_size = tf.shape(dst, name='resized_image_shape')[0]
  # The BICUBIC method causes the output to exceed 0~255...
  # dst = tf.cond(tf.greater_equal(new_hw, square_size),
  #               lambda: tf.image.resize_images(dst, size=[new_hw, new_hw],
  #                                              method=tf.image.ResizeMethod.AREA, ),
  #               lambda: tf.image.resize_images(dst, size=[new_hw, new_hw],
  #                                              method=tf.image.ResizeMethod.BICUBIC, ))

  # RESIZE_MODE_RANDOM_CROP already has the correct image size.
  if resize_mode != RESIZE_MODE_RANDOM_CROP:
    dst = tf.image.resize_images(dst, size=[new_hw, new_hw], method=tf.image.ResizeMethod.BILINEAR, )
  return dst

def transform(W, img):
  shape = img.shape
  img = tf.tensordot(img, tf.matrix_transpose(W), axes=1)
  img = tf.reshape(img, shape=shape)
  return img

def rgb_to_yiq(img):
  W = np.array([[0.299, 0.587, 0.114],
                [0.596, -0.274, -0.322],
                [0.211, -0.523, 0.312]],
               dtype=np.float32)
  W = tf.constant(W, name='rgb_to_yiq_mat')
  return transform(W, img)


def yiq_to_rgb(img):
  W = np.array([[1, 0.956, 0.621],
                [1, -0.272, -0.647],
                [1, -1.106, 1.703]],
               dtype=np.float32)
  W = tf.constant(W, name='yiq_to_rgb_mat')
  return transform(W, img)  # transform(W, img) * 255.0

def random_flip_left_right(image, seed=None, random_var=None, return_random_var=False):
  """Randomly flip an image horizontally (left to right).

  With a 1 in 2 chance, outputs the contents of `image` flipped along the
  second dimension, which is `width`.  Otherwise output the image as-is.

  Args:
    image: A 3-D tensor of shape `[height, width, channels].`
    seed: A Python integer. Used to create a random seed. See
      @{tf.set_random_seed}
      for behavior.
    random_var: A scalar tensor or None. If none, a new random variable
      will be created.
    return_random_var: A boolean. If true, the random variable used
      is also returned as a second argument.

  Returns:
    A 3-D tensor of the same type and shape as `image`. Maybe return
    a scalar random variable.

  Raises:
    ValueError: if the shape of `image` not supported.
  """
  image = tf.convert_to_tensor(image, name='image')
  if random_var is None:
    uniform_random = tf.random_uniform([], 0, 1.0, seed=seed)
  else:
    uniform_random = random_var
  mirror_cond = tf.less(uniform_random, .5)
  ret = tf.cond(mirror_cond,
                   lambda: tf.reverse(image, [1]),
                   lambda: image)
  if return_random_var:
    return ret, uniform_random
  else:
    return ret


def random_brightness(image, max_delta, seed=None, random_var=None, return_random_var=False):
  """Adjust the brightness of images by a random factor.

  Equivalent to `adjust_brightness()` using a `delta` randomly picked in the
  interval `[-max_delta, max_delta)`.

  Args:
    image: An image.
    max_delta: float, must be non-negative.
    seed: A Python integer. Used to create a random seed. See
      @{tf.set_random_seed}
      for behavior.

  Returns:
    The brightness-adjusted image.

  Raises:
    ValueError: if `max_delta` is negative.
  """
  if max_delta < 0:
    raise ValueError('max_delta must be non-negative.')

  if random_var is None:
    delta = tf.random_uniform([], -max_delta, max_delta, seed=seed)
  else:
    delta = random_var
  ret = tf.image.adjust_brightness(image, delta)
  if return_random_var:
    return ret, delta
  else:
    return ret


def random_contrast(image, lower, upper, seed=None, random_var=None, return_random_var=False):
  """Adjust the contrast of an image by a random factor.

  Equivalent to `adjust_contrast()` but uses a `contrast_factor` randomly
  picked in the interval `[lower, upper]`.

  Args:
    image: An image tensor with 3 or more dimensions.
    lower: float.  Lower bound for the random contrast factor.
    upper: float.  Upper bound for the random contrast factor.
    seed: A Python integer. Used to create a random seed. See
      @{tf.set_random_seed}
      for behavior.

  Returns:
    The contrast-adjusted tensor.

  Raises:
    ValueError: if `upper <= lower` or if `lower < 0`.
  """
  if upper <= lower:
    raise ValueError('upper must be > lower.')

  if lower < 0:
    raise ValueError('lower must be non-negative.')

  # Generate an a float in [lower, upper]
  if random_var is None:
    contrast_factor = tf.random_uniform([], lower, upper, seed=seed)
  else:
    contrast_factor = random_var
  ret = tf.image.adjust_contrast(image, contrast_factor)
  if return_random_var:
    return ret, contrast_factor
  else:
    return ret

def mean_image_subtraction(image, means):
  """Subtracts the given means from each image channel.

  Taken from vgg_preprocessing.py.
  For example:
    means = [123.68, 116.779, 103.939]
    image = _mean_image_subtraction(image, means)

  Note that the rank of `image` must be known.

  Args:
    image: a tensor of size [height, width, C].
    means: a C-vector of values to subtract from each channel.

  Returns:
    the centered image.

  Raises:
    ValueError: If the rank of `image` is unknown, if `image` has a rank other
      than three or if the number of channels in `image` doesn't match the
      number of values in `means`.
  """
  if image.get_shape().ndims not in {3, 4}:
    raise ValueError('Input must be of size [height, width, C>0]')
  num_channels = image.get_shape().as_list()[-1]
  if len(means) != num_channels:
    raise ValueError('len(means) must match the number of channels')

  channels = tf.split(axis=-1, num_or_size_splits=num_channels, value=image)
  for i in range(num_channels):
    channels[i] -= means[i]
  return tf.concat(axis=-1, values=channels)

def random_crop_image(image, crop_ratio, resize_hw=None):
  # Randomly crop the image to `crop_ratio` of its given size. If resize_hw is not none, resize the image afterwards.
  assert len(image.shape) == 3

  cropped_min_height = int(int(image.shape[0]) * crop_ratio)
  cropped_min_width = int(int(image.shape[1]) * crop_ratio)
  cropped_height = tf.cast(int(image.shape[0]) * tf.random_uniform([], crop_ratio, 1.0), tf.int32)
  cropped_width = tf.cast(int(image.shape[1]) * tf.random_uniform([], crop_ratio, 1.0), tf.int32)
  if cropped_min_height != image.shape[0] or cropped_min_width != image.shape[1]:
    ret = tf.random_crop(image, (cropped_height, cropped_width, int(image.shape[2])))
    if resize_hw:
      size = [resize_hw, resize_hw]
    else:
      size = [int(image.shape[0]), int(image.shape[1])]
    ret = tf.image.resize_images(ret, size=size, method=tf.image.ResizeMethod.BILINEAR, )
  else:
    ret = image
  return ret