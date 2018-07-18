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
"""Miscellaneous helper functions."""

import collections
import io
import json
import os
import os.path

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from PIL import Image


######################
# Tensorflow related #
######################

def get_latest_checkpoint_path(path):
  """Either returns the latest checkpoint from `path` directory, or returns `path` if `path` is a file."""
  if not path:
    raise IOError('`path` cannot be empty.')
  if tf.gfile.IsDirectory(path):
    checkpoint_path = tf.train.latest_checkpoint(path)
  else:
    checkpoint_path = path
  return checkpoint_path


# Taken from https://github.com/tensorflow/tensorflow/issues/8246 by qianyizhang.
def tf_repeat(tensor, repeats):
  """
  Args:

  input: A Tensor. 1-D or higher.
  repeats: A list. Number of repeat for each dimension, length must be the same as the number of dimensions in input

  Returns:

  A Tensor. Has the same type as input. Has the shape of tensor.shape * repeats
  """
  if isinstance(repeats, tuple):
    repeats = list(repeats)
  assert len(repeats) == len(tensor.shape), 'repeat length must be the same as the number of dimensions in input.'
  with tf.variable_scope("repeat"):
    expanded_tensor = tf.expand_dims(tensor, -1)
    multiples = [1] + repeats
    tiled_tensor = tf.tile(expanded_tensor, multiples=multiples)
    new_shape = tf.TensorShape([tensor.shape[i] * repeats[i] for i in range(len(repeats))])
    repeated_tensor = tf.reshape(tiled_tensor, new_shape)
  return repeated_tensor


def fp16_friendly_leaky_relu(features, alpha=0.2, name=None):
  """Compute the Leaky ReLU activation function.

  "Rectifier Nonlinearities Improve Neural Network Acoustic Models"
  AL Maas, AY Hannun, AY Ng - Proc. ICML, 2013
  http://web.stanford.edu/~awni/papers/relu_hybrid_icml2013_final.pdf

  Args:
    features: A `Tensor` representing preactivation values.
    alpha: Slope of the activation function at x < 0.
    name: A name for the operation (optional).

  Returns:
    The activation value.
  """
  with tf.name_scope(name, "LeakyRelu", [features, alpha]):
    features = tf.convert_to_tensor(features, name="features", dtype=features.dtype)
    alpha = tf.convert_to_tensor(alpha, name="alpha", dtype=features.dtype)
    return tf.maximum(alpha * features, features)


def safe_one_hot_encoding(tensor, num_classes, dtype=None):
  """Given a (possibly out of range) vector of labels, transform them into one-hot encoding."""
  one_hot_encoded = slim.one_hot_encoding(
    tensor, num_classes, on_value=tf.constant(1, tf.int64),
    off_value=tf.constant(0, tf.int64))

  # This makes sure that when there are no labels, reduce_max will not output -inf but output 0s instead.
  stacked = tf.concat((tf.zeros((1, num_classes), dtype=one_hot_encoded.dtype),
                       one_hot_encoded), axis=0)
  tensor = tf.reduce_max(stacked, 0)
  if dtype is not None:
    tensor = tf.cast(tensor, dtype)
  return tensor


def get_image_height(image):
  """Assumes image shape is known."""
  return int(image.shape[-3])


def get_image_width(image):
  """Assumes image shape is known."""
  return int(image.shape[-2])


def grayscale_to_heatmap(gray, is_bgr=False):
  four = tf.constant(4, dtype=gray.dtype)
  zero = tf.constant(0, dtype=gray.dtype)
  one = tf.constant(1, dtype=gray.dtype)

  r = tf.clip_by_value(tf.minimum(four * gray - tf.constant(1.5, dtype=gray.dtype),
                                  -four * gray + tf.constant(4.5, dtype=gray.dtype)), zero, one)
  g = tf.clip_by_value(tf.minimum(four * gray - tf.constant(0.5, dtype=gray.dtype),
                                  -four * gray + tf.constant(3.5, dtype=gray.dtype)), zero, one)
  b = tf.clip_by_value(tf.minimum(four * gray + tf.constant(0.5, dtype=gray.dtype),
                                  -four * gray + tf.constant(2.5, dtype=gray.dtype)), zero, one)
  if is_bgr:
    return tf.concat((b, g, r), axis=-1)
  else:
    return tf.concat((r, g, b), axis=-1)


def extract_random_patches(image, patch_sizes, num_patches):
  """Extracts random patches from an image tensor."""
  if isinstance(patch_sizes, int):
    ksizes = [1, patch_sizes, patch_sizes, 1]
  elif isinstance(patch_sizes, list) or isinstance(patch_sizes, tuple):
    if len(patch_sizes) == 2:
      ksizes = [1, patch_sizes[0], patch_sizes[1], 1]
    elif len(patch_sizes) == 4:
      ksizes = patch_sizes
    else:
      raise ValueError('patch_sizes must be length 2 or length 4.')
  else:
    raise ValueError('patch_sizes must be a length 2 or length 4 list, or an int.')

  # (batch, height, width, patch_size * patch_size * feature)
  patches = tf.extract_image_patches(image, ksizes=ksizes,
                                     strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='VALID')
  patches_shape = tf.unstack(tf.shape(patches))
  patches_hw = patches_shape[1] * patches_shape[2]
  patches = tf.reshape(patches, shape=[patches.shape[0], patches.shape[1] * patches.shape[2], patches.shape[3]])

  def gather_random(p):
    random_indices = tf.random_uniform([num_patches], minval=0, maxval=patches_hw, dtype=tf.int32)
    return tf.gather(p, random_indices)

  # A little bit hard to do random per batch, so I just do the same random patches for all batches.

  ret = tf.map_fn(gather_random, patches, name='random_patches')
  ret = tf.reshape(ret, shape=[ret.shape[0] * ret.shape[1], ksizes[1], ksizes[2], image.shape[3]])
  return ret


#######################
# Miscellaneous Utils #
#######################

def combine_dicts(name_to_dict):
  """Given a dictionary of (name, child_dict), return the combined version of each item in the child dictionary."""
  combined = {}
  for dict_name, current_dict in name_to_dict.iteritems():
    for name, val in current_dict.iteritems():
      combined[dict_name + '_' + name] = val
  return combined


def safe_print(string, *args):
  """Calls string.format() without raising errors like invalid encoding etc."""
  try:
    print string.format(*args)
  except:
    try:
      print unicode(string).format(*args)
    except:
      print('Cannot print string. Moving on without raising an error.')


def get_no_ext_base(file_name):
  return os.path.splitext(os.path.basename(file_name))[0]


def encoded_image_to_numpy(encoded_image):
  """Given an encoded image (e.g. tf.encode_image()), output it's corresponding numpy image."""
  ret = np.asarray(Image.open(io.BytesIO(encoded_image)))
  return ret


def im2gray(image):
  '''Turn numpy images into grayscale.'''
  if len(image.shape) == 2:
    return image
  image = image.astype(np.float32)
  # Use the Conversion Method in This Paper:
  # [http://www.eyemaginary.com/Rendering/TurnColorsGray.pdf]
  if image.shape[-1] == 1:
    image_gray = image
  elif image.shape[-1] == 3:
    image_gray = np.dot(image, [[0.2989], [0.5870], [0.1140]])
  elif image.shape[-1] == 4:
    # May be inaccurate since we lose the a channel.
    image_gray = np.dot(image[..., :3], [[0.2989], [0.5870], [0.1140]])
  else:
    raise NotImplementedError
  return image_gray


#######################
# Anime Face related  #
#######################

def process_anime_face_labels(labels, classification_threshold, labels_id_to_group=None):
  """Given a numpy array of labels and the groups each label belongs to, output the maximum values for each group
  and set the non-max to 0."""
  if labels_id_to_group is None:
    labels_id_to_group = get_tags_dict('./datasets/anime_face_tag_list.txt', 0, 2)

  ret = [0.0 for _ in range(len(labels))]
  group_vals = collections.defaultdict(list)
  for i, val in enumerate(labels):
    group = labels_id_to_group.get(i, None)
    if group is not None:
      group_vals[group].append((i, val))

  hair_color_missing = True
  eye_color_missing = True

  for group, vals in group_vals.iteritems():
    max_item = max(vals, key=lambda x: x[1])
    ret[max_item[0]] = max_item[1]
    if group == '2':
      if max_item[1] >= classification_threshold:
        hair_color_missing = False
    if group == '3':
      if max_item[1] >= classification_threshold:
        eye_color_missing = False

  # Do not output any label if eye color or hair color is missing.
  if eye_color_missing or hair_color_missing:
    return [0.0 for _ in range(len(labels))]
  else:
    return ret


def get_tags_dict(path='./datasets/anime_face_tag_list.txt', key_column_index=0, value_column_index=2):
  """Opens a tab separated file and returns a dictionary with specified key and value column."""
  ret = {}
  with open(path, 'r') as f:
    for i, line in enumerate(f):
      if len(line):
        whole_line = line.rstrip('\n')
        content = whole_line.split('\t')
        key = i if key_column_index is None else int(content[key_column_index])
        value = whole_line if value_column_index is None else content[value_column_index]
        ret[key] = value
  return ret


def get_landmark_dict(directories, landmark_file_name, do_join=True):
  """Converts the anime face landmark detection output json to a dictionary with key = filename and val = landmarks."""
  ret = collections.defaultdict(list)
  if do_join:
    for directory in directories:
      landmark_file_path = os.path.join(directory, landmark_file_name)
      with open(landmark_file_path) as f:
        landmarks = collections.defaultdict(list)
        for line in f:
          line = line.rstrip()
          if line:
            landmark = json.loads(line)
            if 'file' in landmark:
              landmarks[os.path.basename(landmark['file'])].append(landmark)
      ret.update(landmarks)
  else:
    with open(landmark_file_name) as f:
      for line in f:
        line = line.rstrip()
        if line:
          landmark = json.loads(line)
          if 'file' in landmark:
            ret[os.path.basename(landmark['file'])].append(landmark)
  return ret


def get_relative_xywh(json_object, relative_to_x, relative_to_y, width, height):
  """Returns the xywh value relative to the height and width of the image."""
  x, y, w, h = _get_xywh(json_object)
  relative_x = x - relative_to_x
  relative_y = y - relative_to_y
  if relative_x < 0 or relative_y < 0:
    raise ValueError('relative_x < 0 or relative_y < 0: relative_x = %d relative_y = %d' % (relative_x, relative_y))
  if (relative_x + w) / float(width) >= 1 or (relative_y + h) / float(height) >= 1:
    raise ValueError('(relative_x + w) / float(width) >= 1 or (relative_y + h) / float(height) >= 1:'
                     ' (relative_x + w) / float(width) = %f (relative_y + h) / float(height) = %f'
                     % ((relative_x + w) / float(width), (relative_y + h) / float(height)))
  return relative_x / float(width), relative_y / float(height), w / float(width), h / float(height)


def _get_xywh(json_object):
  return json_object['x'], json_object['y'], json_object['height'], json_object['width']


def expand_xywh(x, y, w, h, image_w, image_h, hw_expansion_rate):
  """Expand h, w on each side by `hw_expansion_rate`."""
  x_expanded = max(0, x - int(w * hw_expansion_rate))
  y_expanded = max(0, y - int(h * hw_expansion_rate))
  x_end_expanded = min(image_w, x + int(w * (1.0 + hw_expansion_rate)))
  y_end_expanded = min(image_h, y + int(h * (1.0 + hw_expansion_rate)))
  return x_expanded, y_expanded, x_end_expanded - x_expanded, y_end_expanded - y_expanded,


def unevenly_expand_xywh(x, y, w, h, image_w, image_h, left_w_ratio, right_w_ratio, top_h_ratio, bottom_h_ratio):
  """Expand each side by their respective ratio."""
  x_expanded = max(0, x - int(w * left_w_ratio))
  y_expanded = max(0, y - int(h * top_h_ratio))
  x_end_expanded = min(image_w, x + int(w * (1.0 + right_w_ratio)))
  y_end_expanded = min(image_h, y + int(h * (1.0 + bottom_h_ratio)))
  return x_expanded, y_expanded, x_end_expanded - x_expanded, y_end_expanded - y_expanded,


def get_faces(object_detection_result, width, height):
  faces = object_detection_result['faces']
  # x, y, w, h
  ret = [
    [int(face[2] * width), int(face[1] * height), int((face[4] - face[2]) * width), int((face[3] - face[1]) * height), ]
    for face in faces]
  return ret


def find_boundary(point, image, direction, num_pixels, threshold):
  """For a point on an image, find the first pixel in `direction` that has the next N pixels values >= threshold."""
  assert len(image.shape) == 3 and len(point) == 2 and num_pixels > 0
  hw = (image.shape[0], image.shape[1])
  ret = [point[0], point[1]]
  # height, width  =
  if direction == 'up':
    axis = 0
    candidates = image[:, point[1]]
    i = point[0] - 1
    change = -1
  elif direction == 'down':
    axis = 0
    candidates = image[:, point[1]]
    i = point[0] + 1
    change = 1
  elif direction == 'left':
    axis = 1
    candidates = image[point[0], :]
    i = point[1] - 1
    change = -1
  elif direction == 'right':
    axis = 1
    candidates = image[point[0], :]
    i = point[1] + 1
    change = 1
  else:
    raise ValueError('unsupported direction %s' % direction)

  failed = True
  min_i = 0
  max_i = hw[axis] - 1
  while i > min_i and i < max_i and failed:
    failed = False
    for j in range(num_pixels):
      if i + j * change < min_i or i + j * change > max_i:
        failed = False
        break
      if np.all(candidates[i + j * change] < threshold):
        failed = True
        break
    if failed:
      i = max(min_i, min(max_i, i + j * change + change))
  ret[axis] = i
  return ret


def string_list_to_float_list(string_list):
  return map(float, string_list)