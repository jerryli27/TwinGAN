# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains utilities for downloading and converting datasets."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tarfile
import random
from abc import ABCMeta
from abc import abstractmethod

from six.moves import urllib
import cv2
import numpy as np
import tensorflow as tf
import tensorflow.contrib.lookup as lookup
import tensorflow.contrib.slim as slim

import util_misc

LABELS_FILENAME = 'labels.txt'
TAG_TEXT_DELIMITER = ', '


def int64_feature(values):
  """Returns a TF-Feature of int64s.

  Args:
    values: A scalar or list of values.

  Returns:
    A TF-Feature.
  """
  if not isinstance(values, (tuple, list, np.ndarray)):
    values = [values]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_feature(values):
  """Returns a TF-Feature of bytes.

  Args:
    values: A string.

  Returns:
    A TF-Feature.
  """
  if not isinstance(values, (tuple, list, np.ndarray)):
    values = [values]
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))


def float_feature(values):
  """Returns a TF-Feature of floats.

  Args:
    values: A scalar of list of values.

  Returns:
    A TF-Feature.
  """
  if not isinstance(values, (tuple, list, np.ndarray)):
    values = [values]
  return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def image_to_tfexample(image_data, image_format, height, width, class_id):
  return tf.train.Example(features=tf.train.Features(feature={
      'image/encoded': bytes_feature(image_data),
      'image/format': bytes_feature(image_format),
      'image/class/label': int64_feature(class_id),
      'image/height': int64_feature(height),
      'image/width': int64_feature(width),
  }))


def download_and_uncompress_tarball(tarball_url, dataset_dir):
  """Downloads the `tarball_url` and uncompresses it locally.

  Args:
    tarball_url: The URL of a tarball file.
    dataset_dir: The directory where the temporary files are stored.
  """
  filename = tarball_url.split('/')[-1]
  filepath = os.path.join(dataset_dir, filename)

  def _progress(count, block_size, total_size):
    sys.stdout.write('\r>> Downloading %s %.1f%%' % (
        filename, float(count * block_size) / float(total_size) * 100.0))
    sys.stdout.flush()
  filepath, _ = urllib.request.urlretrieve(tarball_url, filepath, _progress)
  print()
  statinfo = os.stat(filepath)
  print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  tarfile.open(filepath, 'r:gz').extractall(dataset_dir)


def write_label_file(labels_to_class_names, dataset_dir,
                     filename=LABELS_FILENAME):
  """Writes a file with the list of class names.

  Args:
    labels_to_class_names: A map of (integer) labels to class names.
    dataset_dir: The directory in which the labels file should be written.
    filename: The filename where the class names are written.
  """
  labels_filename = os.path.join(dataset_dir, filename)
  with tf.gfile.Open(labels_filename, 'w') as f:
    for label in labels_to_class_names:
      class_name = labels_to_class_names[label]
      f.write('%d:%s\n' % (label, class_name))


def has_labels(dataset_dir, filename=LABELS_FILENAME):
  """Specifies whether or not the dataset directory contains a label map file.

  Args:
    dataset_dir: The directory in which the labels file is found.
    filename: The filename where the class names are written.

  Returns:
    `True` if the labels file exists and `False` otherwise.
  """
  return tf.gfile.Exists(os.path.join(dataset_dir, filename))


def read_label_file(dataset_dir, filename=LABELS_FILENAME):
  """Reads the labels file and returns a mapping from ID to class name.

  Args:
    dataset_dir: The directory in which the labels file is found.
    filename: The filename where the class names are written.

  Returns:
    A map from a label (integer) to class name.
  """
  labels_filename = os.path.join(dataset_dir, filename)
  with tf.gfile.Open(labels_filename, 'rb') as f:
    lines = f.read().decode()
  lines = lines.split('\n')
  lines = filter(None, lines)

  labels_to_class_names = {}
  for line in lines:
    index = line.index(':')
    labels_to_class_names[int(line[:index])] = line[index+1:]
  return labels_to_class_names


def is_png(filename):
  """Determine if a file contains a PNG format image.

  Args:
    filename: string, path of the image file.

  Returns:
    boolean indicating if the image is a PNG.
  """
  base, ext = os.path.splitext(filename)
  return ext.lower() == '.png'


def is_jpeg(filename):
  """Determine if a file contains a JPG format image.

  Args:
    filename: string, path of the image file.

  Returns:
    boolean indicating if the image is a JPG.
  """
  base, ext = os.path.splitext(filename)
  return ext.lower() == '.jpg' or ext.lower() == '.jpeg'


def get_height_width_ratio(height, width):
  if width == 0:
    return sys.float_info.max
  return float(height) / width

def is_image_blurry(image, threshold = 100):
  """Returns true if the image is blurry."""
  blur_map = cv2.Laplacian(image, cv2.CV_64F)
  score = np.var(blur_map)
  return (score < threshold)


class ImageCoder(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self, preprocess_fn = None):
    # Create a single Session to run all image coding calls.
    # All images must either be in the same format, or have their encoding method recorded.

    # Initializes function that converts PNG to JPEG data.
    self._png_data = tf.placeholder(dtype=tf.string)
    self._decode_png = tf.image.decode_png(self._png_data)
    self._png_to_jpeg = tf.image.encode_jpeg(self._decode_png, format='rgb', quality=100)

    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

    #
    self._decode_image_data = tf.placeholder(dtype=tf.string)
    image = tf.image.decode_image(self._decode_image_data, channels=3)
    self._image_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

    self._array_image = tf.placeholder(shape=[None, None, None], dtype=tf.uint8)
    self._encode_array_to_jpeg = tf.image.encode_jpeg(self._array_image, format='rgb', quality=100)


    if preprocess_fn:
      image.set_shape([None, None, 3])
      self._decode_preprocessed_image = preprocess_fn(image)
      assert self._decode_preprocessed_image.dtype == tf.uint8
      le_255 = tf.assert_less_equal(self._decode_preprocessed_image, tf.constant(255, tf.uint8))
      ge_0 = tf.assert_non_negative(self._decode_preprocessed_image)
      with tf.control_dependencies([le_255, ge_0]):
        format = 'grayscale' if self._decode_preprocessed_image.shape[-1] == 1 else 'rgb'
        self._image_to_preprocessed_jpeg = tf.image.encode_jpeg(self._decode_preprocessed_image, format=format, quality=100)
        self._image_preprocessed_shape = tf.shape(self._decode_preprocessed_image)
    else:
      self._image_to_preprocessed_jpeg = None
      self._image_preprocessed_shape = None

  def set_session(self, session):
    self._sess = session
    init = tf.global_variables_initializer()
    self._sess.run([init])

  def maybe_initialize_session(self):
    if not hasattr(self, '_sess') or self._sess is None:
      self._sess = tf.Session()
      init = tf.global_variables_initializer()
      self._sess.run([init])

  def png_to_jpeg(self, image_data):
    self.maybe_initialize_session()
    return self._sess.run(self._png_to_jpeg,
                          feed_dict={self._png_data: image_data})

  def decode_jpeg(self, image_data):
    self.maybe_initialize_session()
    image = self._sess.run(self._decode_jpeg,
                           feed_dict={self._decode_jpeg_data: image_data})
    return image

  def decode_png(self, image_data):
    self.maybe_initialize_session()
    image = self._sess.run(self._decode_png,
                           feed_dict={self._png_data: image_data})
    return image

  def image_to_jpeg(self, image_data):
    self.maybe_initialize_session()
    image = self._sess.run(self._image_to_jpeg,
                           feed_dict={self._decode_image_data: image_data})
    return image

  def image_to_preprocessed_jpeg(self, image_data, return_decoded=False):
    self.maybe_initialize_session()
    run_list = [self._image_to_preprocessed_jpeg, self._image_preprocessed_shape]
    if return_decoded:
      run_list.append(self._decode_preprocessed_image)
    return self._sess.run(run_list, feed_dict={self._decode_image_data: image_data})

  def array_to_jpeg(self, array):
    self.maybe_initialize_session()
    encoded_image = self._sess.run(self._encode_array_to_jpeg,
                                   feed_dict={self._array_image: array})
    return encoded_image

def process_image(filename, coder, do_preprocess, return_image_np=False):
  """Read and maybe preprocess a single image file.

  Args:
    filename: string, path to an image file e.g., '/path/to/example.JPG'.
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
  Returns:
    image_data: encoded RGB image read from `filename`.
    height: integer, image height in pixels.
    width: integer, image width in pixels.
    image: [height, width, channels] array of RGB image. Only returned when
      `return_image_np` is true.
  """
  # Read the image file.
  image_data = tf.gfile.FastGFile(filename, 'r').read()

  if do_preprocess:
    image_data, image_shape = coder.image_to_preprocessed_jpeg(image_data)
    height = image_shape[0]
    width = image_shape[1]
    assert image_shape[2] == 3
  else:
    # Convert all non-jpg data to jpg
    if not is_jpeg(filename):
      tf.logging.debug('Converting to JPEG for %s' % filename)
      image_data = coder.image_to_jpeg(image_data)

    # Decode the RGB JPEG.
    image = coder.decode_jpeg(image_data)

    # Check that image converted to RGB.
    assert len(image.shape) == 3
    height = image.shape[0]
    width = image.shape[1]
    assert image.shape[2] == 3
    assert np.max(image) <= 255.0 and np.min(image) >= 0.0

  if return_image_np:
    assert not do_preprocess, 'Currently return_image_np option is only supported when do_preprocess is off.'
    return image_data, height, width, image
  else:
    return image_data, height, width


def get_filenames(data_dir, acceptable_ext={'jpg', 'jpeg', 'png'}, do_shuffle=True, shuffle_seed=12345):
  """Returns a list of shuffled file names."""
  # Note in python 3.5 or later glob.glob(**) works. In py2.7 I have to use this helper function.
  if not data_dir.endswith('/'):
    data_dir = data_dir + "/"
  filenames = []
  for path, subdirs, files in os.walk(data_dir):
    for name in files:
      full_file_path = os.path.join(path, name)
      _, ext = os.path.splitext(name)
      # Get rid of the dot
      ext = ext.lstrip('.').lower()
      if acceptable_ext and ext in acceptable_ext:
        filenames.append(full_file_path)
  if len(filenames) == 0:
    raise AssertionError('There is no image in directory %s .' % data_dir)

  # Shuffle the ordering of all image files in order to guarantee
  # random ordering of the images with respect to label in the
  # saved TFRecord files. Make the randomization repeatable.
  if do_shuffle:
    shuffled_index = range(len(filenames))
    random.seed(shuffle_seed)
    random.shuffle(shuffled_index)
    filenames = [filenames[i] for i in shuffled_index]
  return filenames


def get_tags_id(filename, tags_dict, space_deliminator):
  tags_file_name = filename + '.txt'
  lines = tf.gfile.FastGFile(tags_file_name, 'r').readlines()

  if space_deliminator:
    tags = []
    for line in lines:
      current_tags = line.rstrip('\n').split(' ')
      tags += current_tags
  else:
    tags = [line.rstrip('\n') for line in lines]

  tags_id = [tags_dict.get(tag) for tag in tags if tag in tags_dict]
  return tags_id


def tag_to_human_readable(tag_indices, id_to_tag):
  return TAG_TEXT_DELIMITER.join(id_to_tag[tag_i] for tag_i in tag_indices)




class OneHotLabelTensor(slim.tfexample_decoder.Tensor):
  """A slim decoder to convert string labels into one-hot encoded tensors."""
  def __init__(self,
               tensor_key,
               tags_id_lookup_file,
               num_classes,
               dtype=tf.float32,
               shape_keys=None,
               shape=None,
               default_value='',
               tags_key_column_index=None,
               tags_value_column_index=None,
               delimiter=None):
    """Initializes the OneHotLabelTensor handler, which decode label text into one-hot encodings."""
    lookup_kwargs = {}
    if tags_key_column_index is not None:
      lookup_kwargs['key_column_index'] = tags_key_column_index
    if tags_value_column_index is not None:
      lookup_kwargs['value_column_index'] = tags_value_column_index
    table = lookup.index_table_from_file(tags_id_lookup_file, **lookup_kwargs)
    self._table = table
    self._delimiter = delimiter or TAG_TEXT_DELIMITER
    self._num_classes = num_classes
    self._dtype=dtype
    super(OneHotLabelTensor, self).__init__(tensor_key, shape_keys, shape, default_value)

  def tensors_to_item(self, keys_to_tensors):
    unmapped_tensor = super(OneHotLabelTensor, self).tensors_to_item(keys_to_tensors)
    labels_text_split = tf.string_split([unmapped_tensor], delimiter=self._delimiter)
    tensor = self._table.lookup(labels_text_split.values)
    tensor = util_misc.safe_one_hot_encoding(tensor, self._num_classes, dtype=self._dtype)
    return tensor

#####################
# tf example parser #
#####################
# tf example parser functions. Some are taken from the tensorflow object detection repo.

class DataToNumpyParser(object):
  __metaclass__ = ABCMeta

  @abstractmethod
  def parse(self, input_data):
    """Parses input and returns a numpy array or a dictionary of numpy arrays.

    Args:
      input_data: an input data

    Returns:
      A numpy array or a dictionary of numpy arrays or None, if input
      cannot be parsed.
    """
    pass


class FloatParser(DataToNumpyParser):
  """Tensorflow Example float parser."""

  def __init__(self, field_name):
    self.field_name = field_name

  def parse(self, tf_example):
    return np.array(
        tf_example.features.feature[self.field_name].float_list.value,
        dtype=np.float).transpose() if tf_example.features.feature[
            self.field_name].HasField("float_list") else None


class StringParser(DataToNumpyParser):
  """Tensorflow Example string parser."""

  def __init__(self, field_name):
    self.field_name = field_name

  def parse(self, tf_example):
    return "".join(tf_example.features.feature[self.field_name]
                   .bytes_list.value) if tf_example.features.feature[
                       self.field_name].HasField("bytes_list") else None


class Int64Parser(DataToNumpyParser):
  """Tensorflow Example int64 parser."""

  def __init__(self, field_name):
    self.field_name = field_name

  def parse(self, tf_example):
    return np.array(
        tf_example.features.feature[self.field_name].int64_list.value,
        dtype=np.int64).transpose() if tf_example.features.feature[
            self.field_name].HasField("int64_list") else None


class BoundingBoxParser(DataToNumpyParser):
  """Tensorflow Example bounding box parser."""

  def __init__(self, xmin_field_name, ymin_field_name, xmax_field_name,
               ymax_field_name):
    self.field_names = [
        ymin_field_name, xmin_field_name, ymax_field_name, xmax_field_name
    ]

  def parse(self, tf_example):
    result = []
    parsed = True
    for field_name in self.field_names:
      result.append(tf_example.features.feature[field_name].float_list.value)
      parsed &= (
          tf_example.features.feature[field_name].HasField("float_list"))

    return np.array(result).transpose() if parsed else None

def iterate_tfrecords(filenames, reader_opts=None):
  """
  Attempt to iterate over every record in the supplied iterable of TFRecord filenames
  :param filenames: iterable of filenames to read
  :param reader_opts: (optional) tf.python_io.TFRecordOptions to use when constructing the record iterator
  """
  i = 0
  for fname in filenames:
    print('validating ', fname)

    record_iterator = tf.python_io.tf_record_iterator(path=fname, options=reader_opts)
    try:
      for rec in record_iterator:
        tf_example = tf.train.Example()
        tf_example.ParseFromString(rec)
        yield tf_example
        i += 1
    except Exception as e:
      print('error in {} at record {}'.format(fname, i))
      print(e)