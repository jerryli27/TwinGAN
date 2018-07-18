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
r"""Downloads and converts svhn data to TFRecords of TF-Example protos.

This module downloads the svhn data, uncompresses it, reads the files
that make up the svhn data and creates two TFRecord datasets: one for train
and one for test. Each TFRecord dataset is comprised of a set of TF-Example
protocol buffers, each of which contain a single image and label.

The script should take about a minute to run.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import sys

import scipy.io
import numpy as np
from six.moves import urllib
import tensorflow as tf

from datasets import dataset_utils

# The URLs where the svhn data can be downloaded.
_DATA_URL = 'http://ufldl.stanford.edu/housenumbers/'
_TRAIN_DATA_FILENAME = 'train_32x32.mat'
# _TRAIN_LABELS_FILENAME = 'train-labels-idx1-ubyte.gz'
_TEST_DATA_FILENAME = 'test_32x32.mat'
# _TEST_LABELS_FILENAME = 't10k-labels-idx1-ubyte.gz'

_IMAGE_SIZE = 32
_NUM_CHANNELS = 3  # 1

# The names of the classes.
_CLASS_NAMES = [
    'zero',
    'one',
    'two',
    'three',
    'four',
    'five',
    'size',
    'seven',
    'eight',
    'nine',
]

pixel_depth = 255.0  # Number of levels per pixel.

def im2gray(image):
  '''Normalize images'''
  image = image.astype(float)
  # Use the Conversion Method in This Paper:
  # [http://www.eyemaginary.com/Rendering/TurnColorsGray.pdf]
  image_gray = np.dot(image, [[0.2989],[0.5870],[0.1140]])
  return image_gray


def GCN(image, min_divisor=1e-4):
  """Global Contrast Normalization"""

  imsize = image.shape[0]
  mean = np.mean(image, axis=(1, 2), dtype=float)
  std = np.std(image, axis=(1, 2), dtype=float, ddof=1)
  std[std < min_divisor] = 1.
  image_GCN = np.zeros(image.shape, dtype=float)

  for i in np.arange(imsize):
    image_GCN[i, :, :] = (image[i, :, :] - mean[i]) / std[i]

  return image_GCN

def _extract_images(filename, num_images):
  """Extract the images into a numpy array.

  Args:
    filename: The path to an svhn images file.
    num_images: The number of images in the file.

  Returns:
    A numpy array of shape [number_of_images, height, width, channels].
  """
  print('Extracting images from: ', filename)
  # with gzip.open(filename) as bytestream:
  #   bytestream.read(16)
  #   buf = bytestream.read(
  #       _IMAGE_SIZE * _IMAGE_SIZE * num_images * _NUM_CHANNELS)
  #   data = np.frombuffer(buf, dtype=np.uint8)
  #   data = data.reshape(num_images, _IMAGE_SIZE, _IMAGE_SIZE, _NUM_CHANNELS)

  data = scipy.io.loadmat(filename, variable_names='X').get('X')
  data = np.transpose(data, (3,0,1,2))
  # data = im2gray(data)[:, :, :, 0]
  # data_GCN = GCN(data)
  # print(data_GCN.shape)
  # data_GCN = np.expand_dims(data_GCN, 3)
  # return data_GCN
  return data

def _extract_labels(filename, num_labels):
  """Extract the labels into a vector of int64 label IDs.

  Args:
    filename: The path to an svhn labels file.
    num_labels: The number of labels in the file.

  Returns:
    A numpy array of shape [number_of_labels]
  """
  print('Extracting labels from: ', filename)
  # with gzip.open(filename) as bytestream:
  #   bytestream.read(8)
  #   buf = bytestream.read(1 * num_labels)
  #   labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
  labels = scipy.io.loadmat(filename, variable_names='y').get('y')
  labels[labels == 10] = 0
  labels = labels[:,0]
  print(labels.shape)
  return labels


def _add_to_tfrecord(data_filename, num_images,
                     tfrecord_writer):
  """Loads data from the binary svhn files and writes files to a TFRecord.

  Args:
    data_filename: The filename of the svhn images.
    labels_filename: The filename of the svhn labels.
    num_images: The number of images in the dataset.
    tfrecord_writer: The TFRecord writer to use for writing.
  """
  images = _extract_images(data_filename, num_images)
  labels = _extract_labels(data_filename, num_images)

  shape = (_IMAGE_SIZE, _IMAGE_SIZE, _NUM_CHANNELS)
  with tf.Graph().as_default():
    image = tf.placeholder(dtype=tf.uint8, shape=shape)
    encoded_png = tf.image.encode_png(image)

    with tf.Session('') as sess:
      for j in range(num_images):
        sys.stdout.write('\r>> Converting image %d/%d' % (j + 1, num_images))
        sys.stdout.flush()

        png_string = sess.run(encoded_png, feed_dict={image: images[j]})

        example = dataset_utils.image_to_tfexample(
            png_string, 'png'.encode(), _IMAGE_SIZE, _IMAGE_SIZE, labels[j])
        tfrecord_writer.write(example.SerializeToString())


def _get_output_filename(dataset_dir, split_name):
  """Creates the output filename.

  Args:
    dataset_dir: The directory where the temporary files are stored.
    split_name: The name of the train/test split.

  Returns:
    An absolute file path.
  """
  return '%s/%s-.tfrecord' % (dataset_dir, split_name)


def _download_dataset(dataset_dir):
  """Downloads svhn locally.

  Args:
    dataset_dir: The directory where the temporary files are stored.
  """
  for filename in [_TRAIN_DATA_FILENAME,
                   # _TRAIN_LABELS_FILENAME,
                   _TEST_DATA_FILENAME,
                   # _TEST_LABELS_FILENAME,
                   ]:
    filepath = os.path.join(dataset_dir, filename)

    if not os.path.exists(filepath):
      print('Downloading file %s...' % filename)
      def _progress(count, block_size, total_size):
        sys.stdout.write('\r>> Downloading %.1f%%' % (
            float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()
      filepath, _ = urllib.request.urlretrieve(_DATA_URL + filename,
                                               filepath,
                                               _progress)
      print()
      with tf.gfile.GFile(filepath) as f:
        size = f.size()
      print('Successfully downloaded', filename, size, 'bytes.')


def _clean_up_temporary_files(dataset_dir):
  """Removes temporary files used to create the dataset.

  Args:
    dataset_dir: The directory where the temporary files are stored.
  """
  for filename in [_TRAIN_DATA_FILENAME,
                   # _TRAIN_LABELS_FILENAME,
                   _TEST_DATA_FILENAME,
                   # _TEST_LABELS_FILENAME
                   ]:
    filepath = os.path.join(dataset_dir, filename)
    tf.gfile.Remove(filepath)


def run(dataset_dir):
  """Runs the download and conversion operation.

  Args:
    dataset_dir: The dataset directory where the dataset is stored.
  """
  if not tf.gfile.Exists(dataset_dir):
    tf.gfile.MakeDirs(dataset_dir)

  training_filename = _get_output_filename(dataset_dir, 'train')
  testing_filename = _get_output_filename(dataset_dir, 'test')

  if tf.gfile.Exists(training_filename) and tf.gfile.Exists(testing_filename):
    print('Dataset files already exist. Exiting without re-creating them.')
    return

  _download_dataset(dataset_dir)

  # First, process the training data:
  with tf.python_io.TFRecordWriter(training_filename) as tfrecord_writer:
    data_filename = os.path.join(dataset_dir, _TRAIN_DATA_FILENAME)
    # labels_filename = os.path.join(dataset_dir, _TRAIN_LABELS_FILENAME)
    _add_to_tfrecord(data_filename, 60000, tfrecord_writer)

  # Next, process the testing data:
  with tf.python_io.TFRecordWriter(testing_filename) as tfrecord_writer:
    data_filename = os.path.join(dataset_dir, _TEST_DATA_FILENAME)
    # labels_filename = os.path.join(dataset_dir, _TEST_LABELS_FILENAME)
    _add_to_tfrecord(data_filename, 10000, tfrecord_writer)

  # Finally, write the labels file:
  labels_to_class_names = dict(zip(range(len(_CLASS_NAMES)), _CLASS_NAMES))
  dataset_utils.write_label_file(labels_to_class_names, dataset_dir)

  _clean_up_temporary_files(dataset_dir)
  print('\nFinished converting the svhn dataset!')

if __name__ == '__main__':
    run('../data/svhn/')