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
"""Adapted from tensorflow/models/research/slim/datasets/cifar10.py

To download the dataset, please see convert_celeba.py
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
from six.moves import urllib
import tensorflow as tf

from datasets import dataset_utils

import tensorflow.contrib.slim as slim

_FILE_PATTERN = '%s-*'

_ITEMS_TO_DESCRIPTIONS = {
    'source': 'The celebrity image',
    'conditional_labels':'The attributes of the image. e.g. Female.',
    'landmarks': 'Facial landmarks of eyes etc.',
    'filename': '',
}


FLAGS = tf.flags.FLAGS
DEFAULT_NUM_CLASSES = 40
_DEFAULT_TRAIN_SIZE = 162770  # For woman-only dataset, it is 94509 images.
_DEFAULT_VALIDATION_SIZE = 19867


def get_split(split_name, dataset_dir, file_pattern=None, reader=None):
  """Gets a dataset tuple with instructions for reading CelebA.

  Args:
    split_name: A train/test split name.
    dataset_dir: The base directory of the dataset sources.
    file_pattern: The file pattern to use when matching the dataset sources.
      It is assumed that the pattern contains a '%s' string so that the split
      name can be inserted.
    reader: The TensorFlow reader type.

  Returns:
    A `Dataset` namedtuple.

  Raises:
    ValueError: if `split_name` is not a valid train/test split.
    AssertionError: if number of classes is different from expected.
  """
  assert FLAGS.num_classes == 0 or FLAGS.num_classes == DEFAULT_NUM_CLASSES
  _SPLITS_TO_SIZES = {
    'train': FLAGS.train_size or _DEFAULT_TRAIN_SIZE,
    'validation': FLAGS.validation_size or _DEFAULT_VALIDATION_SIZE,
  }
  if split_name not in _SPLITS_TO_SIZES:
    raise ValueError('split name %s was not recognized.' % split_name)

  if not file_pattern:
    file_pattern = _FILE_PATTERN
  file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

  # Allowing None in the signature so that dataset_factory can use the default.
  if reader is None:
    reader = tf.TFRecordReader

  keys_to_features = {
    'image/encoded': tf.FixedLenFeature(
        (), tf.string, default_value=''),
    'image/format': tf.FixedLenFeature(
        (), tf.string, default_value='jpeg'),
    'image/filename': tf.FixedLenFeature(
        [], dtype=tf.string, default_value=''),

    'image/attribs': tf.FixedLenFeature([40], tf.int64, ),
    'image/landmarks': tf.FixedLenFeature([10], tf.float32, ),
  }

  output_name = 'target' if FLAGS.dataset_use_target else 'source'
  items_to_handlers = {
    output_name: slim.tfexample_decoder.Image('image/encoded', 'image/format'),
    'conditional_labels': slim.tfexample_decoder.Tensor('image/attribs'),
    'landmarks': slim.tfexample_decoder.Tensor('image/landmarks'),
    'filename':  slim.tfexample_decoder.Tensor('image/filename'),
  }

  decoder = slim.tfexample_decoder.TFExampleDecoder(
      keys_to_features, items_to_handlers)

  return slim.dataset.Dataset(
      data_sources=file_pattern,
      reader=reader,
      decoder=decoder,
      num_samples=_SPLITS_TO_SIZES[split_name],
      items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
      items_used=['conditional_labels', output_name, 'landmarks', 'filename'],
      items_need_preprocessing=['conditional_labels', output_name, ],
      num_classes=FLAGS.num_classes or DEFAULT_NUM_CLASSES,
      has_source=True)
