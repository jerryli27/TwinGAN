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
"""Danbooru dataset for training illust2vec. See https://github.com/rezoo/illustration2vec


"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
import tensorflow.contrib.slim as slim

from datasets import dataset_utils

_FILE_PATTERN = '%s-*'

_ITEMS_TO_DESCRIPTIONS = {
    'source': 'A color image of varying height and width.',
    'target': 'integer. The label id of the image.',
    'label_text': 'The text of the label.',
}


FLAGS = tf.flags.FLAGS
DEFAULT_NUM_CLASSES = 1539
_DEFAULT_TRAIN_SIZE = 1157336
_DEFAULT_VALIDATION_SIZE = 197590


def get_split(split_name, dataset_dir, file_pattern=None, reader=None):
  """Gets a dataset tuple with instructions for reading ImageNet.

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
  """
  assert FLAGS.num_classes == 0 or FLAGS.num_classes == DEFAULT_NUM_CLASSES
  num_classes = FLAGS.num_classes or DEFAULT_NUM_CLASSES
  # assert FLAGS.color_space == 'bgr' and FLAGS.subtract_mean == True

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
      'image/class/label': tf.VarLenFeature(
          dtype=tf.int64),
      'image/class/text': tf.FixedLenFeature(
          [], dtype=tf.string, default_value=''),
  }

  items_to_handlers = {
      'source': slim.tfexample_decoder.Image('image/encoded', 'image/format'),
      'target': dataset_utils.OneHotLabelTensor('image/class/text',
                                                tags_id_lookup_file=FLAGS.tags_id_lookup_file,
                                                num_classes=num_classes,
                                                tags_key_column_index=FLAGS.tags_key_column_index,
                                                tags_value_column_index=FLAGS.tags_value_column_index),
      'label_text': slim.tfexample_decoder.Tensor('image/class/text'),
  }

  decoder = slim.tfexample_decoder.TFExampleDecoder(
      keys_to_features, items_to_handlers)

  return slim.dataset.Dataset(
      data_sources=file_pattern,
      reader=reader,
      decoder=decoder,
      num_samples=_SPLITS_TO_SIZES[split_name],
      items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
      items_used=['source', 'target'],
      num_classes=num_classes,
      has_source=True)