# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Provides data for the MNIST dataset.

The dataset scripts used to create the dataset can be found at:
tensorflow/models/research/slim/datasets/download_and_convert_mnist.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from datasets import dataset_utils

slim = tf.contrib.slim

_FILE_PATTERN = '%s*'

_SPLITS_TO_SIZES = {'train': 73257, 'test': 26032}

_IMAGE_HW = 32

_NUM_CLASSES = 10

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A [32 x 32 x 1] grayscale image.',
    'label': 'A single integer between 0 and 9',
}


def get_split(split_name, dataset_dir, file_pattern=None, reader=None):
  """Gets a dataset tuple with instructions for reading MNIST.

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
  if split_name not in _SPLITS_TO_SIZES:
    raise ValueError('split name %s was not recognized.' % split_name)

  if not file_pattern:
    file_pattern = _FILE_PATTERN
  file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

  # Allowing None in the signature so that dataset_factory can use the default.
  if reader is None:
    reader = tf.TFRecordReader

  keys_to_features = {
      'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format': tf.FixedLenFeature((), tf.string, default_value='raw'),
      # 'image/class/label': tf.FixedLenFeature(
      #     [1], tf.int64, default_value=tf.zeros([1], dtype=tf.int64)),
      # Changed to varlen to be compatible with the rest of the multi-label framework.
      'image/class/label': tf.VarLenFeature(
          tf.int64,),
  }

  num_channels = 3
  if hasattr(tf.flags.FLAGS, 'color_space') and tf.flags.FLAGS.color_space =="gray":
    num_channels = 1
  items_to_handlers = {
      'image': slim.tfexample_decoder.Image(shape=[_IMAGE_HW, _IMAGE_HW, num_channels], channels=num_channels),
      # 'label': slim.tfexample_decoder.Tensor('image/class/label', shape=[]),
      # Took off shape to be compatible with the rest of the multi-label framework.
      'label': slim.tfexample_decoder.Tensor('image/class/label',),
  }
  items_to_handlers['source'] = items_to_handlers['image']
  items_to_handlers['target'] = items_to_handlers['label']
  items_to_handlers['conditional_labels'] = items_to_handlers['label']

  decoder = slim.tfexample_decoder.TFExampleDecoder(
      keys_to_features, items_to_handlers)

  labels_to_names = None
  if dataset_utils.has_labels(dataset_dir):
    labels_to_names = dataset_utils.read_label_file(dataset_dir)

  return slim.dataset.Dataset(
      data_sources=file_pattern,
      reader=reader,
      decoder=decoder,
      num_samples=_SPLITS_TO_SIZES[split_name],
      num_classes=_NUM_CLASSES,
      items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
      labels_to_names=labels_to_names,
      items_used=['image', 'label', 'source', 'target', 'conditional_labels'],
      items_need_preprocessing=['image', 'label', 'source', 'target', 'conditional_labels'],
      has_source=True,)
