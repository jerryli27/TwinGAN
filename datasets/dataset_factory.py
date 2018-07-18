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
"""A factory-pattern class which returns a dataset. Modified from the slim image classification model library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from datasets import anime_faces
from datasets import celeba_facenet
from datasets import image_only
from datasets import image_pair
from datasets import danbooru_2_illust2vec
from datasets import svhn
from datasets import celeba

# The flags here are shared among datasets. Each dataset will definne its own default values.
tf.flags.DEFINE_integer('num_classes', 0, 'number of classses in this dataset.')
tf.flags.DEFINE_integer('embedding_size', 0, 'embedding size of this dataset.')
# Data size is used for decreasing learning rate and for evaluating once through the dataset.
tf.flags.DEFINE_integer('train_size', 0, 'embedding size of this dataset.')
tf.flags.DEFINE_integer('validation_size', 0, 'embedding size of this dataset.')
tf.flags.DEFINE_boolean('dataset_use_target', False,
                         'If set, outputs images to "target". Otherwise outputs to "source". '
                         'Check each dataset to see if this flag is used.')

tf.flags.DEFINE_string('tags_id_lookup_file', './datasets/illust2vec_tag_list.txt',
                       'Optional path to the tags to be processed by tensorflow.contrib.lookup.index_table_from_file'
                       'e.g. for illust2vec (./datasets/illust2vec_tag_list.txt) the line format is: '
                       'original_illust2vec_id, tag, group. ')
tf.flags.DEFINE_integer('tags_key_column_index', None, 'See tensorflow.contrib.lookup.index_table_from_file.')
tf.flags.DEFINE_integer('tags_value_column_index', None, 'See tensorflow.contrib.lookup.index_table_from_file.')

FLAGS = tf.flags.FLAGS

datasets_map = {
  'anime_faces': anime_faces,
  'celeba': celeba,
  'celeba_facenet': celeba_facenet,
  'danbooru_2_illust2vec': danbooru_2_illust2vec,
  'image_only': image_only,
  'image_pair': image_pair,
  'svhn': svhn,
}


def get_dataset(name, split_name, dataset_dir, file_pattern=None, reader=None):
  """Given a dataset name and a split_name returns a Dataset.

  Args:
    name: String, the name of the dataset.
    split_name: A train/test split name.
    dataset_dir: The directory where the dataset files are stored.
    file_pattern: The file pattern to use for matching the dataset source files.
    reader: The subclass of tf.ReaderBase. If left as `None`, then the default
      reader defined by each dataset is used.

  Returns:
    A `Dataset` class.

  Raises:
    ValueError: If the dataset `name` is unknown.
  """
  if name not in datasets_map:
    raise ValueError('Name of dataset unknown %s' % name)
  dataset = datasets_map[name].get_split(
      split_name,
      dataset_dir,
      file_pattern,
      reader)
  dataset.name = name
  if FLAGS.train_size and split_name == 'train':
    dataset.num_samples = FLAGS.train_size
  else:
    if FLAGS.validation_size:
      dataset.num_samples = FLAGS.validation_size
  return dataset

