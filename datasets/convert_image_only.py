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
"""Converts folder containing images into tensorflow examples for faster data loading/preprocessing.

Example usage:

python datasets/convert_image_only.py \
--train_directory="/TRAIN/DIR/" \
--validation_directory="/VAL/DIR/" \
--output_directory="/OUTPUT/DIR/" \
--train_shards=8 \
--validation_shards=2 \
--num_threads=2
"""
import functools
import os.path

import tensorflow as tf

import util_io
import util_misc
from datasets import convert_general_image_data
from datasets import dataset_utils
from preprocessing import danbooru_preprocessing

tf.flags.DEFINE_string('exclude_affix', '',
                       'If not empty, exclude images with this affix.')
tf.flags.DEFINE_integer('preprocessing_hw', 299, 'Height and width of preprocessed image.')
tf.flags.DEFINE_string(
  'resize_mode', 'NONE', 'One of PAD, CROP, RESHAPE, or NONE as specified in preprocessing_util.py.')

FLAGS = tf.flags.FLAGS

class ImageOnlyConverter(convert_general_image_data.GeneralImageDataConverter):
  def _find_image_files(self, data_dir, shared_info):
    file_names = util_io.get_files_in_dir(data_dir, do_sort=False, do_random_ordering=True)
    if FLAGS.exclude_affix:
      file_names = [file_name for file_name in file_names if not file_name.endswith(FLAGS.exclude_affix)]
    return file_names, [None for _ in range(len(file_names))]

  def _generate_shared_info(self, train_directory, validation_directory, output_path):
    return None

  @staticmethod
  def _convert_to_example(filename, image_data, height, width, current_file_info, common_info):
    colorspace = 'RGB'
    channels = 3
    image_format = 'JPEG'

    example = tf.train.Example(features=tf.train.Features(feature={
      'image/colorspace': dataset_utils.bytes_feature(colorspace),
      'image/channels': dataset_utils.int64_feature(channels),
      'image/format': dataset_utils.bytes_feature(image_format),
      'image/filename': dataset_utils.bytes_feature(os.path.basename(filename)),
      'image/encoded': dataset_utils.bytes_feature(image_data),
    }))
    return example

  @staticmethod
  def get_preprocessing_fn():
    if FLAGS.do_preprocessing:
      return functools.partial(danbooru_preprocessing.preprocess_image,
                               output_height=FLAGS.preprocessing_hw,
                               output_width=FLAGS.preprocessing_hw,
                               resize_mode=FLAGS.resize_mode,
                               dtype=tf.uint8,
                               subtract_mean=False,
                               add_image_summaries=False, )
    else:
      return None


def main(_):
  converter = ImageOnlyConverter()
  converter.main()


if __name__ == '__main__':
  tf.app.run()
