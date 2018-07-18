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
r"""CelebA dataset formating.

Adapted from https://github.com/tensorflow/models/blob/master/research/real_nvp/celeba_formatting.py .

Download img_align_celeba.zip from
http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html under the
link "Align&Cropped Images" in the "Img" directory and list_eval_partition.txt
under the link "Train/Val/Test Partitions" in the "Eval" directory. Then do:
unzip img_align_celeba.zip

Use the script as follow:
python celeba_formatting.py \
    --partition_fn [PARTITION_FILE_PATH] \
    --output_directory [OUTPUT_FILE_PATH_PREFIX] \
    --fn_root [CELEBA_FOLDER]

"""

import os
import os.path

import tensorflow as tf

import util_io
from datasets import convert_general_image_data
from datasets import dataset_utils

tf.flags.DEFINE_string("fn_root", "", "Name of root file path.")
tf.flags.DEFINE_string("partition_fn", "", "Partition file path.")

FLAGS = tf.flags.FLAGS


class CelebAConverter(convert_general_image_data.GeneralImageDataConverter):

  def _find_image_files(self, _, shared_info):
    with open(FLAGS.partition_fn, "r") as infile:
      file_names = infile.readlines()
    file_names = [elem.strip().split() for elem in file_names]
    file_names = [os.path.join(FLAGS.fn_root, elem[0]) for elem in file_names if elem[1] == shared_info]

    return file_names, [None for _ in range(len(file_names))]

  def _generate_shared_info(self, train_directory, validation_directory, output_path):
    pass

  @staticmethod
  def _convert_to_example(filename, image_data, height, width, current_file_info, common_info):
    colorspace = 'RGB'
    channels = 3
    image_format = 'JPEG'

    example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_utils.int64_feature(height),
      'image/width': dataset_utils.int64_feature(width),
      'image/colorspace': dataset_utils.bytes_feature(colorspace),
      'image/channels': dataset_utils.int64_feature(channels),
      'image/format': dataset_utils.bytes_feature(image_format),
      'image/filename': dataset_utils.bytes_feature(os.path.basename(filename)),
      'image/encoded': dataset_utils.bytes_feature(image_data)}))
    return example

  def main(self, ):
    assert not FLAGS.train_shards % FLAGS.num_threads, (
      'Please make the FLAGS.num_threads commensurate with FLAGS.train_shards')
    assert not FLAGS.validation_shards % FLAGS.num_threads, (
      'Please make the FLAGS.num_threads commensurate with '
      'FLAGS.validation_shards')
    print('Saving results to %s' % FLAGS.output_directory)

    util_io.touch_folder(FLAGS.output_directory)

    # Run it!
    self._process_dataset('validation', None, FLAGS.validation_shards, '2')
    self._process_dataset('test', None, FLAGS.validation_shards, '1')
    self._process_dataset('train', None, FLAGS.train_shards, '0')


def main(_):
  converter = CelebAConverter()
  converter.main()


if __name__ == "__main__":
  tf.app.run()
