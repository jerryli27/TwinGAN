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
"""Converts Danbooru data to TFRecords of TF-Example protos.

This module reads the files that make up the Danbooru data and creates
two TFRecord datasets: one for train and one for test. Each TFRecord
dataset is comprised of a set of TF-Example protocol buffers, each of
which contain a single image and label.

It is intended for the danbooru downloader. https://github.com/Nandaka/DanbooruDownloader. It assumes each image has
a txt file with the same name with all the tags of that image.
There's also https://www.gwern.net/Danbooru2017 which is easier to use but you will need to modify this script to
use that dataset.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import csv
import os
from collections import Counter

import tensorflow as tf

from datasets import dataset_utils
from datasets import danbooru_utils
from datasets import convert_general_image_data
from preprocessing import danbooru_preprocessing


tf.flags.DEFINE_string('tags_file', '/tmp/tags.xml',
                           'The tags xml file name.')
tf.flags.DEFINE_integer('max_num_labels', 10000, 'Maximum number of most common tags to keep.')
tf.flags.DEFINE_integer('preprocessing_hw', 299, 'Height and width of preprocessed image.')

FLAGS = tf.flags.FLAGS

TAG_TEXT_DELIMITER = ', '
# NSFW_TO_LABEL = {'s': 'safe', 'e': 'explicit', 'q': 'questionable'}

class DanbooruDataConverter(convert_general_image_data.GeneralImageDataConverter):
  @staticmethod
  def _tag_to_human_readable(tag_indices, tags):
    return TAG_TEXT_DELIMITER.join(tags[tag_i-1][-2] for tag_i in tag_indices)

  @staticmethod
  def _convert_to_example(filename, image_buffer, height, width, current_file_info, common_info):
    """Build an Example proto for an example.

    Args:
      filename: string, path to an image file, e.g., '/path/to/example.JPG'
      image_buffer: string, JPEG encoding of RGB image
      height: integer, image height in pixels
      width: integer, image width in pixels
      current_file_info:  equivalent to label: integer, identifier for the ground truth for the network
      common_info: a list of tags with format: ('type', 'ambiguous', 'count', 'name', 'id')

    Returns:
      Example proto
    """
    colorspace = 'RGB'
    channels = 3
    image_format = 'JPEG'
    human_readable_tags = DanbooruDataConverter._tag_to_human_readable(current_file_info,common_info)

    example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_utils.int64_feature(height),
      'image/width': dataset_utils.int64_feature(width),
      'image/colorspace': dataset_utils.bytes_feature(colorspace),
      'image/channels': dataset_utils.int64_feature(channels),
      'image/class/label': dataset_utils.int64_feature(current_file_info),
      'image/class/text': dataset_utils.bytes_feature(human_readable_tags),
      'image/format': dataset_utils.bytes_feature(image_format),
      'image/filename': dataset_utils.bytes_feature(os.path.basename(filename)),
      'image/encoded': dataset_utils.bytes_feature(image_buffer)}))
    return example

  def _process_tags(self, filenames, tags, tag_name_to_index_dict, max_num_labels):
    # Get labels.
    raw_labels = danbooru_utils.get_raw_labels(filenames)
    labels_count = Counter(
      [item for sublist in raw_labels for item in sublist])  # Flattens labels and count the elements.
    labels_most_common = []
    for i, nsfw_rating in enumerate(('s', 'q', 'e')):
      # Insert the nsfw ratings first.
      nsfw_name = 'NSFWRating_'+nsfw_rating
      labels_most_common.append((nsfw_name, labels_count.get(nsfw_name, 0)))
    labels_most_common += labels_count.most_common()

    # Get the corresponding most common tags in the input dataset.
    tags_most_common = []
    tag_name_to_id = {}
    # Leave label index 0 empty as a background class.
    label_index = 1
    for label, count in labels_most_common:
      tag_index = tag_name_to_index_dict.get(label, None)
      # Excluding authors.
      if tag_index is not None:
        tags_most_common.append((tags[tag_index][:2] + (count, label, label_index)))
        tag_name_to_id[label] = label_index
        label_index += 1
      if label_index >= max_num_labels:
        break

    return tags_most_common

  def get_labels_per_file(self, filenames, shared_info):
    labels_per_file = danbooru_utils.get_raw_labels(filenames)
    tag_name_to_id = {}
    for row in shared_info:
      assert len(row) == 5
      label_index = int(row[-1])
      label = row[-2]
      tag_name_to_id[label] = label_index

    # Filter uncommon tags.
    for i in range(len(labels_per_file)):
      labels_per_file[i] = [tag_name_to_id[label] for label in labels_per_file[i] if label in tag_name_to_id]
    return labels_per_file


  def _read_shared_info(self, path):
    with open(path, 'r') as out:
      reader = csv.reader(out)
      tags_most_common = []
      for row in reader:
        tags_most_common.append(row)
    return tags_most_common

  def _generate_shared_info(self, train_directory, validation_directory, output_path):

    tags_file = FLAGS.tags_file
    max_num_labels = FLAGS.max_num_labels

    tags, tag_name_to_index_dict = danbooru_utils.parse_tags_xml(tags_file)
    print('Finished parsing %d tags.' %(len(tags)))

    # Construct the list of image files. Excludes txt files.
    filenames = dataset_utils.get_filenames(train_directory) + dataset_utils.get_filenames(validation_directory)

    tags_most_common = self._process_tags(filenames, tags, tag_name_to_index_dict, max_num_labels)
    tags_output_file_name = output_path
    with open(tags_output_file_name, 'w') as out:
      writer = csv.writer(out)
      writer.writerows(tags_most_common)
    return tags_most_common

  def _find_image_files(self, data_dir, shared_info):
    """Build a list of all images files and labels in the data set.

    Args:
      data_dir: string, path to the root directory of images.

        Assumes that the Danbooru data set resides in JPEG files located in
        the following directory structure.

          data_dir/<safe_letter> - id - <tags separated by space>.jpg

      tags_file: string, path to the labels file.

        The list of valid labels are held in this file. Assumes that the file
        contains entries as such:
          <tag type="0" ambiguous="false" count="10029" name="^_^" id="402217" />
        where ... explain what type, ambiguous is.

        The reason we start the integer labels at 1 is to reserve label 0 as an
        unused background class.
      max_num_labels: only keep the top X labels with the most count.

    Returns:
      filenames: list of strings; each string is a path to an image file.
      synsets: list of strings; each string is a unique WordNet ID.
      labels: list of integer; each integer identifies the ground truth.
    """
    # Construct the list of image files. Excludes txt files.
    filenames = dataset_utils.get_filenames(data_dir)
    labels_per_file = self.get_labels_per_file(filenames, shared_info)
    print('Found %d image files across %d tags inside %s.' %
          (len(filenames), len(shared_info), data_dir))
    return filenames, labels_per_file,

  @staticmethod
  def get_preprocessing_fn():
    if FLAGS.do_preprocessing:
      return functools.partial(danbooru_preprocessing.preprocess_image,
                               output_height=FLAGS.preprocessing_hw,
                               output_width=FLAGS.preprocessing_hw,
                               dtype=tf.uint8,
                               subtract_mean=False,
                               add_image_summaries=False,)
    else:
      return None

  def main(self,):
    assert FLAGS.shared_info_output
    super(DanbooruDataConverter, self).main()

def main(_):
  converter = DanbooruDataConverter()
  converter.main()

if __name__ == '__main__':
  tf.app.run()