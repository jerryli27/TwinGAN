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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import xml.etree.ElementTree

import tensorflow as tf

GENERAL_TYPE = 0
AUTHOR_TYPE = 1
COPYRIGHT_TYPE = None
CHARACTER_TYPE = None
META_TYPE = 5
TAG_TEXT_DELIMITER = ', '


def parse_file_name(file_name):
  """Given a file name, parse its id and the list of tags out.

  Args:
    file_name: a string of a file name with format:
      data_dir/<nsfw_rating> - id.jpg

  Returns:
    nsfw_rating, id, tags
  """
  basename, ext = os.path.splitext(os.path.basename(file_name))
  split_result = basename.split(' - ')
  if len(split_result) != 2:
    raise AssertionError('File name has illegal format. Got %s', file_name)

  nsfw_rating, id = split_result
  tags_file_name = file_name + '.txt'
  tags = tf.gfile.FastGFile(tags_file_name, 'r').readlines()
  tags = [tag.rstrip('\n') for tag in tags]
  tags.append('NSFWRating_' + nsfw_rating)
  return nsfw_rating, id, tags


def parse_tags_xml(tags_file):
  print('Determining list of input files and labels from %s.' % tags_file)
  tags_tree = xml.etree.ElementTree.parse(tags_file).getroot()
  tags = []
  for i, nsfw_rating in enumerate(('s', 'q', 'e')):
    # Insert the nsfw ratings first. Assign count as 0 and id as a large number.
    tags.append((GENERAL_TYPE, 'false', 0, 'NSFWRating_' + nsfw_rating, 1e10 + i))
  for tag in tags_tree.findall('tag'):
    type = int(tag.get('type'))
    if type != AUTHOR_TYPE and type != META_TYPE:
      tags.append((type, tag.get('ambiguous'), int(tag.get('count')), tag.get('name'), int(tag.get('id'))))
  tag_name_to_index_dict = {tag[3]: i for i, tag in enumerate(tags)}
  return tags, tag_name_to_index_dict


def get_raw_labels(filenames):
  labels_per_file = []
  i = 0
  while i < len(filenames):
    file_name = filenames[i]
    try:
      labels_per_file.append(parse_file_name(file_name)[2])
    except Exception as e:
      print('Got error %s when processing file %s. Skipping that file.' % (e, file_name))
      del filenames[i]
    else:
      i += 1
  return labels_per_file
