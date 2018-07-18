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
r"""This module provides an extendable base class for converting general image data into tfrecord format.

See convert_image_only.py for a child class of `GeneralImageDataConverter`.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import threading
from datetime import datetime

import numpy as np
import tensorflow as tf

import util_io
import util_misc
from datasets import dataset_utils


tf.flags.DEFINE_string('train_directory', '/tmp/',
                       'Training data directory')
tf.flags.DEFINE_string('validation_directory', '/tmp/',
                       'Validation data directory')
tf.flags.DEFINE_string('output_directory', None,
                       'Output data directory')
tf.flags.DEFINE_string('shared_info_output', '',
                       'The output of the information shared among training and validation.')

tf.flags.DEFINE_integer('train_shards', 8,
                        'Number of shards in training TFRecord files.')
tf.flags.DEFINE_integer('validation_shards', 1,
                        'Number of shards in validation TFRecord files.')

tf.flags.DEFINE_integer('num_threads', 1,
                        'Number of threads to preprocess the images.')

tf.flags.DEFINE_float('allowed_hw_ratio', 0.0,
                      'The maximum allowed height:width ratio or width:height ratio, whichever is greater.'
                      'set to anything smaller than 1.0 to disable filtering images with wierd h:w ratios.')

tf.flags.DEFINE_integer('allowed_max_hw', 0,
                        'The maximum allowed height or width, whichever is greater. Set to 0 to disable.')
tf.flags.DEFINE_integer('allowed_min_hw', 0,
                        'The minimum allowed height or width, whichever is smaller. Set to 0 to disable.')

tf.flags.DEFINE_boolean('do_preprocessing', False, 'If true, all images goes through preprocessing.')
# tf.flags.mark_flags_as_required(['output_directory'])
FLAGS = tf.flags.FLAGS


class InvalidHeightWidth(BaseException):
  def __init__(self, *args):
    super(InvalidHeightWidth, self).__init__(*args)


class GeneralImageDataConverter(object):
  def __init__(self):
    """Converts general images into tfrecord format. Intended to be inherited for more specific uses."""
    self.coder = self.get_coder()
    self.session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    self.coder.set_session(session=self.session)

  ##############################
  # Functions to be overridden #
  ##############################
  def _find_image_files(self, data_dir, shared_info):
    """Returns filenames, per_file_info."""
    raise NotImplementedError('To be implemented by the child class.')

  @staticmethod
  def _convert_to_example(filename, image_buffer, height, width, current_file_info, common_info):
    raise NotImplementedError('To be implemented by the child class.')

  def _read_shared_info(self, path):
    """A general function to read any data shared by all processed data items. Useful if the data is precalculated."""
    raise NotImplementedError('To be optionally overwritten by a child class.')

  def _generate_shared_info(self, train_directory, validation_directory, output_path):
    """A general function to generate the data shared by all processed data items across train and validation set."""
    raise NotImplementedError('To be overwritten by a child class. Simply return None if not applicable to your data.')

  #######################
  # Shared info section.#
  #######################
  def _get_shared_info(self):
    """Wrapper around _read_shared_info() and _generate_shared_info()."""
    # If shared info already exists, read it instead of generating it again.
    if os.path.isfile(FLAGS.shared_info_output):
      return self._read_shared_info(FLAGS.shared_info_output)
    else:
      shared_info_output_dir = os.path.split(FLAGS.shared_info_output)[0]
      if not os.path.exists(shared_info_output_dir):
        util_io.touch_folder(shared_info_output_dir)
      return self._generate_shared_info(FLAGS.train_directory, FLAGS.validation_directory, FLAGS.shared_info_output)

  ###########################
  # Process dataset section.#
  ###########################
  def _process_dataset(self, name, directory, num_shards, shared_info, ):
    """Process a data set and save it as a TFRecord.

    Args:
      name: string, unique identifier specifying the data set. e.g. 'train'
      directory: string, root path to the data set.
      num_shards: integer number of shards for this data set.
      shared_info: any, optional general information that may be necessary for processing.
    """
    filenames, per_file_info = self._find_image_files(directory, shared_info)
    self._process_image_files(name, num_shards, filenames, per_file_info, shared_info)

  def _process_image_files(self, name, num_shards, filenames, per_file_info, shared_info, ):
    """Process and save list of images as TFRecord of Example protos.

    Args:
      name: string, unique identifier specifying the data set
      num_shards: integer number of shards for this data set.
      filenames: list of strings; each string is a path to an image file
      per_file_info: related information for each filename.
      shared_info: information shared by all files.
    """
    assert len(filenames) == len(per_file_info), 'Number of file names and len(per_file_info) does not match.'

    # Break all images into batches with a [ranges[i][0], ranges[i][1]].
    spacing = np.linspace(0, len(filenames), FLAGS.num_threads + 1).astype(np.int)
    ranges = []
    for i in xrange(len(spacing) - 1):
      ranges.append([spacing[i], spacing[i + 1]])

    # Launch a thread for each batch.
    print('Launching %d threads for spacings: %s' % (FLAGS.num_threads, ranges))
    sys.stdout.flush()

    # Create a mechanism for monitoring when all threads are finished.
    coord = tf.train.Coordinator()

    threads = []
    for thread_index in xrange(len(ranges)):
      args = (self.coder, thread_index, ranges, name, num_shards, filenames, per_file_info, shared_info)
      t = threading.Thread(target=self.process_image_files_batch, args=args)
      t.start()
      threads.append(t)

    # Wait for all the threads to terminate.
    coord.join(threads)
    print('%s: Finished processing all %d images in data set.' %
          (datetime.now(), len(filenames)))
    sys.stdout.flush()

  def _process_and_convert(self, filename, current_file_info, shared_info, coder):
    """Wrapper around the process_image(), check_hw(), and _convert_to_example() pipeline."""
    image_buffer, height, width = self.process_image(filename, coder, FLAGS.do_preprocessing)
    self.check_hw(height, width)
    example = self._convert_to_example(filename, image_buffer, height, width, current_file_info, shared_info)
    return example

  def process_image(self, filename, coder, do_preprocess):
    """Process a single image file. Wraps around dataset_utils.process_image()."""
    return dataset_utils.process_image(filename, coder, do_preprocess, return_image_np=False)

  def process_image_files_batch(self, coder, thread_index, ranges, name, num_shards, filenames, per_file_info,
                                shared_info):
    """Processes and saves list of images as TFRecord in 1 thread.

    Args:
      coder: instance of ImageCoder to provide TensorFlow image coding utils.
      thread_index: integer, unique batch to run index is within [0, len(ranges)).
      ranges: list of pairs of integers specifying ranges of each batches to
        analyze in parallel.
      name: string, unique identifier specifying the data set
      num_shards: integer number of shards for this data set.
      filenames: list of strings; each string is a path to an image file
      per_file_info: list of tuples/dictionaries. Contains information corresponding to each filename.
      shared_info: information shared by all files. Generated by _generate_shared_info().
    """
    # Each thread produces N shards where N = int(num_shards / num_threads).
    # For instance, if num_shards = 128, and the num_threads = 2, then the first
    # thread would produce shards [0, 64).
    num_threads = len(ranges)
    assert not num_shards % num_threads, ('Please make the FLAGS.num_threads mod # output shards == 0')
    num_shards_per_batch = int(num_shards / num_threads)

    shard_ranges = np.linspace(ranges[thread_index][0],
                               ranges[thread_index][1],
                               num_shards_per_batch + 1).astype(int)
    num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

    counter = 0
    num_success = 0
    num_corrupted = 0
    num_invalid_hw = 0
    for s in xrange(num_shards_per_batch):
      # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
      shard = thread_index * num_shards_per_batch + s
      output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)
      output_file = os.path.join(FLAGS.output_directory, output_filename)
      writer = tf.python_io.TFRecordWriter(output_file)

      shard_counter = 0
      files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
      for i in files_in_shard:
        filename = filenames[i]
        current_file_info = per_file_info[i]

        try:
          example = self._process_and_convert(filename, current_file_info, shared_info, coder)
        except (tf.errors.InvalidArgumentError, AssertionError) as e:
          util_misc.safe_print('Image process error for file {0} : error is {1}', filename, str(e))
          num_corrupted += 1
        except InvalidHeightWidth:
          num_invalid_hw += 1
        except Exception as e:
          num_corrupted += 1
          util_misc.safe_print('Other image process error for file {0} : error is {1}', filename, str(e))
        else:
          if example is not None:
            if isinstance(example, list):
              for single_example in example:
                writer.write(single_example.SerializeToString())
            else:
              assert isinstance(example, tf.train.Example)
              writer.write(example.SerializeToString())
            num_success += 1
        shard_counter += 1
        counter += 1

        if not counter % 1000:
          print('%s [thread %d]: Processed %d of %d images in thread batch.' %
                (datetime.now(), thread_index, counter, num_files_in_thread))
          sys.stdout.flush()

      writer.close()
      print('%s [thread %d]: Wrote %d images to %s' %
            (datetime.now(), thread_index, num_success, output_file))
      sys.stdout.flush()
    print('%s [thread %d]: Wrote %d images to %d shards. %d images are corrupted. %d images have invalid hw ratio.' %
          (datetime.now(), thread_index, num_success, num_shards, num_corrupted, num_invalid_hw))
    sys.stdout.flush()

  ###########################
  # Other utility functions #
  ###########################
  @staticmethod
  def get_preprocessing_fn():
    """No preprocessing functions by default."""
    return None

  def get_coder(self):
    """Gets an image encoder with the specified preprocessing function."""
    return dataset_utils.ImageCoder(preprocess_fn=self.get_preprocessing_fn())

  @staticmethod
  def _convert_to_example_util(feature_name_to_feature_and_type):
    """Build an Example proto for an example.

    Args:
      feature_name_to_feature_and_type: a dict. key = feature name[str], val = (feature[any], type[tf.train.Feature]).

    Returns:
      Example proto
    """
    example = tf.train.Example(features=tf.train.Features(feature={
      name: feature_type(feature) for name, (feature, feature_type) in feature_name_to_feature_and_type.iteritems()
    }))
    return example

  @staticmethod
  def check_hw(height, width):
    """Given the height and width, raise an error if they do not follow the size/ratios specified in the flags."""
    assert height and width, 'height and width must not be empty.'
    ratio = dataset_utils.get_height_width_ratio(height, width)
    if FLAGS.allowed_hw_ratio >= 1 and (ratio > FLAGS.allowed_hw_ratio or (1.0 / ratio) > FLAGS.allowed_hw_ratio):
      raise InvalidHeightWidth
    if FLAGS.allowed_max_hw > 0 and max(height, width) > FLAGS.allowed_max_hw:
      raise InvalidHeightWidth
    if FLAGS.allowed_min_hw > 0 and min(height, width) < FLAGS.allowed_min_hw:
      raise InvalidHeightWidth

  ########
  # Main #
  ########
  def main(self, ):
    tf.logging.set_verbosity(tf.logging.INFO)
    assert not FLAGS.train_shards % FLAGS.num_threads, (
      'Please make the FLAGS.num_threads mod FLAGS.train_shards == 0')
    assert not FLAGS.validation_shards % FLAGS.num_threads, (
      'Please make the FLAGS.num_threads mod FLAGS.validation_shards == 0')
    print('Saving results to %s' % FLAGS.output_directory)

    util_io.touch_folder(FLAGS.output_directory)
    shared_info = self._get_shared_info()

    if FLAGS.validation_directory:
      self._process_dataset('validation', FLAGS.validation_directory, FLAGS.validation_shards, shared_info, )
    if FLAGS.train_directory:
      self._process_dataset('train', FLAGS.train_directory, FLAGS.train_shards, shared_info, )


def main(_):
  raise NotImplementedError('This is a file containing a general parent class.')


if __name__ == '__main__':
  tf.app.run()
