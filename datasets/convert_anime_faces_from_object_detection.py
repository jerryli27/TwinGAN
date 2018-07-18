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
"""Given tfrecords with detected faces, extract and crop the detected faces, and save to tfrecords.

TODO: show example usage. Also write a md file that goes from anime images to face detection to this.
Example usage:

python datasets/convert_anime_faces_from_object_detection.py \
--train_directory="/TRAIN/DIR/CONTAINING/TFRECORDS/" \
--validation_directory="/VAL/DIR/CONTAINING/TFRECORDS/" \
--output_directory="/OUTPUT/DIR/" \
--train_shards=8 \
--validation_shards=2 \
--num_threads=2
"""

import os.path

import cv2
import numpy as np
import tensorflow as tf

import util_io
import util_misc
from datasets import convert_general_image_data
from datasets import dataset_utils

tf.flags.DEFINE_boolean('do_safe_only', False,
                        'If true, do not include questionable and nsfw images.')
tf.flags.DEFINE_boolean('do_unsafe_only', False,
                        'If true, only include questionable and nsfw images.')
FLAGS = tf.flags.FLAGS

# The following ratio works the best for myanimelist images. It corresponds well to the faces positions in celeba.
W_EXPANSION_RATE = 0.6
TOP_H_EXPANSION_RATE = 1.0
BOTTOM_H_EXPANSION_RATE = 0.4
DO_DEBUG = False  # Set to True to see how images are cropped.
# Helper class constants.
FACE_ONLY = True
CLASSES_TEXT = ['face', 'eye', 'mouth']
FACE_LABEL = 1
FACE_SCORE_THRESHOLD = 0.9
CLASSES_TEXT_TO_LABEL = {text: i + 1 for i, text in enumerate(CLASSES_TEXT)}
CLASSES_LABEL_TO_TEXT = {i + 1: text for i, text in enumerate(CLASSES_TEXT)}
NUM_EXAMPLES_PER_SHARD = 20000


class AnimeFaceConverter(convert_general_image_data.GeneralImageDataConverter):
  #######################
  # Shared info section.#
  #######################
  def _generate_shared_info(self, train_directory, validation_directory, output_path):
    # Read TfExamples.
    tf_records = util_io.get_files_in_dir(train_directory, do_sort=True, do_random_ordering=False,
                                          allowed_extensions=None)
    model = AnimeFaceObjectExtractor(tf_records, do_safe_only=FLAGS.do_safe_only, do_unsafe_only=FLAGS.do_unsafe_only)
    model.main()
    base_file_names = model.ret.keys()
    ret = {'object_detection_result_dict': model.ret, 'base_file_names': base_file_names}
    return ret

  ###########################
  # Process dataset section.#
  ###########################
  def _find_image_files(self, data_dir, shared_info):
    """Returns filenames, per_file_info."""
    file_names = dataset_utils.get_filenames(data_dir)
    if 'base_file_names' in shared_info:
      base_file_names = shared_info['base_file_names']
      file_names = [file_name for file_name in file_names if util_misc.get_no_ext_base(file_name) in base_file_names]
    print('got %d files from %s' % (len(file_names), data_dir))
    return file_names, [None for _ in range(len(file_names))]

  def _process_and_convert(self, filename, current_file_info, shared_info, coder):
    """Wrapper around the process_image(), check_hw(), and _convert_to_example() pipeline."""
    processed = self.process_image(filename, self._real_coder, shared_info)
    if processed is None:
      return None
    examples = []
    for item in processed:
      image_data, img_info = item
      example = self._convert_to_example(filename, image_data, img_info[1], img_info[0], img_info, shared_info)
      examples.append(example)
    return examples

  def process_image(self, filename, coder, shared_info):
    """Process a single image file.

    Args:
      filename: string, path to an image file e.g., '/path/to/example.JPG'.
      coder: instance of ImageCoder to provide TensorFlow image coding utils.
    Returns:
      image_data: [height, width, channels] array of RGB image.
      height: integer, image height in pixels.
      width: integer, image width in pixels.
      image: [height, width, channels] array of RGB image.
    """
    object_detection_result_dict = shared_info['object_detection_result_dict']
    object_detection_result = object_detection_result_dict.get(os.path.basename(filename), None)
    if not object_detection_result:
      return None

    # Read the image file.
    image_data = tf.gfile.FastGFile(filename, 'r').read()

    # Convert all non-jpg data to jpg
    if not dataset_utils.is_jpeg(filename):
      image_data = coder.image_to_jpeg(image_data)

    # Decode the RGB JPEG.
    image = coder.decode_jpeg(image_data)

    # Check image is correctly converted to RGB.
    assert len(image.shape) == 3
    height = image.shape[0]
    width = image.shape[1]
    assert image.shape[2] == 3
    assert np.max(image) <= 255.0 and np.min(image) >= 0.0

    # Get face bounding box.
    faces = util_misc.get_faces(object_detection_result, width, height)
    if len(faces) <= 0:
      return None

    ret = []
    for i in range(len(faces)):
      (x, y, w, h) = faces[i]
      # Expand the face bounding box to include the whole head.
      x_expanded, y_expanded, w_expanded, h_expanded = util_misc.unevenly_expand_xywh(
        x, y, w, h, width, height, W_EXPANSION_RATE, W_EXPANSION_RATE, TOP_H_EXPANSION_RATE, BOTTOM_H_EXPANSION_RATE)
      try:
        # Filter out images too small or too large, or has weird height width ratio.
        self.check_hw(h_expanded, w_expanded)
      except convert_general_image_data.InvalidHeightWidth:
        continue
      crop_img = image[y_expanded: y_expanded + h_expanded, x_expanded:x_expanded + w_expanded]

      if DO_DEBUG:
        # SHOW IMAGE FOR DEBUGGING:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(image, (x_expanded, y_expanded), (x_expanded + w_expanded, y_expanded + h_expanded), (0, 255, 0),
                      2)
        cv2.imshow("AnimeFaceDetect", image)
        cv2.waitKey(0)

      try:
        encoded_crop_img = coder.array_to_jpeg(crop_img)
      except tf.errors.InternalError:
        tf.logging.info('Got error while doing `array_to_jpeg` for image %s', filename)
        return None
      ret.append(
        (encoded_crop_img, (x_expanded, y_expanded, w_expanded, h_expanded, width, height, None, image_data, faces[i])))
    return ret

  @staticmethod
  def _convert_to_example(filename, image_data, height, width, current_file_info, shared_info):
    colorspace = 'RGB'
    channels = 3
    image_format = 'JPEG'
    (x_expanded, y_expanded, w_expanded, h_expanded, image_w, image_h, tags_id, original_image,
     face_xywh) = current_file_info

    feature = {
      'image/x': dataset_utils.int64_feature(x_expanded),
      'image/y': dataset_utils.int64_feature(y_expanded),
      'image/height': dataset_utils.int64_feature(h_expanded),
      'image/width': dataset_utils.int64_feature(w_expanded),

      'image/face_xywh': dataset_utils.float_feature(face_xywh),
      # 'image/left_eye_xywh': dataset_utils.float_feature(left_eye_xywh),
      # 'image/right_eye_xywh': dataset_utils.float_feature(right_eye_xywh),
      # 'image/mouth_xywh': dataset_utils.float_feature(mouth_xywh),

      'image/colorspace': dataset_utils.bytes_feature(colorspace),
      'image/channels': dataset_utils.int64_feature(channels),
      'image/format': dataset_utils.bytes_feature(image_format),
      'image/filename': dataset_utils.bytes_feature(os.path.basename(filename)),
      'image/encoded': dataset_utils.bytes_feature(image_data),
      # Encoding original takes up too much space. Not recommended.
      # 'image/original': dataset_utils.bytes_feature(original_image),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example

  ###########################
  # Other utility functions #
  ###########################
  # Inherits from parent class.

  ########
  # Main #
  ########
  # Inherits from parent class.


################
# Helper class #
################
class AnimeFaceObjectExtractor(object):
  def __init__(self, filenames, do_safe_only=False, do_unsafe_only=False):
    """Extract features from the tf example.

    Args:
      filenames: a list of string, paths to tf records.
      do_safe_only: only output images starting with 's'.
      do_safe_only: only output images NOT starting with 's'.
    """
    assert not (do_safe_only and do_unsafe_only)
    self._do_safe_only = do_safe_only
    self._do_unsafe_only = do_unsafe_only
    self._filenames = filenames
    self._filename_parser = dataset_utils.StringParser('image/filename')
    self._label_parser = dataset_utils.Int64Parser('image/detection/label')
    self._ymin_parser = dataset_utils.FloatParser('image/detection/bbox/ymin')
    self._xmin_parser = dataset_utils.FloatParser('image/detection/bbox/xmin')
    self._ymax_parser = dataset_utils.FloatParser('image/detection/bbox/ymax')
    self._xmax_parser = dataset_utils.FloatParser('image/detection/bbox/xmax')
    self._score_parser = dataset_utils.FloatParser('image/detection/score')

  def main(self):
    """Go through the tf records in self._filenames and save faces in self.ret dictionary with key = filename."""
    self.ret = {}
    num_examples = 0
    for tf_example in dataset_utils.iterate_tfrecords(self._filenames):
      num_examples += 1
      if num_examples % 100 == 0 and num_examples != 0:
        print('Going over %d items.' % (num_examples))
      filename = self._filename_parser.parse(tf_example)
      # Assume the dataset is danbooru...
      if self._do_safe_only and not filename.startswith('s'):
        continue
      if self._do_unsafe_only and filename.startswith('s'):
        continue

      label = self._label_parser.parse(tf_example)
      ymin = self._ymin_parser.parse(tf_example)
      xmin = self._xmin_parser.parse(tf_example)
      ymax = self._ymax_parser.parse(tf_example)
      xmax = self._xmax_parser.parse(tf_example)
      score = self._score_parser.parse(tf_example)
      faces = []
      if not label:
        continue
      # For each face with score above threshold, find the corresponding eyes and mouths with highest score.
      # Or for now just output the faces.
      for label_i in range(label.shape[0]):
        if label[label_i] == FACE_LABEL and score[label_i] >= FACE_SCORE_THRESHOLD:
          faces.append((label[label_i], ymin[label_i], xmin[label_i], ymax[label_i], xmax[label_i], score[label_i]))

      self.ret[filename] = {
        'faces': faces,
      }

    print('Done! Went over %d tf examples.' % num_examples)


def main(_):
  converter = AnimeFaceConverter()
  converter.main()


if __name__ == '__main__':
  tf.app.run()
