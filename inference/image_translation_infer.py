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
"""A class that loads a pretrained model and do inference on the given inputs.

Example usage:
python inference/image_translation_infer.py \
--model_path="PATH/TO/MODEL" \
--image_hw=4 \
--input_tensor_name="sources_ph" \
--output_tensor_name="custom_generated_t_style_source:0" \
--input_image_path="PATH/TO/IMAGE.JPG" \
--output_image_path="PATH/TO/IMAGE.JPG"
"""

import sys
import os
sys.path.append('./')

import numpy as np
import tensorflow as tf


import util_io


tf.flags.DEFINE_string("model_path", "", "Path containing a checkpoint.")
tf.flags.DEFINE_integer('image_hw', 256, 'height and width of the input image.')
tf.flags.DEFINE_string('input_tensor_name', None, 'Optional input tensor name. e.g. sources_ph.')
tf.flags.DEFINE_string('output_tensor_name', None, 'e.g. custom_generated_t_style_source:0')

tf.flags.mark_flags_as_required(['model_path', 'output_tensor_name'])
FLAGS = tf.flags.FLAGS

class ImageInferer(object):
  def __init__(self):
    # Load the model
    print('Loading inference model')
    session_config = tf.ConfigProto(allow_soft_placement=True, )
    self.sess = tf.Session(config=session_config)
    with self.sess.as_default():
      input_map = None
      if FLAGS.input_tensor_name:
        self.images_placeholder = tf.placeholder(dtype=tf.uint8, shape=[None, None, None])
        image = tf.image.convert_image_dtype(self.images_placeholder, dtype=tf.float32)
        image = tf.image.resize_images(image, (FLAGS.image_hw, FLAGS.image_hw))
        image = tf.expand_dims(image, axis=0)
        input_map = {FLAGS.input_tensor_name: image}

      util_io.load_model(FLAGS.model_path, input_map=input_map)

      # Get input and output tensors
      self.output = tf.get_default_graph().get_tensor_by_name(FLAGS.output_tensor_name)


  def infer(self, input_image_path, return_image_paths=False, num_output=None):
    """Given an image, a path containing images, or a list of paths, return the outputs."""
    one_output = False
    if input_image_path:
      if isinstance(input_image_path, list) or isinstance(input_image_path, tuple):
        image_paths = input_image_path
      else:
        if os.path.isfile(input_image_path):
          image_paths = [input_image_path]
          one_output = True
        else:
          image_paths = util_io.get_files_in_dir(input_image_path, do_sort=True, do_random_ordering=False)
      images = [util_io.imread(image_path, dtype=np.uint8) for image_path in image_paths]
    else:
      assert num_output >= 1
      images = [None for _ in range(num_output)]
      image_paths = [str(i) for i in range(num_output)]
      one_output = (num_output == 1)
    outputs = []
    for image in images:
      if image is None:
        feed_dict = None
      else:
        feed_dict = {self.images_placeholder: image}
      output = self.sess.run(self.output, feed_dict=feed_dict)
      output = output[0] * 255.0  # Batch size == 1, range = 0~1.
      outputs.append(output)
    if one_output:
      outputs = outputs[0]
      image_paths = image_paths[0]
    if return_image_paths:
      return outputs, image_paths
    return outputs


def main(_):
  inferer = ImageInferer()
  if FLAGS.input_image_path:
    outputs, image_paths = inferer.infer(FLAGS.input_image_path, return_image_paths=True)
  else:
    print('Generating images conditioned on random vector.')
    assert FLAGS.num_output >= 0, 'you have to specify the `num_output` flag for non-translational generators.'
    outputs, image_paths = inferer.infer(FLAGS.input_image_path, return_image_paths=True, num_output=FLAGS.num_output)

  if isinstance(outputs, list):
    util_io.touch_folder(FLAGS.output_image_path)
    for i, output in enumerate(outputs):
      util_io.imsave(os.path.join(FLAGS.output_image_path, os.path.basename(image_paths[i])), output)
  else:
    util_io.touch_folder(os.path.dirname(FLAGS.output_image_path))
    util_io.imsave(FLAGS.output_image_path, outputs)

if __name__ == '__main__':
  tf.flags.DEFINE_string('input_image_path', None,
                         'Path to the input image (or directory containing images) used for inference.'
                         'Used for image translation model e.g. TwinGAN')
  tf.flags.DEFINE_integer('num_output', None,
                          'Number of outputs. Used for generative model, e.g. PGGAN or other kinds of GANs.')
  tf.flags.DEFINE_string('output_image_path', None,
                         'Path to output the output image (or directory).')
  tf.flags.mark_flags_as_required(['output_image_path'])
  tf.app.run()