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
"""This is the helper script to easily train PGGAN. It handles all the stage switches automatically.

Example usage:
python pggan_runner.py
--program_name=image_generation
--dataset_name="celeba"
--dataset_dir="./data/celeba/"
--train_dir="./checkpoints/pggan_celeba/"
--dataset_split_name=train
--preprocessing_name="danbooru"
--learning_rate=0.0001
--learning_rate_decay_type=fixed
--is_training=True

See /docs/training.md for more details.
"""

import ast
import math
import os
import time

import tensorflow as tf

import image_generation
import twingan

tf.flags.DEFINE_string('program_name', 'image_generation',
                       'image_generation, twingan, or other supported program name in `select_program()`.')
tf.flags.DEFINE_integer('num_images_per_resolution', 300000,
                        'The number applies to growth and no growth stage separately.'
                        'That is to say, if batch size is 10, and this flag is 100, then it will run for 10 steps for'
                        '4x4, 4x4 growing to 8x8, 8x8, 8x8 growing to 16x16, etc...')
tf.flags.DEFINE_integer('start_hw', 4,
                        'Starting height and width.')
tf.flags.DEFINE_integer('max_hw', 256,
                        'maximum height and width to train the model on.')
tf.flags.DEFINE_string('hw_to_batch_size', '{4: 16, 8: 16, 16: 16, 32: 16, 64: 12, 128: 12, 256: 12, 512: 6}',
                       'String expression for the dictionary where key = hw integer and value = batch size integer.'
                       'The flag `batch_size` will be ignored and modified according to this flag.'
                       'Recommended value for training TwinGAN:'
                       '{4: 8, 8: 8, 16: 8, 32: 8, 64: 8, 128: 4, 256: 3, 512: 2}'
                       'Changing this flag half way during training a model will result in bugs.')

FLAGS = tf.flags.FLAGS


def set_flags(param_val_dict):
  """ Set/reset the values of tensorflow flags.
  :param param_val_dict: a dictionary containing argument names and values
  """
  FLAGS.__dict__['__parsed'] = False
  for param, val in param_val_dict.iteritems():
    setattr(FLAGS, param, val)  # Set tensorflow flags.


def select_program(name):
  # The alternative naming is mainly for backward compatibility with some old scripts. Please ignore them.
  if name == 'image_generation' or name == 'image_translation':
    model = image_generation.GanModel()
  elif name == 'twingan' or name == 'image_translation_sc':
    model = twingan.GanModel()
  else:
    raise NotImplementedError('model %s not implemented.' % (name))
  return model


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  # Get flag values before flags are modified.
  base_dir = FLAGS.train_dir
  is_training = FLAGS.is_training
  last_train_dir = None

  resolutions = [2 ** i for i in range(int(math.log(FLAGS.start_hw, 2)), int(math.log(FLAGS.max_hw, 2)) + 1)]
  hw_to_batch_size = ast.literal_eval(FLAGS.hw_to_batch_size)
  # Iterate over the list of resolutions to train.
  for res in resolutions:
    batch_size = hw_to_batch_size[res]
    # Train this many steps per resolution.
    max_number_of_steps = int(FLAGS.num_images_per_resolution / batch_size)

    for is_growing in [True, False, ]:
      # First stage is not growing.
      if is_growing and res == resolutions[0]:
        continue
      # Train indefinitely for the last stage.
      if res == resolutions[-1] and not is_growing:
        max_number_of_steps = 10000000

      if is_growing:
        current_train_dir = os.path.join(base_dir, '%dto%d' % (res / 2, res))
      else:
        current_train_dir = os.path.join(base_dir, '%d' % (res))

      # Check existing checkpoints.
      checkpoint_in_current_train_dir = tf.train.latest_checkpoint(current_train_dir)
      if checkpoint_in_current_train_dir:
        # Load from this checkpoint in the current training directory.
        last_train_dir = current_train_dir
        checkpoint_steps = checkpoint_in_current_train_dir.split('.ckpt-')
        if len(checkpoint_steps) != 2:
          tf.logging.warning('Invalid checkpoint steps.')
        checkpoint_steps = int(checkpoint_steps[1])
        if checkpoint_steps >= max_number_of_steps:
          tf.logging.info('Skipping already trained model %s .' % (current_train_dir))
          continue

      # Eval specific check.
      if not is_training and not checkpoint_in_current_train_dir:
        tf.logging.warning('Checkpoint for resolution %d does not exist yet! Falling back to the previous checkpoint.'
                           % (res))
        current_train_dir = last_train_dir
        if is_growing:
          res = res / 2
          is_growing = False
        else:
          is_growing = True

      # Modify flags that are changing per training stage.
      flags_dict = {'is_growing': is_growing,
                    'train_image_size': res,
                    'max_number_of_steps': max_number_of_steps,
                    'train_dir': current_train_dir,
                    'batch_size': batch_size,
                    'ignore_missing_vars': is_growing}
      flags_dict['eval_dir'] = os.path.join(current_train_dir, 'eval')
      if last_train_dir:
        flags_dict['checkpoint_path'] = last_train_dir
      if not is_training:
        flags_dict['checkpoint_path'] = current_train_dir
      if FLAGS.do_export:
        assert not is_training
        flags_dict['export_path'] = os.path.join(current_train_dir, 'export', str(int(time.time())))
      set_flags(flags_dict)

      # Run the program.
      model = select_program(FLAGS.program_name)
      model.main()
      if FLAGS.is_training == False:
        return

      # Reset.
      tf.reset_default_graph()
      last_train_dir = current_train_dir


if __name__ == '__main__':
  tf.app.run()
