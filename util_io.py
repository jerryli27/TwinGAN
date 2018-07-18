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
"""This file contains utility functions for general purposes file/folder/image reading/writing."""
import errno
import os
import random
from os.path import dirname

import numpy as np
import scipy.misc
import tensorflow as tf
from PIL import Image
from typing import Union


###########
# Folders #
###########

def touch_folder(file_path):
  # type: (Union[str,unicode]) -> None
  """Create a folder along with its parent folders recursively if they do not exist."""
  # Taken from https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist .
  if not file_path.endswith('/'):
    file_path = file_path + "/"
  dn = dirname(file_path)
  if dn != '':
    try:
      os.makedirs(dn)
    except OSError as e:
      if e.errno != errno.EEXIST:
        raise


#########
# Files #
#########

def get_files_in_dir(directory, do_sort=False, do_random_ordering=False,
                     allowed_extensions={'.jpg', '.png', '.jpeg'}):
  """Returns all files in the directory and subdirectories with certain extensions.
  :param directory: The parent directory of the images, or a file containing paths to images.
  :param do_sort: returns a sorted list.
  :param do_random_ordering: returns a deliberately shuffled list.
  :param allowed_extensions: (optional) a set of allowed extensions. If not set, it allows all extensions.
  :return: A sorted list of paths to images in the directory as well as all of its subdirectories.
  """
  assert not (do_random_ordering and do_sort), '`do_sort` and `do_random_ordering` cannot both be true'
  if os.path.isdir(directory):
    if not directory.endswith('/'):
      directory = directory + "/"
    content_dirs = []
    for path, subdirs, files in os.walk(directory):
      for name in files:
        full_file_path = os.path.join(path, name)
        _, ext = os.path.splitext(full_file_path)
        ext = ext.lower()
        if allowed_extensions and ext in allowed_extensions:
          content_dirs.append(full_file_path)
    if len(content_dirs) == 0:
      print('There is no requested file in directory %s.' % directory)
  elif os.path.isfile(directory):
    content_dirs = []
    with open(directory, 'r') as f:
      for line in f.readlines():
        line = line.strip()
        if len(line) > 0:
          content_dirs.append(line)
    if len(content_dirs) == 0:
      print('File %s is empty.' % directory)
  else:
    content_dirs = []
    print('There is no file or directory named %s.' % directory)
  if do_sort:
    content_dirs.sort()
  elif do_random_ordering:
    random.shuffle(content_dirs)
  return content_dirs


##########
# Images #
##########

def imread(path, shape=None, bw=False, rgba=False, dtype=np.float32):
  # type: (str, tuple, bool, bool, np.dtype) -> np.ndarray
  """Reads an image.
  :param path: path to the image
  :param shape: (Height, width)
  :param bw: Whether the image is black and white.
  :param rgba: Whether the image is in rgba format.
  :param dtype: dtype of the returned array.
  :return: np array with shape (height, width, num_color(1, 3, or 4))
  """
  assert not (bw and rgba)
  if bw:
    convert_format = 'L'
  elif rgba:
    convert_format = 'RGBA'
  else:
    convert_format = 'RGB'

  if shape is None:
    return np.asarray(Image.open(path).convert(convert_format), dtype)
  else:
    return np.asarray(Image.open(path).convert(convert_format).resize((shape[1], shape[0])), dtype)


def imsave(path, img):
  # type: (str, np.ndarray) -> None
  """
  Automatically clip the image represented in a numpy array to 0~255 and save the image.
  :param path: Path to save the image.
  :param img: Image represented in numpy array with a legal format for scipy.misc.imsave
  :return: None
  """
  if img.shape[-1] > 3:
    # Convert the image into one channel by summing all channels together
    img = np.sum(img, axis=-1, keepdims=True)
  img = np.clip(img, 0, 255).astype(np.uint8)
  if len(img.shape) == 3 and img.shape[-1] == 1:
    img = np.squeeze(img, -1)
  scipy.misc.imsave(path, img)


def save_float_image(filename, img):
  """Saves a numpy image to `filename` assuming the image has values from 0~1.0"""
  img = img * 255.0
  img = img.astype(np.int32)
  return imsave(filename, img)


##############
# Tensorflow #
##############

# Adapted from https://github.com/davidsandberg/facenet/blob/master/src/facenet.py
def load_model(model, input_map=None):
  """Loads a tensorflow model and restore the variables to the default session."""
  # Check if the model is a model directory (containing a metagraph and a checkpoint file)
  #  or if it is a protobuf file with a frozen graph
  model_exp = os.path.expanduser(model)
  if (os.path.isfile(model_exp)):
    print('Model filename: %s' % model_exp)
    with tf.gfile.FastGFile(model_exp, 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      tf.import_graph_def(graph_def, input_map=input_map, name='')
  else:
    print('Model directory: %s' % model_exp)
    meta_file, ckpt_file = get_model_filenames(model_exp)

    print('Metagraph file: %s' % meta_file)
    print('Checkpoint file: %s' % ckpt_file)

    saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file), input_map=input_map)
    saver.restore(tf.get_default_session(), os.path.join(model_exp, ckpt_file))


def get_model_filenames(model_dir):
  ckpt = tf.train.get_checkpoint_state(model_dir)
  if ckpt and ckpt.model_checkpoint_path:
    ckpt_file = os.path.basename(ckpt.model_checkpoint_path)
    meta_file = ckpt_file + '.meta'
    return meta_file, ckpt_file
  else:
    raise ValueError('No checkpoint file found in the model directory (%s)' % model_dir)
