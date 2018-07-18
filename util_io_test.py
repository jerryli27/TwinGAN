#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
import os
import shutil
import tempfile
import unittest

import numpy as np
import scipy.misc

import util_io

def _rgb2gray(rgb):
  """Turns a numpy rgb image into grayscale."""
  return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

class TestDataUtilMethods(unittest.TestCase):
  def test_get_files_in_dir_in_dir(self):
    dirpath = tempfile.mkdtemp()
    image_path = dirpath + '/image.jpg'
    f = open(image_path, 'w')
    f.close()
    subfolder = '/subfolder'
    os.makedirs(dirpath + subfolder)
    image_path2 = dirpath + subfolder + u'/骨董屋・三千世界の女主人_12746957.png'
    f = open(image_path2, 'w')
    f.close()
    actual_answer = util_io.get_files_in_dir(dirpath + '/')
    expected_answer = [image_path, image_path2.encode('utf8')]
    shutil.rmtree(dirpath)
    self.assertEqual(expected_answer, actual_answer)

  def test_imread_rgba(self):
    height = 256
    width = 256

    content_folder = tempfile.mkdtemp()
    image_path = content_folder + '/image.png'
    current_image = np.ones((height, width, 4)) * 255.0
    current_image[0, 0, 0] = 0
    scipy.misc.imsave(image_path, current_image)

    content_pre_list = util_io.imread(image_path, rgba=True)
    expected_answer = current_image
    np.testing.assert_almost_equal(expected_answer, content_pre_list)

    shutil.rmtree(content_folder)

  def test_imread_bw(self):
    height = 256
    width = 256

    content_folder = tempfile.mkdtemp()
    image_path = content_folder + u'/骨董屋・三千世界の女主人_12746957.png'
    current_image = np.ones((height, width, 3)) * 255.0
    current_image[0, 0, 0] = 0
    util_io.imsave(image_path, current_image)

    actual_output = util_io.imread(util_io.get_files_in_dir(content_folder + '/')[0], bw=True)

    expected_answer = np.floor(_rgb2gray(np.array(current_image)))
    np.testing.assert_almost_equal(expected_answer, actual_output)

  def test_imread_and_imsave_utf8(self):
    height = 256
    width = 256

    content_folder = tempfile.mkdtemp()
    image_path = content_folder + u'/骨董屋・三千世界の女主人_12746957.png'
    current_image = np.ones((height, width, 3)) * 255.0
    current_image[0, 0, 0] = 0
    util_io.imsave(image_path, current_image)

    actual_output = util_io.imread(util_io.get_files_in_dir(content_folder + '/')[0])

    expected_answer = np.round(np.array(current_image))
    np.testing.assert_almost_equal(expected_answer, actual_output)


if __name__ == '__main__':
  unittest.main()
