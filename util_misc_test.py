#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Copyright 2018 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import util_misc

import os
import numpy as np
import tensorflow as tf

import util_io

os.environ['CUDA_VISIBLE_DEVICES'] = ''

class UtilMiscTest(tf.test.TestCase):
  def test_extract_random_patches(self):
    with self.test_session() as sess:
      hw = (4, 4)
      num_patches = 5000

      mask = np.zeros((1, hw[0], hw[1], 1), dtype=np.float32)
      mask[0, 0, 0, 0] = 1
      # Not the best test case due to non-deterministic nature, but works for now.
      actual_output_tensor = util_misc.extract_random_patches(mask, patch_sizes=1, num_patches=num_patches)
      actual_output, = self.evaluate([actual_output_tensor,])
      self.assertEqual(actual_output.shape, (num_patches, 1, 1, 1))
      actual_output_avg = np.average(actual_output,)
      self.assertLessEqual(abs(actual_output_avg - 1.0 / hw[0] / hw[1]), 0.01)

  def test_find_boundary(self):
    image = np.ones((50, 50, 3))
    up, down, left, right = 20, 30, 45, 47
    image[up+1:down, left+1:right] = np.zeros((down-up-1, right-left-1, 3))
    point = [(up + down) / 2, (left + right) / 2]
    num_pixels = 5
    threshold = 1

    expected_outputs = {'up': [up, point[1]], 'down': [down, point[1]], 'left': [point[0], left], 'right': [point[0], right]}

    for direction in expected_outputs:
      expected_output = expected_outputs[direction]
      actual_output = util_misc.find_boundary(point, image, direction, num_pixels, threshold)
      self.assertAllEqual(actual_output, expected_output)



if __name__ == '__main__':
  tf.test.main()
