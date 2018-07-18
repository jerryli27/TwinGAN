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
"""Implements gdrop from the PGGAN paper."""

import numpy as np
import tensorflow as tf

def gdrop(layer, mode='prop', strength=0.4, axes=(0,3), normalize=False, input_mode='NHWC'):
  """Implements gdrop from the PGGAN paper."""
  if input_mode == 'NHWC':
    assert axes==(0,3)
  elif input_mode == 'NCHW':
    assert axes==(0,1)
  else:
    raise NotImplementedError('Invalid input_mode', input_mode)

  in_axes = len(layer.shape)
  rnd_shape = tf.TensorShape([layer.shape[i] if i in axes else 1 for i in range(in_axes)])
  one = tf.constant(1, dtype=layer.dtype)

  if mode == 'prop':
    coef = strength * tf.constant(np.sqrt(np.float32(int(layer.shape[-1]))), dtype=layer.dtype)
    rnd = tf.random_normal(rnd_shape, dtype=layer.dtype, name='gdrop') * coef + one
    return layer * rnd
  else:
    raise ValueError('Invalid GDropLayer mode', mode)
