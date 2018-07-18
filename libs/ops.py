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
"""Contains custom tensorflow layers, normalizations, regularizers, etc.

Some functions are taken from https://github.com/minhnhat93/tf-SNDCGAN."""

from libs.sn import convolution
from libs.sn import fully_connected
import libs.batch_norm
import libs.instance_norm
import libs.self_attention
import libs.gdrop
import libs.sn


conditional_batch_norm = libs.batch_norm.conditional_batch_norm
instance_norm = libs.instance_norm.instance_norm
self_attention_layer = libs.self_attention.self_attention_layer
gdrop = libs.gdrop.gdrop
SPECTRAL_NORM_OPS = libs.sn.SPECTRAL_NORM_OPS

def spectral_normed_conv(inputs, num_outputs, kernel_size, **kwargs):
  return convolution(inputs, num_outputs=num_outputs, kernel_size=kernel_size, do_spec_norm=True, **kwargs)

def spectral_normed_fc(inputs, num_outputs, **kwargs):
  if 'kernel_constraint' in kwargs:
    raise ValueError('kernel_constraint should not be in kwargs for `spectral_normed_conv`.')
  return fully_connected(inputs, num_outputs, **kwargs)

