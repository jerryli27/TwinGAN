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

"""Implements self attention in 'Self-Attention Generative Adversarial Networks' https://arxiv.org/abs/1805.08318

If you find any errors in this implementation, please file an issue on Github.
"""

import tensorflow as tf
from libs.sn import convolution

def self_attention_layer(layer, **convolution_kwargs):
  """Implements self attention in SAGAN."""
  # Number of channels in the self attention layer.

  layer_shape = layer.shape  # Assuming NHWC format. Note that all other N below will refer to h*w, not batch size.
  batch_size, n, c = layer_shape[0], layer_shape[1] * layer_shape[2], layer_shape[3]
  c_bar = c / 8
  f = convolution(inputs=layer,
                  num_outputs=c_bar,
                  kernel_size=1,
                  stride=1,
                  activation_fn=tf.nn.tanh,  # tf.nn.relu? Not specified in the paper preprint.
                  scope='sa_f',
                  **convolution_kwargs
                  )
  g = convolution(inputs=layer,
                  num_outputs=c_bar,
                  kernel_size=1,
                  stride=1,
                  activation_fn=tf.nn.tanh,  # tf.nn.relu? Not specified in the paper preprint.
                  scope='sa_g',
                  **convolution_kwargs
                  )
  h = convolution(inputs=layer,
                  num_outputs=c,
                  kernel_size=1,
                  stride=1,
                  activation_fn=None,  # tf.nn.relu? Not specified in the paper preprint.
                  scope='sa_h',
                  **convolution_kwargs
                  )

  # In the paper, f(x) = W_f dot x, where x is (C x N) and W_f is (C_bar x C). Thus f should have dimension (C_bar x N)
  # s = Transpose(f) dot g. Thus s has dimension (N x N), which makes sense. S specifies where each region should focus
  # on spacially.
  f = tf.reshape(f, shape=[-1, n, c_bar])  # Using -1 instead of batch_size because batch_size can be unknown.
  g = tf.reshape(g, shape=[-1, n, c_bar])
  h = tf.reshape(h, shape=[-1, n, c])
  s = tf.matmul(f, g, transpose_b=True)  # Note the input channel format NHWC is different from the paper.

  beta = tf.nn.softmax(s, axis=-1, name='beta')

  o = tf.matmul(beta, h)  # beta = N x N where the last dim N has a softmax distribution. h = N x C.
  o = tf.reshape(o, tf.shape(layer))
  gamma = tf.get_variable('sa_gamma', [1], dtype=layer.dtype, initializer=tf.constant_initializer(0.0), trainable=True)
  y = gamma * o + layer
  return y
