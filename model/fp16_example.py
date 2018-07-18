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
"""Loss scaling mixed precision training example from Nvidia."""
import tensorflow as tf
import numpy as np

def float32_variable_storage_getter(getter, name, shape=None, dtype=None,
                                    initializer=None, regularizer=None,
                                    trainable=True,
                                    *args, **kwargs):
    """Custom variable getter that forces trainable variables to be stored in
    float32 precision and then casts them to the training precision.
    """
    storage_dtype = tf.float32 if trainable else dtype
    variable = getter(name, shape, dtype=storage_dtype,
                      initializer=initializer, regularizer=regularizer,
                      trainable=trainable,
                      *args, **kwargs)
    if trainable and dtype != tf.float32:
        variable = tf.cast(variable, dtype)
    return variable

def gradients_with_loss_scaling(loss, variables, loss_scale):
    """Gradient calculation with loss scaling to improve numerical stability
    when training with float16.
    """
    return [grad / loss_scale
            for grad in tf.gradients(loss * loss_scale, variables)]

def create_simple_model(nbatch, nin, nout, dtype):
    """A simple softmax model."""
    data    = tf.placeholder(dtype, shape=(nbatch, nin))
    weights = tf.get_variable('weights', (nin, nout), dtype)
    biases  = tf.get_variable('biases',        nout,  dtype,
                              initializer=tf.zeros_initializer())
    logits  = tf.matmul(data, weights) + biases
    target  = tf.placeholder(tf.float32, shape=(nbatch, nout))
    # Note: The softmax should be computed in float32 precision
    loss    = tf.losses.softmax_cross_entropy(
        target, tf.cast(logits, tf.float32))
    return data, target, loss

if __name__ == '__main__':
    nbatch = 64
    nin    = 100
    nout   = 10
    learning_rate = 0.1
    momentum      = 0.9
    loss_scale    = 128
    dtype         = tf.float16
    tf.set_random_seed(1234)
    np.random.seed(4321)

    # Create training graph
    with tf.device('/gpu:0'), \
         tf.variable_scope(
             # Note: This forces trainable variables to be stored as float32
             'fp32_storage', custom_getter=float32_variable_storage_getter):
        data, target, loss = create_simple_model(nbatch, nin, nout, dtype)
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        # Note: Loss scaling can improve numerical stability for fp16 training
        grads = gradients_with_loss_scaling(loss, variables, loss_scale)
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
        training_step_op = optimizer.apply_gradients(zip(grads, variables))
        init_op = tf.global_variables_initializer()

    # Run training
    sess = tf.Session()
    sess.run(init_op)
    np_data   = np.random.normal(size=(nbatch, nin)).astype(np.float16)
    np_target = np.zeros((nbatch, nout), dtype=np.float32)
    np_target[:,0] = 1
    print 'Step Loss'
    for step in xrange(30):
        np_loss, _ = sess.run([loss, training_step_op],
                              feed_dict={data: np_data, target: np_target})
        print '%4i %6f' % (step + 1, np_loss)
