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
"""General GAN framework for training image translation/generation networks."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import copy
import functools
import math
import os

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.gan.python.eval.python import sliced_wasserstein_impl as swd

import util_misc
import util_io
from datasets import dataset_factory
from deployment import model_deploy
from libs import ops
from model import model_inheritor
from nets import cyclegan
from nets import cyclegan_dis
from nets import nets_factory
from nets import pggan

#################
# Dataset Flags #
#################
# Inherited from `model_inheritor`.

#################
# Network Flags #
#################
tf.flags.DEFINE_string(
  'generator_network', 'pggan',
  'The name of the generator architecture, one of the supported network under _select_network() such as "pggan".')

tf.flags.DEFINE_boolean(
  'use_conditional_labels', False,
  'If true, use conditional_labels in generator/discriminator.')

tf.flags.DEFINE_integer(
  'cyclegan_num_channels', 32,
  'Number of channels/filters for cyclegan.')

tf.flags.DEFINE_boolean(
  'do_self_attention', False,
  'If true, adds self attention layer at `self_attention_hw` layers in the encoder, generator, and the discriminator.')
tf.flags.DEFINE_integer(
  'self_attention_hw', 64,
  'See `do_self_attention`.')

tf.flags.DEFINE_boolean(
  'is_growing', None,
  'Used only for PGGAN. If true, then the model is growing.'
  'Note: is_growing and max_number_of_steps does not interact well when batch size changes half way during training.')
tf.flags.DEFINE_integer(
  'grow_start_number_of_steps', 0,
  'The number of training steps when current cycle of growth starts.')

##############
# Loss Flags #
##############

tf.flags.DEFINE_string(
  'loss_architecture', 'dragan',
  'The name of the loss architecture, one of "gan", "wgan_gp", "wgan", "dragan".')
tf.flags.DEFINE_float(
  'gan_weight', 1.0,
  'The weight for the GAN losses. Does not include weights for "wgan_gp" or "dragan".')
tf.flags.DEFINE_integer(
  'n_critic', 2,
  'The generator is updated every n_critic rounds and the discriminator is updated on the other rounds.'
  'e.g. If n_critic == 2, generator and discriminator are updated alternatingly.')

tf.flags.DEFINE_float(
  'gradient_penalty_lambda', 10,
  'Gradient Penalty weight for WGAN GP and DRAGAN model. Default in the papers is 10 for WGAN GP and 10 for DRAGAN.'
  'Note that larger values can also lead to unrealistic outputs.')
tf.flags.DEFINE_float(
  'wgan_drift_loss_weight', 0.0,
  'Drift loss weight for WGAN (and GP) model.')

tf.flags.DEFINE_boolean(
  'use_gdrop', False,
  'If true, Adds a general dropping term in the discriminator. Used by the PGGAN to ensure training stability.')
tf.flags.DEFINE_float(
  'gdrop_coef', 0.2,
  'gdrop parameter. gdrop_strength = gdrop_coef * tf.pow(tf.maximum(generator_loss_cur - gdrop_lim, 0.0), gdrop_exp)')
tf.flags.DEFINE_float(
  'gdrop_lim', 0.5,
  'gdrop parameter. gdrop_strength = gdrop_coef * tf.pow(tf.maximum(generator_loss_cur - gdrop_lim, 0.0), gdrop_exp)')
tf.flags.DEFINE_float(
  'gdrop_exp', 2.0,
  'gdrop parameter. gdrop_strength = gdrop_coef * tf.pow(tf.maximum(generator_loss_cur - gdrop_lim, 0.0), gdrop_exp)')

tf.flags.DEFINE_boolean(
  'use_ttur', False,
  'If true, D and G uses different learning rate following "Two time-scale update rule for training GANs". The flag'
  '`learning_rate` is assumed to be the generator learning rate. Discriminator learning rate flag is defined below.'
)
tf.flags.DEFINE_float(
  'discriminator_learning_rate', 0.0004,
  'Only used when `use_ttur` flag is set.'
)

#################
# Logging Flags #
#################

tf.flags.DEFINE_integer(
  'log_image_every_n_iter', 1000,
  'Every n iteration, output samples of input and generated images.')
tf.flags.DEFINE_integer(
  'log_image_n_per_hw', 8,
  'Stack n images on each side of the image square. Used along with `log_image_every_n_iter`.')

##############################
# Fine-Tuning and Eval Flags #
##############################

tf.flags.DEFINE_integer(
  'eval_every_n_iter_in_training', 0,
  'Every n iteration, do evaluation on the generated output.')

tf.flags.DEFINE_boolean(
  'calc_inception_score', False,
  'If true, calculates the inception score of the generated images.')

tf.flags.DEFINE_boolean(
  'calc_swd', False,
  'If true, calculates the sliced wasserstein score as described in sec. 5 of PGGAN paper.'
)
tf.flags.DEFINE_boolean(
  'use_tf_swd', False,
  'If true, uses tensorflow native implementation (which has a bug as of tf1.8).'
)
tf.flags.DEFINE_integer(
  'swd_num_images', 1024,
  'Number of randomly sampled images used to calculate swd. For eval please use a large number, such as 8196.'
)
tf.flags.DEFINE_boolean(
  'swd_save_images', False,
  'If true, save the generated images for debugging.'
)

tf.flags.DEFINE_string(
  'incep_classifier_name', None,
  '')
tf.flags.DEFINE_string(
  'incep_classifier_path', None,
  '')
tf.flags.DEFINE_integer(
  'incep_num_classes', None,
  '')

tf.flags.DEFINE_boolean(
  'output_single_file', False,
  'If true, the output mode will only output one file.')
tf.flags.DEFINE_string(
  'output_single_file_name', 'output.csv',
  'Name of the output file.')

FLAGS = tf.flags.FLAGS

#############
# Constants #
#############
GENERATOR_SCOPE = 'generator'
DISCRIMINATOR_SCOPE = 'discriminator'
GENERATOR_LOSS_COLLECTION = 'GENERATOR_LOSSES'
DISCRIMINATOR_LOSS_COLLECTION = 'DISCRIMINATOR_LOSSES'
CUSTOM_GENERATED_TARGETS = 'custom_generated_targets'
CUSTOM_INPUT_PH = 'custom_input_ph'
GDROP_STRENGTH_VAR_NAME = 'gdrop_strength'


class GanModel(model_inheritor.GeneralModel):
  #######################
  # Select the dataset  #
  #######################
  def _select_dataset(self):
    """Selects and returns the dataset used for training/eval.

    :return: One ore more slim.dataset.Dataset.
    """
    dataset = super(GanModel, self)._select_dataset()
    if FLAGS.unpaired_target_dataset_name:
      target_dataset = dataset_factory.get_dataset(
        FLAGS.unpaired_target_dataset_name, FLAGS.dataset_split_name, FLAGS.unpaired_target_dataset_dir)
      return (dataset, target_dataset)
    else:
      return dataset

  ######################
  # Select the network #
  ######################
  def _select_network(self):
    get_noise_shape = None
    if FLAGS.generator_network == 'pggan':
      generator_network_fn = pggan.generator
      discriminator_network_fn = pggan.discriminator
      get_noise_shape = pggan.get_noise_shape
    elif FLAGS.generator_network == 'cyclegan':
      generator_network_fn = cyclegan.cyclegan_generator_resnet
      discriminator_network_fn = cyclegan_dis.cyclegan_discriminator_resnet
    else:
      raise NotImplementedError('Generator network %s is not implemented.', FLAGS.generator_network)
    return {'generator_network_fn': generator_network_fn,
            'discriminator_network_fn': discriminator_network_fn,
            'get_noise_shape': get_noise_shape, }

  ####################
  # Define the model #
  ####################
  @staticmethod
  def _clone_fn(networks, batch_queue, batch_names, data_batched=None, is_training=False, **kwargs):
    """Allows data parallelism by creating multiple clones of network_fn."""
    # Get Data
    data_batched = super(GanModel, GanModel)._get_data_batched(batch_queue, batch_names, data_batched)
    targets = data_batched.get('target')

    # Get network functions
    generator_network_fn = networks['generator_network_fn']
    discriminator_network_fn = networks['discriminator_network_fn']
    get_noise_shape = networks['get_noise_shape']
    global_step = kwargs['global_step']

    # Source will be either None or a tensor which the generator output is conditioned on.
    generator_input = data_batched.get('source', None)

    # Define kwargs.
    generator_kwargs = {'is_training': is_training, 'target_shape': targets.shape}
    discriminator_kwargs = {'is_training': is_training}
    if FLAGS.generator_network == 'cyclegan':
      del generator_kwargs['target_shape']
      generator_kwargs['num_outputs'] = targets.shape[-1]
      generator_kwargs['num_filters'] = FLAGS.cyclegan_num_channels
    elif FLAGS.generator_network == 'pggan':
      sources, targets, alpha_grow = GanModel.get_growing_source_and_target(data_batched, global_step)
      GanModel._add_pggan_kwargs(data_batched, sources, targets, alpha_grow, generator_kwargs, discriminator_kwargs)

    with tf.variable_scope(GENERATOR_SCOPE):
      generated_targets, generator_end_points = generator_network_fn(generator_input, **generator_kwargs)


    if generator_input is None:
      custom_input_ph = tf.placeholder(targets.dtype, shape=get_noise_shape(), name=CUSTOM_INPUT_PH)
    else:
      custom_input_ph = tf.placeholder(generator_input.dtype, shape=generator_input.shape, name=CUSTOM_INPUT_PH)
    with tf.variable_scope(GENERATOR_SCOPE, reuse=True):
      not_training_generator_kwargs = copy.copy(generator_kwargs)
      not_training_generator_kwargs['is_training'] = False
      custom_generated_targets, custom_generator_end_points = generator_network_fn(custom_input_ph,
                                                                                   **not_training_generator_kwargs)
      # Do post-processing for outputting the custom output image.
      custom_generated_targets = GanModel._post_process_image(custom_generated_targets)
    # Change name for convenience during inference.
    custom_generated_targets = tf.identity(custom_generated_targets, name=CUSTOM_GENERATED_TARGETS)

    with tf.variable_scope(DISCRIMINATOR_SCOPE):
      real_target_prediction, real_target_end_points = discriminator_network_fn(targets, **discriminator_kwargs)
    with tf.variable_scope(DISCRIMINATOR_SCOPE, reuse=True):
      generated_target_prediction, generated_target_end_points = discriminator_network_fn(generated_targets,
                                                                                          **discriminator_kwargs)

    # Combine the end points.
    end_points = util_misc.combine_dicts({
      'generator': generator_end_points,
      'discriminator_real': real_target_end_points,
      'discriminator_generated': generated_target_end_points
    })
    if generator_input is not None:
      end_points['sources'] = generator_input
    end_points['targets'] = targets
    end_points[CUSTOM_GENERATED_TARGETS] = custom_generated_targets
    end_points[CUSTOM_INPUT_PH] = custom_input_ph

    # Define ops for evaluation during training.
    if FLAGS.use_tf_swd:
      GanModel._prepare_tf_swd(end_points, targets, generated_targets)

    #############################
    # Specify the loss function #
    #############################
    GanModel.add_loss(data_batched, end_points, functools.partial(discriminator_network_fn, **discriminator_kwargs))
    return end_points

  ####################
  # Define the loss  #
  ####################
  @staticmethod
  def add_loss(data_batched, end_points, discriminator_network_fn=None):
    GanModel.add_gan_loss(end_points['discriminator_generated_prediction'], end_points['discriminator_real_prediction'],
                          end_points['generator_output'], end_points['targets'], discriminator_network_fn)
    if FLAGS.generator_network == 'cyclegan':
      tf.logging.log_every_n(tf.logging.INFO, 'Assuming cyclegan has a paired dataset.', 100)
      tf.losses.absolute_difference(end_points['targets'], end_points['generator_output'],
                                    scope='l1_loss', loss_collection=GENERATOR_LOSS_COLLECTION)

  @staticmethod
  def add_gan_loss(generated_prediction, real_prediction, generated_image, real_image, discriminator_network_fn,
                   name_postfix='', discriminator_var_scope=DISCRIMINATOR_SCOPE, only_real_fake_loss=False,
                   overall_weight=1.0):
    """This function takes the combined end_points and adds gan losses to the corresponding loss collections."""
    assert (generated_prediction is not None and
            real_prediction is not None and
            generated_image is not None and
            real_image is not None)
    # Generator fool discriminator loss.
    generator_loss_name = 'generator_fool_loss%s' % (name_postfix)
    generated_prediction = tf.cast(generated_prediction, tf.float32)
    real_prediction = tf.cast(real_prediction, tf.float32)

    if FLAGS.loss_architecture in ['wgan_gp', 'wgan', 'hinge']:
      # Note the losses below are not returned because it's added to the loss collection already.
      generator_fool_loss = tf.negative(tf.reduce_mean(generated_prediction), name=generator_loss_name)
      generator_fool_loss = tf.losses.compute_weighted_loss(generator_fool_loss,
                                                            weights=FLAGS.gan_weight * overall_weight,
                                                            scope=generator_loss_name,
                                                            loss_collection=GENERATOR_LOSS_COLLECTION)
    else:
      assert FLAGS.loss_architecture == 'gan' or FLAGS.loss_architecture == 'dragan'
      # Equivalent to maximizing log D(G(z)).
      generator_fool_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(generated_prediction), generated_prediction,
                                                            weights=FLAGS.gan_weight * overall_weight,
                                                            scope=generator_loss_name,
                                                            loss_collection=GENERATOR_LOSS_COLLECTION)

    discriminator_loss_name = 'discriminator_loss%s' % (name_postfix)

    if FLAGS.loss_architecture == 'wgan_gp' or FLAGS.loss_architecture == 'wgan':
      # Discriminator loss from WGAN
      discriminator_loss = tf.subtract(tf.reduce_mean(generated_prediction), tf.reduce_mean(real_prediction),
                                       name=discriminator_loss_name)
      discriminator_loss = tf.losses.compute_weighted_loss(discriminator_loss,
                                                           weights=FLAGS.gan_weight * overall_weight,
                                                           scope=discriminator_loss_name,
                                                           loss_collection=DISCRIMINATOR_LOSS_COLLECTION)
      if only_real_fake_loss:
        return

      # Adds additional penalty term to keep the scores from drifting too far from zero.
      if FLAGS.wgan_drift_loss_weight:
        discriminator_drift_loss_name = 'discriminator_drift_loss%s' % (name_postfix)
        discriminator_drift_loss = (tf.constant(FLAGS.wgan_drift_loss_weight, dtype=real_prediction.dtype)
                                    * tf.reduce_mean(tf.square(real_prediction, name='discriminator_drift_loss')))
        discriminator_drift_loss = tf.losses.compute_weighted_loss(discriminator_drift_loss,
                                                                   weights=1.0 * overall_weight,
                                                                   scope=discriminator_drift_loss_name,
                                                                   loss_collection=DISCRIMINATOR_LOSS_COLLECTION)

      ########################
      # WGAN GP loss         #
      ########################
      if FLAGS.loss_architecture == 'wgan_gp':
        scope_name = 'discriminator_gradient_penalty%s' % (name_postfix)
        gradient_penalty = GanModel._add_wgan_gp_loss(real_image,
                                                      generated_image,
                                                      discriminator_network_fn,
                                                      loss_scope=scope_name,
                                                      discriminator_var_scope=discriminator_var_scope,
                                                      overall_weight=overall_weight)

    elif FLAGS.loss_architecture == 'hinge':
      # Discriminator loss from WGAN
      discriminator_loss = tf.add(tf.reduce_mean(tf.nn.relu(1+generated_prediction)),
                                  tf.reduce_mean(tf.nn.relu(1-real_prediction)),
                                  name=discriminator_loss_name)
      discriminator_loss = tf.losses.compute_weighted_loss(discriminator_loss,
                                                           weights=FLAGS.gan_weight * overall_weight,
                                                           scope=discriminator_loss_name,
                                                           loss_collection=DISCRIMINATOR_LOSS_COLLECTION)
    elif FLAGS.loss_architecture == 'gan' or FLAGS.loss_architecture == 'dragan':
      # Equivalent to minimizing -(log D(x) + log (1 - D(G(z))))
      discriminator_fake_loss = tf.losses.sigmoid_cross_entropy(tf.zeros_like(generated_prediction),
                                                                generated_prediction,
                                                                weights=FLAGS.gan_weight * overall_weight,
                                                                scope='discriminator_fake_loss%s' % (name_postfix),
                                                                loss_collection=DISCRIMINATOR_LOSS_COLLECTION)
      discriminator_real_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(real_prediction),
                                                                real_prediction,
                                                                weights=FLAGS.gan_weight * overall_weight,
                                                                scope='discriminator_real_loss%s' % (name_postfix),
                                                                loss_collection=DISCRIMINATOR_LOSS_COLLECTION)
      if only_real_fake_loss:
        return
      if FLAGS.loss_architecture == 'dragan':
        scope_name = 'discriminator_gradient_penalty%s' % (name_postfix)
        gradient_penalty = GanModel._add_dragan_loss(real_image,
                                                     discriminator_network_fn,
                                                     loss_scope=scope_name,
                                                     var_scope=discriminator_var_scope,
                                                     overall_weight=overall_weight)
    else:
      raise NotImplementedError('unsupported loss architecture: %s' %FLAGS.loss_architecture)

  @staticmethod
  def _add_wgan_gp_loss(real_image, generated_image, discriminator_network_fn, loss_scope='gradient_penalty',
                        discriminator_var_scope=DISCRIMINATOR_SCOPE, overall_weight=1.0):
    tf.logging.info('wgan_gp does not interact well when discriminator uses batch norm (according to wgan-gp paper).'
                    'Double check the discriminator architecture if necessary.')
    with tf.variable_scope('wgan_gp'):
      alpha_shape = tf.TensorShape(
        [real_image.shape[0]] + [1 for _ in range(1, len(real_image.shape))])
      alpha = tf.random_uniform(shape=alpha_shape, minval=0., maxval=1., name='alpha')
      real_image_casted = tf.cast(real_image, generated_image.dtype)
      interpolates = real_image_casted + alpha * (generated_image - real_image_casted)
    with tf.variable_scope(discriminator_var_scope, reuse=True):
      interpolate_prediction, interpolate_end_points = discriminator_network_fn(interpolates)
      interpolate_prediction = tf.cast(interpolate_prediction, tf.float32)

    interpolate_gradients = tf.gradients(interpolate_prediction, [interpolates])[0]
    interpolate_gradients = tf.cast(interpolate_gradients, tf.float32)
    interpolate_slopes = tf.sqrt(tf.reduce_sum(tf.square(interpolate_gradients), axis=[1, 2, 3]))

    gradient_penalty = tf.reduce_mean(tf.square((interpolate_slopes - 1.0)))
    # WGAN-GP loss
    gradient_penalty = tf.losses.compute_weighted_loss(gradient_penalty,
                                                       weights=FLAGS.gradient_penalty_lambda * overall_weight,
                                                       scope=loss_scope,
                                                       loss_collection=DISCRIMINATOR_LOSS_COLLECTION)
    return gradient_penalty

  @staticmethod
  def get_perturbed_batch(minibatch):
    """Adds a random noise to each item in the minibatch. Used by dragan loss."""
    # Notice that std is calculated within a mini-batch, implying that lower batch size may yield worse performance.
    std = tf.nn.moments(minibatch, axes=[i for i in range(len(minibatch.shape))])[1]
    # Note that different from the "On Convergence and Stability of GANs"paper, which uses N(0,cI) where c ~ 10, here
    # we follow "How to Train Your DRAGAN", except that we change random uniform from [0,1] to [-1,1]
    # https://github.com/pfnet-research/chainer-gan-lib/blob/master/dragan/updater.py
    return minibatch + 0.5 * std * tf.random_uniform(minibatch.shape, minval=-1.0, maxval=1.0, dtype=minibatch.dtype)

  @staticmethod
  def _add_dragan_loss(real_image, discriminator_network_fn, loss_scope='gradient_penalty',
                       var_scope=DISCRIMINATOR_SCOPE,
                       overall_weight=1.0):
    tf.logging.info('Dragan does not interact well with batch norm in both generator and discriminator, according to '
                    '"How to Train Your DRAGAN". Please double check your network setup.')
    with tf.variable_scope('dragan'):
      alpha_shape = tf.TensorShape([real_image.shape[0]] + [1 for _ in range(1, len(real_image.shape))])
      alpha = tf.random_uniform(shape=alpha_shape, minval=0., maxval=1., name='alpha', dtype=real_image.dtype)
      difference = GanModel.get_perturbed_batch(real_image) - real_image
      interpolates = real_image + alpha * difference
    with tf.variable_scope(var_scope, reuse=True):
      interpolate_prediction, interpolate_end_points = discriminator_network_fn(interpolates)
      interpolate_prediction = tf.cast(interpolate_prediction, tf.float32)

    interpolate_gradients = tf.gradients(interpolate_prediction, [interpolates])[0]
    interpolate_gradients = tf.cast(interpolate_gradients, tf.float32)
    interpolate_slopes = tf.sqrt(tf.reduce_sum(tf.square(interpolate_gradients),
                                               reduction_indices=[i for i in
                                                                  range(1, len(interpolate_gradients.shape))]))
    gradient_penalty = tf.reduce_mean((interpolate_slopes - 1.0) ** 2)
    gradient_penalty = tf.losses.compute_weighted_loss(gradient_penalty,
                                                       weights=FLAGS.gradient_penalty_lambda * overall_weight,
                                                       scope=loss_scope,
                                                       loss_collection=DISCRIMINATOR_LOSS_COLLECTION)
    return gradient_penalty

  ################
  # Optimization #
  ################
  def _get_generator_variable_scopes(self):
    return [GENERATOR_SCOPE]

  def _get_discriminator_variable_scopes(self):
    return [DISCRIMINATOR_SCOPE]

  def _get_generator_variables_to_train(self):
    generator_variables_to_train = []
    for scope in self._get_generator_variable_scopes():
      generator_variables_to_train += self._get_variables_to_train(trainable_scopes=scope)
    return generator_variables_to_train

  def _get_discriminator_variables_to_train(self):
    generator_variables_to_train = []
    for scope in self._get_discriminator_variable_scopes():
      generator_variables_to_train += self._get_variables_to_train(trainable_scopes=scope)
    return generator_variables_to_train

  def _check_trainable_vars(self, generator_variables_to_train, discriminator_variables_to_train):
    assert len(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)) == (len(generator_variables_to_train) +
                                                                        len(discriminator_variables_to_train))

  @staticmethod
  def _configure_learning_rate(num_samples_per_epoch, global_step, start_learning_rate=None):
    """Configures the learning rate.

    Args:
      num_samples_per_epoch: The number of samples in each epoch of training.
      global_step: The global_step tensor.
      start_learning_rate: An optional float specifying the starting learning rate.
        If unspecified, it uses `FLAGS.learning_rate`.

    Returns:
      A `Tensor` representing the learning rate.

    Raises:
      AssertionError: if the flag `learning_rate_decay_type` is not set to fixed.
      ValueError: if the flag `learning_rate_decay_type` is not supported.
    """
    assert FLAGS.learning_rate_decay_type == 'fixed', 'Only fixed learning rate has been tested in this implementation.'
    return super(GanModel, GanModel)._configure_learning_rate(num_samples_per_epoch, global_step,
                                                              start_learning_rate=start_learning_rate)

  @staticmethod
  def _get_items_in_scope(items, scopes):
    """Given a list of update operations and a list of scopes, separate the ops into in-scope and not-in-scope."""
    in_scopes = []
    not_in_scopes = []
    for tensor in items:
      is_item_in_scope = False
      for scope in scopes:
        if tensor.name.startswith(scope):
          is_item_in_scope = True
          in_scopes.append(tensor)
          break
      if is_item_in_scope:
        continue
      else:
        not_in_scopes.append(tensor)
    return in_scopes, not_in_scopes

  @staticmethod
  def maybe_apply_gradients(optimizer, gradients, step, update_ops=None):
    """If `gradients` is none, increase `step` by 1. Otherwise apply gradient as normal."""
    dependencies = []
    if update_ops is not None:
      dependencies = update_ops
    with tf.control_dependencies(dependencies):
      if gradients:
        return optimizer.apply_gradients(gradients, global_step=step)
      else:
        return tf.cast(step.assign(step + 1), tf.bool)  # because `apply_gradients()` returns a boolean.

  def _get_discriminator_optimizer(self, generator_optimizer, global_step):
    d_optimizer = generator_optimizer
    if FLAGS.use_ttur:
      tf.logging.info('Using TTUR. Generator learning rate: %f, Discriminator learning rate: %f'
                      % (FLAGS.learning_rate, FLAGS.discriminator_learning_rate))
      d_lr = self._configure_learning_rate(0, global_step, start_learning_rate=FLAGS.discriminator_learning_rate)
      d_optimizer = self._configure_optimizer(d_lr)
    return d_optimizer

  def _maybe_add_gdrop_update_op(self, global_step, generator_loss, other_update_ops, summaries):
    """gdrop is used in PGGAN to stabilize training."""
    if FLAGS.use_gdrop:
      # If there exists gdrop_strength variable, update that.
      try:
        gdrop_strength = slim.get_unique_variable(GDROP_STRENGTH_VAR_NAME)
      except ValueError:
        raise ValueError('`gdrop_strength` variable cannot be found!')
      else:
        # Adding the cond may help for training from checkpoints without gdrop?
        gdrop_coef = tf.cond(tf.greater(global_step, 100), lambda: FLAGS.gdrop_coef,  # 0.2
                             lambda: 0.0)
        gdrop_lim = FLAGS.gdrop_lim  # 0.5
        gdrop_exp = FLAGS.gdrop_exp  # 2.0
        generator_loss_cur = tf.clip_by_value(tf.reduce_mean(generator_loss), 0.0, 1.0, )

        ema = tf.train.ExponentialMovingAverage(decay=0.9)
        generator_loss_cur_update_op = ema.apply([generator_loss_cur])
        with tf.control_dependencies([generator_loss_cur_update_op]):
          gdrop_strength_val = gdrop_coef * tf.pow(tf.maximum(generator_loss_cur - gdrop_lim, 0.0), gdrop_exp)
          gdrop_strength_assign_op = gdrop_strength.assign(gdrop_strength_val)
        other_update_ops.append(gdrop_strength_assign_op)
        summaries.add(tf.summary.scalar('gdrop_strength', gdrop_strength))

  def _add_optimization(self, clones, optimizer, summaries, update_ops, global_step):
    # Variables to train.
    generator_variables_to_train = self._get_generator_variables_to_train()
    discriminator_variables_to_train = self._get_discriminator_variables_to_train()
    self._check_trainable_vars(generator_variables_to_train, discriminator_variables_to_train)

    # Check optimizer.
    if FLAGS.loss_architecture == 'wgan' and FLAGS.optimizer != 'rmsprop':
      tf.logging.warning('It is recommended in WGAN paper that the optimizer should be rmsprop.')
    # TTUR
    d_optimizer = self._get_discriminator_optimizer(optimizer, global_step)

    generator_loss, generator_clones_gradients = model_deploy.optimize_clones(
      clones,
      optimizer,
      gradient_scale=self._get_gradient_scale(),
      loss_collection=GENERATOR_LOSS_COLLECTION,
      var_list=generator_variables_to_train)
    discriminator_loss, discriminator_clones_gradients = model_deploy.optimize_clones(
      clones,
      d_optimizer,
      gradient_scale=self._get_gradient_scale(),
      loss_collection=DISCRIMINATOR_LOSS_COLLECTION,
      var_list=discriminator_variables_to_train)
    # Add losses to summary.
    summaries.add(tf.summary.scalar('generator_loss', generator_loss))
    summaries.add(tf.summary.scalar('discriminator_loss', discriminator_loss))
    # Add gradient summaries.
    if generator_clones_gradients:
      summaries |= set(model_deploy.add_gradients_summaries(generator_clones_gradients))
    if discriminator_clones_gradients:
      # Add summaries to the gradients.
      summaries |= set(model_deploy.add_gradients_summaries(discriminator_clones_gradients))

    # Create gradient updates.
    n_critic_counter = tf.get_variable('n_critic_counter', shape=[], dtype=tf.int32, initializer=tf.zeros_initializer(),
                                       trainable=False)

    # Here the `generator_update_ops` includes the encoder as well.
    generator_update_ops, non_gen_ops = GanModel._get_items_in_scope(
      update_ops, self._get_generator_variable_scopes())
    discriminator_update_ops, other_update_ops = GanModel._get_items_in_scope(
      non_gen_ops, self._get_discriminator_variable_scopes())

    # A note on the `tf.cond()` function: It will evaluate both true and false branch regardless of the result, unless
    # the op used in that branch is created within the lambda function.
    # Thus all the update ops for both generator or discriminator will be ran regardless of whether generator or the
    # discriminator is being optimized. Most operations requires both gen and dis to be called, but things like
    # evaluating discriminator score on real data does not... There seems to be no easy way to fix this.
    # Example code: Both generator run and discriminator run will be printed even though only variables on one side
    # gets updated, because update variable op is created inside lambda func where as print is created outside.
    # generator_update_ops.append(tf.Print('gen run', ['gen run', n_critic_counter, global_step], first_n=10))
    # discriminator_update_ops.append(tf.Print('dis run', ['dis run', n_critic_counter, global_step], first_n=10))
    grad_updates = tf.cond(tf.equal(tf.mod(n_critic_counter, FLAGS.n_critic), 0),
                           lambda: GanModel.maybe_apply_gradients(optimizer, generator_clones_gradients,
                                                                  step=n_critic_counter,
                                                                  update_ops=generator_update_ops),
                           lambda: GanModel.maybe_apply_gradients(optimizer, discriminator_clones_gradients,
                                                                  step=n_critic_counter,
                                                                  update_ops=discriminator_update_ops), )

    with tf.control_dependencies([grad_updates, ]):
      increase_global_step = tf.cond(tf.equal(tf.mod(n_critic_counter, FLAGS.n_critic), 0),
                                     lambda: tf.assign(global_step, global_step + 1), lambda: tf.identity(global_step),
                                     name='increase_global_step')
    other_update_ops.append(increase_global_step)
    self._maybe_add_gdrop_update_op(global_step, generator_loss, other_update_ops, summaries)

    update_op = tf.group(*other_update_ops)
    with tf.control_dependencies([update_op]):
      if FLAGS.loss_architecture.startswith('wgan') or FLAGS.loss_architecture == 'hinge':
        train_tensor = tf.identity(discriminator_loss, name='train_op')
      else:
        # The closer discriminator is to 0 (or 1 for non-WGAN), usually the better the output is.
        train_tensor = tf.negative(discriminator_loss, name='train_op')
    return train_tensor

  #################
  # Add summaries #
  #################
  def _add_image_summaries(self, end_points, _):
    # Add summaries for images, if there are any.
    for end_point_name in ['sources', 'targets', 'generator_output']:
      if (end_point_name in end_points and len(end_points[end_point_name].shape) == 4):
        self._add_one_image_summary(end_point_name, self._post_process_image(end_points[end_point_name]))

  @staticmethod
  def _add_loss_summaries(first_clone_scope, summaries, end_points):
    for collection in [tf.GraphKeys.LOSSES, GENERATOR_LOSS_COLLECTION, DISCRIMINATOR_LOSS_COLLECTION]:
      for loss in tf.get_collection(collection, first_clone_scope):
        summaries.add(tf.summary.scalar('losses/%s' % loss.op.name, loss))

  @staticmethod
  def _add_end_point_summaries(end_points, summaries):
    """Wrapper around the inherited _add_end_point_summaries(). Excludes some end points."""
    # Exclude two end points for generated images with custom noise inputs.
    excluded_end_points = {CUSTOM_INPUT_PH, CUSTOM_GENERATED_TARGETS, 'swd_real_ph', 'swd_fake_ph', 'swd'}
    not_excluded_end_points = copy.copy(end_points)
    for end_point in end_points:
      if end_point in excluded_end_points:
        del not_excluded_end_points[end_point]
    super(GanModel, GanModel)._add_end_point_summaries(not_excluded_end_points, summaries)

  ###################################################
  # Extra function to run after each training step. #
  ###################################################
  @staticmethod
  def do_extra_train_step_aux(session, run_list, out_list, feed_dict_per_hw=None):
    """Save the output of each tensor in `run_list` as an image to their corresponding path in `out_list`."""
    image_list = [[] for _ in range(len(out_list))]
    for i in range(FLAGS.log_image_n_per_hw):
      run_results = session.run(run_list, feed_dict=None if feed_dict_per_hw is None else feed_dict_per_hw[i])
      for run_items_i, run_result in enumerate(run_results):
        # For task like anime_faces, the source is a conditional one-hot tensor. Visualize that as an image as well.
        if len(run_result.shape) == 2:
          run_result = np.expand_dims(np.expand_dims(run_result, axis=2), axis=3)
          run_result = np.repeat(run_result, 64, axis=2)
        image_list[run_items_i].append(run_result)

    for i, save_path in enumerate(out_list):
      images = image_list[i]
      if len(images):
        concatenated_images = np.concatenate(images, axis=2)
        concatenated_images = np.reshape(concatenated_images, (
        concatenated_images.shape[0] * concatenated_images.shape[1], concatenated_images.shape[2],
        concatenated_images.shape[3]))
        util_io.save_float_image(save_path, concatenated_images)
    return image_list

  @staticmethod
  def do_extra_train_step(session, end_points, global_step):
    """This function is hooked up to the main training step function. It is executed after each training step.
       Child class should change this function to fit it's input/output.
    """
    current_step = global_step.eval(session)
    if current_step % FLAGS.log_image_every_n_iter == 0:
      image_dir = os.path.join(FLAGS.train_dir, 'generated_samples')
      if not tf.gfile.Exists(image_dir):
        tf.gfile.MakeDirs(image_dir)

      run_list = []
      out_list = []
      if 'generator_conditional_layer' in end_points:
        run_list.append(end_points['generator_conditional_layer'])
        out_list.append(os.path.join(image_dir, '%d_conditional_layer.jpg' % (current_step)))

      if 'sources' in end_points:
        run_list = run_list + [end_points['sources'], end_points['generator_output'], end_points['targets']]
        out_list = out_list + [os.path.join(image_dir, '%d_source.jpg' % (current_step)),
                               os.path.join(image_dir, '%d.jpg' % (current_step)),
                               os.path.join(image_dir, '%d_target.jpg' % (current_step)), ]
        feed_dict_per_hw = None

      else:
        # For image generation task where the generator source is a random tensor.
        target_img_name = 'targets'
        run_list = run_list + [end_points[CUSTOM_GENERATED_TARGETS], end_points[target_img_name]]
        out_list = out_list + [os.path.join(image_dir, '%d.jpg' % (current_step)),
                               os.path.join(image_dir, '%d_target.jpg' % (current_step))]

        np_dtype = getattr(np, FLAGS.dataset_dtype)
        ph = end_points[CUSTOM_INPUT_PH]
        ph_shape = ph.get_shape().as_list()
        np.random.seed(314)
        noise = np.random.standard_normal([FLAGS.batch_size, ] + ph_shape[-1:]).astype(dtype=np_dtype)
        noise = np.expand_dims(np.expand_dims(noise, 1), 1)
        noise2 = np.random.standard_normal([FLAGS.batch_size, ] + ph_shape[-1:]).astype(dtype=np_dtype)
        noise2 = np.expand_dims(np.expand_dims(noise2, 1), 1)

        feed_dict_per_hw = []
        for i in range(FLAGS.log_image_n_per_hw):
          current_noise_vectors = (noise * i + noise2 * (FLAGS.log_image_n_per_hw - i - 1)) / float(FLAGS.log_image_n_per_hw - 1)
          assert list(current_noise_vectors.shape[1:]) == ph_shape[1:]
          feed_dict = {ph: current_noise_vectors}
          feed_dict_per_hw.append(feed_dict)
      GanModel.do_extra_train_step_aux(session, run_list=run_list, out_list=out_list, feed_dict_per_hw=feed_dict_per_hw)
    if FLAGS.eval_every_n_iter_in_training and current_step % FLAGS.eval_every_n_iter_in_training == 0:
      # TODO: Just use swd for now. in the future trigger different eval by flags.
      GanModel._calc_swd(session, end_points, current_step=current_step)

  ########
  # Eval #
  ########
  @staticmethod
  def _define_eval_metrics(end_points, data_batched):
    metric_map = {}
    generator_losses = tf.get_collection(GENERATOR_LOSS_COLLECTION, )
    discriminator_losses = tf.get_collection(DISCRIMINATOR_LOSS_COLLECTION, )
    for loss in generator_losses + discriminator_losses:
      metric_map[loss.name.rstrip('/value:0')] = slim.metrics.streaming_mean(loss)
    return metric_map

  def get_items_to_encode(self, end_points, data_batched):
    """Outputs a list with format (name, is_image, tensor)"""
    items_to_encode = []
    if 'source' in data_batched:
      items_to_encode.append(('sources', True, self._post_process_image(data_batched.get('source'))))
    generated_targets = end_points['generator_output']
    generated_target_prediction = end_points['discriminator_generated_prediction']
    real_target_prediction = end_points['discriminator_real_prediction']
    targets = data_batched.get('target')
    items_to_encode.append(('targets', True, self._post_process_image(targets)))
    items_to_encode.append(('generated_targets', True, self._post_process_image(generated_targets)))
    items_to_encode.append(('generated_target_prediction', False, generated_target_prediction))
    items_to_encode.append(('real_target_prediction', False, real_target_prediction))

    best_generated_target_i = tf.argmax(tf.squeeze(generated_target_prediction, axis=1))
    worst_real_target_i = tf.argmin(tf.squeeze(real_target_prediction, axis=1))

    items_to_encode.append(
      ('best_generated_target', True, self._post_process_image(generated_targets[best_generated_target_i])))
    items_to_encode.append(('worst_real_target', True, self._post_process_image(targets[worst_real_target_i])))
    return items_to_encode

  @staticmethod
  def to_human_friendly(eval_items):
    """For non-image items, use space to join the list values."""
    ret = []
    for name, is_image, vals in eval_items:
      if is_image:
        ret.append((name, is_image, vals))
      else:
        human_readable_vals = []
        for val in vals:
          human_readable_val = []
          for i, item in enumerate(val):
            human_readable_val.append(str(item))
          human_readable_vals.append(' '.join(human_readable_val))
        ret.append((name, is_image, human_readable_vals))
    return ret

  @staticmethod
  def prepare_inception_score_classifier(classifier_name, num_classes, images, return_saver=True):
    network_fn = nets_factory.get_network_fn(
      classifier_name,
      num_classes=num_classes,
      weight_decay=0.0,
      is_training=False,
    )
    # Note: you may need to change the prediction_fn here.
    try:
      logits, end_points = network_fn(images, prediction_fn=tf.sigmoid, create_aux_logits=False)
    except TypeError:
      tf.logging.warning('Cannot specify prediction_fn=tf.sigmoid, create_aux_logits=False.')
      logits, end_points = network_fn(images, )

    variables_to_restore = slim.get_model_variables(scope=nets_factory.scopes_map[classifier_name])
    predictions = end_points['Predictions']
    if return_saver:
      saver = tf.train.Saver(variables_to_restore)
      return predictions, end_points, saver
    else:
      return predictions, end_points

  @staticmethod
  def calc_inception_score(predictions, saver, classifier_path, session,
                           splits=10):
    # Currently this function is not designed for use during training.
    saver.restore(session, util_misc.get_latest_checkpoint_path(classifier_path))
    # The inception score is by convention calculated using 10 batches of 5000 samples
    # with each batch separated into mini-batches.
    num_batches = int(math.ceil(5000.0 * splits / int(predictions.shape[0])))

    preds = []
    for i in range(num_batches):
      pred = session.run(predictions, )
      preds.append(pred)
    preds = np.concatenate(preds, 0)
    scores = []
    for i in range(splits):
      part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
      kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
      kl = np.mean(np.sum(kl, 1))
      scores.append(np.exp(kl))
    return np.mean(scores), np.std(scores)

  @staticmethod
  def _get_swd_real_fake(end_points):
    return end_points['targets'], end_points['generator_output']

  @staticmethod
  def _calc_swd(session, end_points, current_step=0, get_swd_real_fake=None):
    if not FLAGS.train_image_size >= 16:
      tf.logging.log_every_n(tf.logging.INFO, 'Not doing swd on small images.', 100)
      return

    if not FLAGS.is_training:
      tf.logging.info('Beware of preprocessing! If the score you get from training and eval are too different, '
                      'you may be preprocessing images during training but not during eval.')

    num_batches = int(FLAGS.swd_num_images / FLAGS.batch_size)
    util_io.touch_folder(FLAGS.eval_dir)
    if FLAGS.swd_save_images:
      save_image_dir = os.path.join(FLAGS.eval_dir, 'swd_debug', str(int(time.time())))
      util_io.touch_folder(save_image_dir)
    save_result_path = os.path.join(FLAGS.eval_dir, 'swd_%s_step_%d_%d_images.txt' %(
      'train' if FLAGS.is_training else 'eval', current_step, FLAGS.swd_num_images))
    if os.path.exists(save_result_path) and FLAGS.is_training:
      print('not repeating swd calculation during training.')
      return

    if get_swd_real_fake is None:
      get_swd_real_fake = GanModel._get_swd_real_fake
    source, t_prime = get_swd_real_fake(end_points)
    if FLAGS.use_tf_swd:
      reals = []
      fakes = []
      for i in range(num_batches):
        real_minibatch, fake_minibatch = session.run([source, t_prime])
        reals.append(real_minibatch)
        fakes.append(fake_minibatch)
      reals = np.concatenate(reals, axis=0)
      fakes = np.concatenate(fakes, axis=0)

      if FLAGS.swd_save_images:
        for image_i in range(reals.shape[0]):
          util_io.imsave(os.path.join(save_image_dir, str(image_i) + '_real.jpg'),
                         reals[image_i] * 255.0)
          util_io.imsave(os.path.join(save_image_dir, str(image_i) + '_fake.jpg'),
                         fakes[image_i] * 255.0)

      score = session.run(end_points['swd'],
                          feed_dict={end_points['swd_real_ph']: reals, end_points['swd_fake_ph']: fakes})
      score = np.array(score) * 1e3  # In the PGGAN paper numbers are reported on this scale.
      print(score)
      resolutions = []
      res = int(end_points['swd_real_ph'].shape[1])
      while res >= 16:
        resolutions.append(res)
        res //= 2

      with open(save_result_path, 'w') as f:
        f.write('swd sliced wasserstein score evaluated on %d images.\n' % FLAGS.swd_num_images)
        f.write('res\treal\tfake\n')
        for i, hw in enumerate(resolutions):
          f.write('%d\t%f\t%f\n' % (hw, score[i][0], score[i][1]))
        avg = np.average(score, axis=0)
        f.write('Average\t%f\t%f\n' % (avg[0], avg[1]))
        assert len(score) == len(resolutions)
    else:
        raise NotImplementedError('Dependent library cannot be open sourced due to licencing issues. Sorry. :(')

  @staticmethod
  def _prepare_tf_swd(end_points, real, fake):
    raise AssertionError('There is a bug in the tensorflow 1.8 implementation. It is wrongly normalizing by patch.')
    swd_real_ph = tf.placeholder(real.dtype,
                                           tf.TensorShape([None, real.shape[1], real.shape[2], real.shape[3]]),
                                           name='swd_real_ph')
    swd_fake_ph = tf.placeholder(fake.dtype,
                                           tf.TensorShape([None, fake.shape[1], fake.shape[2], fake.shape[3]]),
                                           name='swd_fake_ph')
    distance = swd.sliced_wasserstein_distance(swd_real_ph, swd_fake_ph, patches_per_image=128, random_sampling_count=4, random_projection_dim=128)
    end_points['swd_real_ph'] = swd_real_ph
    end_points['swd_fake_ph'] = swd_fake_ph
    end_points['swd'] = distance


  def _define_extra_eval_actions(self, end_points, data_batched):
    if FLAGS.calc_inception_score:
      return self.prepare_inception_score_classifier(FLAGS.incep_classifier_name, FLAGS.incep_num_classes,
                                                     end_points['generator_output'])
    elif FLAGS.calc_swd:

      source, t_prime = self._get_swd_real_fake(end_points)
      if FLAGS.use_tf_swd:
        self._prepare_tf_swd(end_points, source, t_prime)
      return (end_points, data_batched)
    else:
      raise NotImplementedError('please specify the extra eval action type. e.g. set `calc_swd` flag to True.')

  def _do_extra_eval_actions(self, session, extra_eval):
    if FLAGS.calc_inception_score:
      predictions, end_points, saver = extra_eval
      self.calc_inception_score(predictions, saver, FLAGS.incep_classifier_path, session, )
    elif FLAGS.calc_swd:
     (end_points, data_batched) = extra_eval
     self._calc_swd(session, end_points)


  ##########
  # Export #
  ##########
  @staticmethod
  def _build_signature_def_map(end_points, data_batched):
    # Build the signature_def_map.
    sources = tf.saved_model.utils.build_tensor_info(
      end_points[CUSTOM_INPUT_PH])
    outputs = tf.saved_model.utils.build_tensor_info(
      end_points[CUSTOM_GENERATED_TARGETS])

    domain_transfer_signature = (
      tf.saved_model.signature_def_utils.build_signature_def(
        inputs={
          tf.saved_model.signature_constants.PREDICT_INPUTS:
            sources
        },
        outputs={
          tf.saved_model.signature_constants.PREDICT_OUTPUTS:
            outputs,
        },
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))
    ret = {
      tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
        domain_transfer_signature,
    }
    return ret

  @staticmethod
  def _build_assets_collection(end_points, data_batched):
    return None

  #############################
  # PGGAN specific functions. #
  #############################
  @staticmethod
  def get_growing_image(image, alpha_grow, name_postfix='image'):
    low_res_image = tf.nn.avg_pool(image, (1, 2, 2, 1), (1, 2, 2, 1), 'VALID')
    low_res_image = tf.image.resize_nearest_neighbor(
      low_res_image, (low_res_image.shape[1] * 2, low_res_image.shape[2] * 2), name='low_res_%s' % name_postfix)
    return alpha_grow * image + (1 - alpha_grow) * low_res_image

  @staticmethod
  def get_growing_source_and_target(data_batched, global_step):
    # TODO: should I grow the source as well?
    print('TODO: should I grow the source as well? Run an experiment on this!')
    sources = data_batched.get('source')
    targets = data_batched.get('target')
    if FLAGS.is_growing:
      with tf.variable_scope('alpha_grow'):
        alpha_grow = tf.cast(global_step - FLAGS.grow_start_number_of_steps, targets.dtype) / (
            FLAGS.max_number_of_steps - FLAGS.grow_start_number_of_steps)
        if sources is not None:
          sources = GanModel.get_growing_image(sources, alpha_grow, name_postfix='sources')
        if targets is not None:
          targets = GanModel.get_growing_image(targets, alpha_grow, name_postfix='targets')
    else:
      alpha_grow = 0.0
    return sources, targets, alpha_grow

  @staticmethod
  def _add_pggan_kwargs(data_batched, sources, targets, alpha_grow, generator_kwargs, discriminator_kwargs):
    additional_kwargs = {'is_growing': FLAGS.is_growing, 'alpha_grow': alpha_grow, 'do_self_attention': FLAGS.do_self_attention, 'self_attention_hw': FLAGS.self_attention_hw}
    generator_kwargs.update(**additional_kwargs)
    discriminator_kwargs.update(**additional_kwargs)
    generator_kwargs['do_pixel_norm'] = FLAGS.do_pixel_norm
    generator_kwargs['dtype'] = targets.dtype

    if FLAGS.use_gdrop:
      discriminator_kwargs[GDROP_STRENGTH_VAR_NAME] = slim.model_variable(GDROP_STRENGTH_VAR_NAME, shape=[],
                                                                          dtype=targets.dtype,
                                                                          initializer=tf.zeros_initializer,
                                                                          trainable=False)
    else:
      discriminator_kwargs['do_dgrop'] = False

    # Conditional related params.
    if FLAGS.use_conditional_labels:
      conditional_labels = data_batched.get('conditional_labels', None)
      if conditional_labels is not None:
        generator_kwargs['arg_scope_fn'] = functools.partial(pggan.conditional_progressive_gan_generator_arg_scope,
                                                             conditional_layer=conditional_labels)
        source_embed = GanModel._embed_one_hot(conditional_labels, FLAGS.conditional_embed_dim, )
        discriminator_kwargs['conditional_embed'] = source_embed

  def main(self):
    if not FLAGS.train_image_size:
      raise ValueError('Please set the `train_image_size` flag.')
    super(GanModel, self).main()


def main(_):
  model = GanModel()
  model.main()


if __name__ == '__main__':
  tf.app.run()
