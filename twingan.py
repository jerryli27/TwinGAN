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
"""TwinGAN implementation. Key parts are in 'Define the model' and 'Define the loss' sections."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import base64
import copy
import csv
import functools
import os

import numpy as np
import scipy.misc
import tensorflow as tf
import tensorflow.contrib.slim as slim

import image_generation
import util_io
import util_misc
from nets import pggan

#################
# Dataset Flags #
#################
tf.flags.DEFINE_string('custom_sources_np_path', 'sample_source_validation_100.npy',
                       'A path to a numpy array containing images for the source dataset. Used for showing '
                       'the outputs for the same images as training progresses.')

#################
# Network Flags #
#################

tf.flags.DEFINE_boolean('use_style_embedding', False,
                        'If true, then the conditional batch norm layers will be condition on the style embedding.'
                        'See MUNIT paper for details.')
tf.flags.DEFINE_integer('style_embed_size', 16,
                        'Number of dimension for the style embedding.')

tf.flags.DEFINE_boolean(
  'use_unet', False,
  'If true, then the layers in the encoder network will be concatenated to the corresponding '
  'layers in the generator network.')

tf.flags.DEFINE_boolean('do_encoder_distillation', False,
                        'If true, the content encoder distills knowledge from embeddings obtained from a pretrained'
                        'network. Used primarily to encourage image identities to be kept after translation to the '
                        'other domain. For example Elon Musk still should look like Elon after translation.')
tf.flags.DEFINE_float('distillation_weight', 1.0,
                      'Weight on the distillation term.')
tf.flags.DEFINE_float('distillation_start_hw', 16,
                      'Do not apply distillation loss until it reaches this hw.')

##############
# Loss Flags #
##############

tf.flags.DEFINE_float(
  'l_cyc_weight', 1.0,
  'Weight of the cycle consistency loss term where the encoder-decoder acts as an identity function for inputs and '
  'outputs belonging to the same domain. Used for cross-domain GAN models.')
tf.flags.DEFINE_boolean(
  'do_l_cyc_gan', True,
  'If true, add GAN loss on cycle output G(E(x)).')
tf.flags.DEFINE_float(
  'l_content_weight', 0.1,
  'Weight of the content embedding loss term. L = E(x) - E(G(E(x)))')

##############################
# Fine-Tuning and Eval Flags #
##############################

tf.flags.DEFINE_boolean(
  'calc_cond_inception_score', False,
  'If true, calculates the conditional inception score (from Multimodal Unsupervised Image-to-Image Translation).')

#############
# Constants #
#############
FLAGS = tf.flags.FLAGS

GENERATOR_LOSS_COLLECTION = image_generation.GENERATOR_LOSS_COLLECTION
DISCRIMINATOR_LOSS_COLLECTION = image_generation.DISCRIMINATOR_LOSS_COLLECTION
CUSTOM_SOURCES_INPUT_PH = 'sources_ph'
CUSTOM_TARGETS_INPUT_PH = 'targets_ph'
CUSTOM_GENERATED_SOURCES = 'custom_generated_%s_style_%s' % ('s', 'target')
CUSTOM_GENERATED_TARGETS = 'custom_generated_%s_style_%s' % ('t', 'source')

GDROP_STRENGTH_VAR_NAME = image_generation.GDROP_STRENGTH_VAR_NAME
ENCODER_DISTILL_EMBEDDING_NAME = 'embedding'  # Name of the pre-populated distillation embeddings.

ENCODER_CONTENT_VAR_SCOPE = 'encoder_content'
ENCODER_STYLE_VAR_SCOPE = 'encoder_style'
GENERATOR_VAR_SCOPE = 'generator'
DISCRIMINATOR_VAR_SCOPE_PREFIX = image_generation.DISCRIMINATOR_SCOPE
DISCRIMINATOR_VAR_SCOPE_SOURCE = DISCRIMINATOR_VAR_SCOPE_PREFIX + '_s'
DISCRIMINATOR_VAR_SCOPE_TARGET = DISCRIMINATOR_VAR_SCOPE_PREFIX + '_t'


class GanModel(image_generation.GanModel):
  #######################
  # Select the dataset  #
  #######################
  # Inherited from the parent class.

  ######################
  # Select the network #
  ######################
  def _select_network(self):
    if FLAGS.generator_network == 'pggan':
      generator_network_fn = pggan.generator
      discriminator_network_fn = pggan.discriminator
      get_noise_shape = pggan.get_noise_shape
      encoder_network_fn = pggan.encoder_before_classification
      encoder_style_network_fn = pggan.encoder
      encoder_classification_fn = pggan.encoder_classification
      # Intentionally encoder_distillation_fn is the same as classification. In the future they may be different.
      encoder_distillation_fn = pggan.encoder_classification
    else:
      raise NotImplementedError('Generator network %s is not implemented.', FLAGS.generator_network)
    return {'generator_network_fn': generator_network_fn,
            'discriminator_network_fn': discriminator_network_fn,
            'encoder_network_fn': encoder_network_fn,
            'encoder_style_network_fn': encoder_style_network_fn,
            'encoder_classification_fn': encoder_classification_fn,
            'encoder_distillation_fn': encoder_distillation_fn,
            'get_noise_shape': get_noise_shape, }


  ####################
  # Define the model #
  ####################
  @staticmethod
  def _clone_fn(networks, batch_queue, batch_names, data_batched=None, is_training=False, **kwargs):
    """Allows data parallelism by creating multiple clones of network_fn."""
    # Get data
    data_batched = super(GanModel, GanModel)._get_data_batched(batch_queue, batch_names, data_batched)
    sources = data_batched.get('a_source')
    targets = data_batched.get('b_source')
    assert sources is not None and targets is not None, 'Both source and target must be available in the dataset.'

    # Get generator/encoder/discriminator network functions.
    encoder_network_fn = networks['encoder_network_fn']
    generator_network_fn = networks['generator_network_fn']
    discriminator_network_fn = networks['discriminator_network_fn']
    encoder_style_network_fn = networks['encoder_style_network_fn'] if FLAGS.use_style_embedding else None

    # Encoder Distillation (Optional)
    if FLAGS.do_encoder_distillation:
      encoder_distillation_fn = networks['encoder_distillation_fn']
      source_distill_embed = data_batched.get('a_%s' % (ENCODER_DISTILL_EMBEDDING_NAME))
      source_distill_embed_dim = 0
      if source_distill_embed is not None:
        source_distill_embed_dim = int(source_distill_embed.shape[-1])
      else:
        tf.logging.warning('No source distill embedding found in the database.')
      target_distill_embed = data_batched.get('b_%s' % (ENCODER_DISTILL_EMBEDDING_NAME))
      target_distill_embed_dim = 0
      if target_distill_embed is not None:
        target_distill_embed_dim = int(target_distill_embed.shape[-1])
      else:
        tf.logging.warning('No target distill embedding found in the database.')
      assert (source_distill_embed_dim or target_distill_embed_dim,
              'One of source or target must have embeddings when `do_encoder_distillation` is on.')
    else:
      encoder_distillation_fn = None

    # Get input-dependent kwargs for network functions.
    global_step = kwargs['global_step']
    generator_kwargs = {'is_training': is_training, 'target_shape': targets.shape}
    discriminator_kwargs = {'is_training': is_training}
    # Define the fallback None values for style encodings.
    encoded_source_style = encoded_target_style = random_style_embed = style_embed_ph = encoded_sources_ph_style = encoded_targets_ph_style = None
    assert FLAGS.generator_network == 'pggan', 'Currently it only supports PGGAN framework.'

    sources, targets, alpha_grow = GanModel.get_growing_source_and_target(data_batched, global_step)
    (encoder_kwargs, generator_source_kwargs, generator_target_kwargs,
     discriminator_source_kwargs, discriminator_target_kwargs) = GanModel._add_pggan_kwargs(
      data_batched, sources, targets, alpha_grow, generator_kwargs, discriminator_kwargs)

    # **********
    # Encoder for input images.
    # **********
    # Source content encoder.
    with tf.variable_scope(ENCODER_CONTENT_VAR_SCOPE):
      source_encoder_kwargs = GanModel._copy_kwargs(encoder_kwargs, scope_fn_postfix='_s')
      encoded_source_content, encoded_source_content_end_points = encoder_network_fn(sources, **source_encoder_kwargs)
    # Source style encoder(optional).
    if FLAGS.use_style_embedding:
      with tf.variable_scope(ENCODER_STYLE_VAR_SCOPE):
        encoded_source_style, encoded_source_style_end_points = encoder_style_network_fn(
          sources, output_dim=FLAGS.style_embed_size, **source_encoder_kwargs)

    if FLAGS.do_encoder_distillation:
      source_encoder_distillation_kwargs = GanModel._copy_kwargs(
        source_encoder_kwargs, output_dim=source_distill_embed_dim or target_distill_embed_dim)
      with tf.variable_scope(ENCODER_CONTENT_VAR_SCOPE + '/encoder_distillation_source'):
        encoded_source_distillation, encoded_source_distillation_end_points = encoder_distillation_fn(
          encoded_source_content, **source_encoder_distillation_kwargs)

    # Target content encoder.
    with tf.variable_scope(ENCODER_CONTENT_VAR_SCOPE, reuse=tf.AUTO_REUSE):
      target_encoder_kwargs = GanModel._copy_kwargs(encoder_kwargs, scope_fn_postfix='_t')
      encoded_target_content, encoded_target_content_end_points = encoder_network_fn(targets, **target_encoder_kwargs)
    # Target style encoder(optional).
    if FLAGS.use_style_embedding:
      with tf.variable_scope(ENCODER_STYLE_VAR_SCOPE, reuse=tf.AUTO_REUSE):
        encoded_target_style, encoded_target_style_end_points = encoder_style_network_fn(
          targets, output_dim=FLAGS.style_embed_size, **target_encoder_kwargs)
        encoded_target_style = encoded_target_style

    if FLAGS.do_encoder_distillation:
      target_encoder_distillation_kwargs = GanModel._copy_kwargs(
        target_encoder_kwargs, output_dim=target_distill_embed_dim or source_distill_embed_dim)
      with tf.variable_scope(ENCODER_CONTENT_VAR_SCOPE + '/encoder_distillation_target'):
        encoded_target_distillation, encoded_target_distillation_end_points = encoder_distillation_fn(
          encoded_target_content, **target_encoder_distillation_kwargs)

    # Random style embedding for s_prime and t_prime.
    if FLAGS.use_style_embedding:
      random_style_embed = tf.random_normal(shape=encoded_target_style.shape, dtype=encoded_target_style.dtype,
                                            name='random_style_embed')

    # **********
    # Generators
    # **********
    # A note on the output image naming. The prefix goes with the output domain. So s_xxx outputs to source domain.
    # s_prime -- Uses content from target image and outputs to source domain.
    with tf.variable_scope(GENERATOR_VAR_SCOPE):
      # Notice the unet_end_points -- it corresponds with the input/output content because it's a translation task.
      generator_s_prime_kwargs = GanModel._copy_kwargs(
        generator_source_kwargs, scope_fn_postfix='_s', scope_fn_cond=random_style_embed,
        unet_end_points=encoded_target_content_end_points if FLAGS.use_unet else None)
      s_prime, s_prime_end_points = generator_network_fn(encoded_target_content, **generator_s_prime_kwargs)

    # s_cycle -- Uses everything from source image. Expects the network to act as an identity function.
    with tf.variable_scope(GENERATOR_VAR_SCOPE, reuse=True):
      generator_source_cyc_kwargs= GanModel._copy_kwargs(
        generator_source_kwargs, scope_fn_postfix='_s', scope_fn_cond=encoded_source_style,
        unet_end_points=encoded_source_content_end_points if FLAGS.use_unet else None
      )
      s_cycle, s_cycle_end_points = generator_network_fn(encoded_source_content, **generator_source_cyc_kwargs)

    # t_prime -- Uses content from source image and outputs to target domain.
    with tf.variable_scope(GENERATOR_VAR_SCOPE, reuse=tf.AUTO_REUSE):
      generator_t_prime_kwargs = GanModel._copy_kwargs(
        generator_target_kwargs, scope_fn_postfix='_t', scope_fn_cond=random_style_embed,
        unet_end_points=encoded_source_content_end_points if FLAGS.use_unet else None)
      t_prime, t_prime_end_points = generator_network_fn(encoded_source_content, **generator_t_prime_kwargs)

    # t_cycle -- Uses everything from target image. Expects the network to act as an identity function.
    with tf.variable_scope(GENERATOR_VAR_SCOPE, reuse=True):
      generator_target_cyc_kwargs = GanModel._copy_kwargs(
        generator_target_kwargs, scope_fn_postfix='_t', scope_fn_cond=encoded_target_style,
        unet_end_points=encoded_target_content_end_points if FLAGS.use_unet else None)
      t_cycle, t_cycle_end_points = generator_network_fn(encoded_target_content, **generator_target_cyc_kwargs)

    # **********
    # Encoder for generated images.
    # **********
    # Encode t_prime content and style.
    with tf.variable_scope(ENCODER_CONTENT_VAR_SCOPE, reuse=True):
      encoded_t_prime_content, encoded_t_prime_content_end_points = encoder_network_fn(t_prime, **target_encoder_kwargs)
    if FLAGS.use_style_embedding:
      with tf.variable_scope(ENCODER_STYLE_VAR_SCOPE, reuse=True):
        encoded_t_prime_style, encoded_t_prime_style_end_points = encoder_style_network_fn(
          t_prime, output_dim=FLAGS.style_embed_size, **target_encoder_kwargs)

    # Encode s_prime content and style.
    with tf.variable_scope(ENCODER_CONTENT_VAR_SCOPE, reuse=True):
      encoded_s_prime_content, encoded_s_prime_content_end_points = encoder_network_fn(s_prime, **source_encoder_kwargs)
    if FLAGS.use_style_embedding:
      with tf.variable_scope(ENCODER_STYLE_VAR_SCOPE, reuse=True):
        encoded_s_prime_style, encoded_s_prime_style_end_points = encoder_style_network_fn(
          s_prime, output_dim=FLAGS.style_embed_size, **source_encoder_kwargs)

    # Encoder distillation.
    if FLAGS.do_encoder_distillation:
      with tf.variable_scope(ENCODER_CONTENT_VAR_SCOPE + '/encoder_distillation_source', reuse=True):
        encoded_s_prime_distillation, encoded_s_prime_distillation_end_points = encoder_distillation_fn(
          encoded_s_prime_content, **source_encoder_distillation_kwargs)

      with tf.variable_scope(ENCODER_CONTENT_VAR_SCOPE + '/encoder_distillation_target', reuse=True):
        encoded_t_prime_distillation, encoded_t_prime_distillation_end_points = encoder_distillation_fn(
          encoded_t_prime_content, **target_encoder_distillation_kwargs)

    # ******
    # Placeholders for debugging, testing, and inference.
    # ******
    # Placeholder Content
    sources_ph = tf.placeholder(sources.dtype,
                                tf.TensorShape([None, sources.shape[1], sources.shape[2], sources.shape[3]]),
                                name=CUSTOM_SOURCES_INPUT_PH)
    targets_ph = tf.placeholder(targets.dtype,
                                tf.TensorShape([None, targets.shape[1], targets.shape[2], targets.shape[3]]),
                                name=CUSTOM_TARGETS_INPUT_PH)
    with tf.variable_scope(ENCODER_CONTENT_VAR_SCOPE, reuse=True):

      source_ph_encoder_kwargs = GanModel._copy_kwargs(source_encoder_kwargs, is_training=False, )
      target_ph_encoder_kwargs = GanModel._copy_kwargs(target_encoder_kwargs, is_training=False, )

      encoded_sources_ph_content, encoded_sources_ph_content_end_points = encoder_network_fn(sources_ph,
                                                                                             **source_ph_encoder_kwargs)
      encoded_targets_ph_content, encoded_targets_ph_content_end_points = encoder_network_fn(targets_ph,
                                                                                             **target_ph_encoder_kwargs)
      # generator_target_ph -- content from source and outputs to target domain. Similarly for generator_source_ph.
      generator_target_ph_kwargs = GanModel._copy_kwargs(
        generator_target_kwargs, is_training=False,
        unet_end_points=encoded_sources_ph_content_end_points if FLAGS.use_unet else None)
      generator_source_ph_kwargs = GanModel._copy_kwargs(
        generator_source_kwargs, is_training=False,
        unet_end_points=encoded_targets_ph_content_end_points if FLAGS.use_unet else None)

    # Placeholder Style
    if FLAGS.use_style_embedding:
      with tf.variable_scope(ENCODER_STYLE_VAR_SCOPE, reuse=True):
        encoded_sources_ph_style, encoded_sources_ph_style_end_points = encoder_style_network_fn(
          sources_ph, output_dim=FLAGS.style_embed_size, **source_ph_encoder_kwargs)
        encoded_targets_ph_style, encoded_targets_ph_style_end_points = encoder_style_network_fn(
          targets_ph, output_dim=FLAGS.style_embed_size, **target_ph_encoder_kwargs)
        style_embed_ph = tf.placeholder(random_style_embed.dtype, tf.TensorShape([None, random_style_embed.shape[1]]),
                                        name='style_embed_ph')
    # Placeholder Generator
    custom_generated_dict = {}
    for source_or_target in ['s', 't']:
      # Generates images with four different style embeddings:
      # - A fixed random style embedding,
      # - the style embedding from the encoded source placeholder,
      # - from the encoded target placeholder,
      # - and directly from the style embedding placeholder.
      for name, conditional_layer in [('custom_generated_%s_style_rand' % (source_or_target), random_style_embed),
                                      ('custom_generated_%s_style_%s' % (source_or_target, 'source'),
                                       encoded_sources_ph_style),
                                      ('custom_generated_%s_style_%s' % (source_or_target, 'target'),
                                       encoded_targets_ph_style),
                                      ('custom_generated_%s_style_ph' % (source_or_target), style_embed_ph), ]:
        generator_loop_ph_kwargs = copy.copy(
          generator_source_ph_kwargs if source_or_target == 's' else generator_target_ph_kwargs)

        generator_loop_ph_kwargs['arg_scope_fn'] = functools.partial(generator_loop_ph_kwargs.get(
          'arg_scope_fn', GanModel._get_generator_arg_scope_fn()),
          conditional_layer_var_scope_postfix='_%s' % source_or_target, conditional_layer=conditional_layer)
        with tf.variable_scope(GENERATOR_VAR_SCOPE, reuse=True):
          custom_loop_generated, _ = generator_network_fn(
            encoded_targets_ph_content if source_or_target == 's' else encoded_sources_ph_content,
            **generator_loop_ph_kwargs)
          custom_generated_dict[name] = custom_loop_generated

    # Now out of all the variable scopes, define identity tensors for all custom outputs.
    # This is mainly for convenience at inference time.
    for name in custom_generated_dict:
      custom_generated_dict[name] = tf.identity(custom_generated_dict[name], name=name)

    # ******
    # Discriminators
    # ******
    with tf.variable_scope(DISCRIMINATOR_VAR_SCOPE_SOURCE, reuse=False):
      real_source_prediction, real_source_end_points = discriminator_network_fn(sources, **discriminator_source_kwargs)
    with tf.variable_scope(DISCRIMINATOR_VAR_SCOPE_SOURCE, reuse=True):
      dis_s_prime_prediction, dis_s_prime_end_points = discriminator_network_fn(s_prime, **discriminator_source_kwargs)
    with tf.variable_scope(DISCRIMINATOR_VAR_SCOPE_SOURCE, reuse=True):
      dis_s_cycle_prediction, dis_s_cycle_end_points = discriminator_network_fn(s_cycle, **discriminator_source_kwargs)
    with tf.variable_scope(DISCRIMINATOR_VAR_SCOPE_TARGET, reuse=False):
      real_target_prediction, real_target_end_points = discriminator_network_fn(targets, **discriminator_target_kwargs)
    with tf.variable_scope(DISCRIMINATOR_VAR_SCOPE_TARGET, reuse=True):
      dis_t_prime_prediction, dis_t_prime_end_points = discriminator_network_fn(t_prime, **discriminator_target_kwargs)
    with tf.variable_scope(DISCRIMINATOR_VAR_SCOPE_TARGET, reuse=True):
      dis_t_cycle_prediction, dis_t_cycle_end_points = discriminator_network_fn(t_cycle, **discriminator_target_kwargs)

    # Combine the end points. See util_misc.combine_dicts for details.
    end_points_dict = {
      'encoded_source_content': encoded_source_content_end_points,
      'encoded_target_content': encoded_target_content_end_points,
      's_prime': s_prime_end_points,
      't_prime': t_prime_end_points,
      's_cycle': s_cycle_end_points,
      't_cycle': t_cycle_end_points,

      'discriminator_real_s': real_source_end_points,
      'discriminator_s_prime': dis_s_prime_end_points,
      'discriminator_s_cycle': dis_s_cycle_end_points,
      'discriminator_real_t': real_target_end_points,
      'discriminator_t_prime': dis_t_prime_end_points,
      'discriminator_t_cycle': dis_t_cycle_end_points,

      'encoded_s_prime_content': encoded_s_prime_content_end_points,
      'encoded_t_prime_content': encoded_t_prime_content_end_points,
    }
    if FLAGS.use_style_embedding:
      end_points_dict.update({
        'encoded_source_style': encoded_source_style_end_points,
        'encoded_target_style': encoded_target_style_end_points,
        'encoded_s_prime_style': encoded_s_prime_style_end_points,
        'encoded_t_prime_style': encoded_t_prime_style_end_points,
      })

    if FLAGS.do_encoder_distillation:
      if source_distill_embed_dim:
        end_points_dict.update({
          'encoded_source_distillation': encoded_source_distillation_end_points,
          'encoded_t_prime_distillation': encoded_t_prime_distillation_end_points,
        })
      if target_distill_embed_dim:
        end_points_dict.update({
          'encoded_target_distillation': encoded_target_distillation_end_points,
          'encoded_s_prime_distillation': encoded_s_prime_distillation_end_points,
        })

    end_points = util_misc.combine_dicts(end_points_dict)
    end_points.update({
      'sources': sources,
      'targets': targets,
      CUSTOM_SOURCES_INPUT_PH: sources_ph,
      CUSTOM_TARGETS_INPUT_PH: targets_ph,
    })
    if FLAGS.use_style_embedding:
      end_points.update({
        'random_style_embed': random_style_embed,
        'style_embed_ph': style_embed_ph,
      })
    end_points.update(custom_generated_dict)

    #############################
    # Specify the loss function #
    #############################
    # Needed for Dragan loss.
    discriminator_network_fns = {
      'discriminator_t': functools.partial(discriminator_network_fn, **discriminator_target_kwargs),
      'discriminator_s': functools.partial(discriminator_network_fn, **discriminator_source_kwargs)
    }
    GanModel.add_loss(data_batched, end_points, discriminator_network_fns)
    return end_points


  ####################
  # Define the loss  #
  ####################
  @staticmethod
  def add_loss(data_batched, end_points, discriminator_network_fn=None):
    for domain in ['s', 't']:
      # Convenience variables to get desired outputs from the `end_points`.
      domain_full_str = 'source' if domain is 's' else 'target'
      opposite = 't' if domain is 's' else 's'
      dataset = 'a' if domain == 's' else 'b'

      expected_original = end_points[domain_full_str + 's']
      prime_generated = end_points['%s_prime_output' % (domain)]
      cycle_generated = end_points['%s_cycle_output' % (domain)]

      # For regenerated cycle images, we use l1 as the regeneration loss.
      tf.losses.absolute_difference(expected_original, cycle_generated, weights=FLAGS.l_cyc_weight,
                                    scope='l_cyc_%s' % domain, loss_collection=GENERATOR_LOSS_COLLECTION)
      if FLAGS.train_image_size >= 64 and FLAGS.do_l_cyc_gan:  # ">= 64" is for faster training.
        # And the GAN loss because only using l1 results in artifacts, blurry output, and dull color.
        GanModel.add_gan_loss(end_points['discriminator_%s_cycle_prediction' % (domain)],
                              end_points['discriminator_real_%s_prediction' % (domain)],
                              cycle_generated, expected_original,
                              discriminator_network_fn['discriminator_%s' % (domain)],
                              discriminator_var_scope='discriminator_%s' % (domain),
                              only_real_fake_loss=True,  # Do no add things like dragon loss on real image twice.
                              name_postfix='_cycle')

      # For prime, we use the GAN loss.
      GanModel.add_gan_loss(end_points['discriminator_%s_prime_prediction' % (domain)],
                            end_points['discriminator_real_%s_prediction' % (domain)],
                            prime_generated, expected_original,
                            discriminator_network_fn['discriminator_%s' % (domain)],
                            discriminator_var_scope='discriminator_%s' % (domain),
                            name_postfix='_prime')

      # The encoded content loss = abs(E(x) - E(G(E(x))))
      if FLAGS.l_content_weight:
        for content_or_style in ['content', 'style']:
          layer_name = None
          encoded_original_layer = None
          encoded_prime_layer = None
          if content_or_style == 'content':
            layer_name = 'before_classification'
            encoded_original_layer = end_points['encoded_%s_%s_%s' % (domain_full_str, content_or_style, layer_name)]
            # Note: s_prime = Source Style + Target Content! Thus encoded_source = encoded_t_prime and vice versa.
            encoded_prime_layer = end_points['encoded_%s_prime_%s_%s' % (opposite, content_or_style, layer_name)]
          elif FLAGS.use_style_embedding:
            layer_names = ['prediction']
            for layer_name in layer_names:
              encoded_original_layer = end_points['random_style_embed']
              encoded_prime_layer = end_points['encoded_%s_prime_%s_%s' % (domain, content_or_style, layer_name)]

          if encoded_original_layer is not None:
            tf.losses.absolute_difference(encoded_original_layer, encoded_prime_layer, weights=FLAGS.l_content_weight,
                                          scope='l_%s_%s_%s' % (
                                            domain_full_str, content_or_style, layer_name),
                                          loss_collection=GENERATOR_LOSS_COLLECTION)

      # The distillation loss = cosine_distance(E(x), distill_embedding)
      if FLAGS.do_encoder_distillation and FLAGS.train_image_size >= FLAGS.distillation_start_hw:
        expected = data_batched.get('%s_%s' % (dataset, ENCODER_DISTILL_EMBEDDING_NAME))
        if expected is not None:
          for prefix in [domain_full_str, opposite + '_prime']:
            embedding_name = 'encoded_%s_distillation_prediction' % prefix
            embedding = end_points.get(embedding_name)
            if embedding is not None:
              expected_normalized = tf.nn.l2_normalize(expected, axis=-1)
              embedding_normalized = tf.nn.l2_normalize(embedding, axis=-1)
              tf.losses.cosine_distance(expected_normalized, embedding_normalized, weights=FLAGS.distillation_weight,
                                        axis=-1, scope='l_%s_distillation' % (prefix),
                                        loss_collection=GENERATOR_LOSS_COLLECTION)
            else:
              raise AssertionError('Embedding %s does not exist in end_points.' % embedding_name)

  ################
  # Optimization #
  ################
  def _get_generator_variable_scopes(self):
    return [ENCODER_CONTENT_VAR_SCOPE, ENCODER_STYLE_VAR_SCOPE, GENERATOR_VAR_SCOPE]

  def _get_generator_variables_to_train(self):
    if not FLAGS.use_style_embedding:
      assert not self._get_variables_to_train(trainable_scopes=ENCODER_STYLE_VAR_SCOPE)
    return super(GanModel, self)._get_generator_variables_to_train()

  def _check_trainable_vars(self, generator_variables_to_train, discriminator_variables_to_train):
    # Note here that generator_variables_to_train includes encoder variables.
    assert len(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)) == (len(generator_variables_to_train) +
                                                                         len(discriminator_variables_to_train)), \
            'Discriminator and generator variables does not add up to form all trainable variables. Suspicious...'

  # Inherits from the parent class.
  # def _add_optimization(self, clones, optimizer, summaries, update_ops, global_step):

  #################
  # Add summaries #
  #################
  @staticmethod
  def _add_end_point_summaries(end_points, summaries):
    # Exclude the placeholder-dependent end points.
    excluded_end_points = {
      'custom_generated_t_style_rand',
      'custom_generated_t_style_source',
      'custom_generated_t_style_target',
      'custom_generated_t_style_ph',
      'custom_generated_s_style_rand',
      'custom_generated_s_style_source',
      'custom_generated_s_style_target',
      'custom_generated_s_style_ph',
      CUSTOM_SOURCES_INPUT_PH,
      CUSTOM_TARGETS_INPUT_PH,
      'style_embed_ph',
    }
    not_excluded_end_points = copy.copy(end_points)
    for end_point in end_points:
      if end_point in excluded_end_points:
        del not_excluded_end_points[end_point]
    super(GanModel, GanModel)._add_end_point_summaries(not_excluded_end_points, summaries)


  def _add_image_summaries(self, end_points, _):
    # Add summaries for images, if there are any.
    for end_point_name in ['sources', 'targets', 's_prime_output', 't_prime_output', 's_cycle_output',
                           't_cycle_output']:
      if (end_point_name in end_points and len(end_points[end_point_name].shape) == 4):
        self._add_one_image_summary(end_point_name, self._post_process_image(end_points[end_point_name]))


  ###################################################
  # Extra function to run after each training step. #
  ###################################################

  @staticmethod
  def get_fixed_sources(np_path):
    """Returns a fixed set of images read from the npy file, which are repeated `batch_size` times on the y axis."""
    ret = np.load(np_path)
    ret = np.array(
      [scipy.misc.imresize(ret[i], (FLAGS.train_image_size, FLAGS.train_image_size)) for i in range(ret.shape[0])])
    ret = ret.astype(np.float32) / 255.0
    ret = np.expand_dims(ret, 1)
    ret = np.repeat(ret, repeats=FLAGS.batch_size, axis=1)
    return ret

  @staticmethod
  def get_fixed_rand_style_embed(seed=31415, ):
    """The style embedding at location i in each batch is a linear interpolation of two random embeddings"""
    np.random.seed(seed)
    ret = []
    point_a = np.random.normal(size=(FLAGS.style_embed_size,))
    point_b = np.random.normal(size=(FLAGS.style_embed_size,))
    num_points = FLAGS.style_embed_size
    for i in range(FLAGS.batch_size):
      even_interpolation = np.array(
        [point_a * float(i) / float(num_points) + point_b * (1.0 - i) / float(num_points) for x in range(num_points)])
      ret.append(even_interpolation)
    return np.transpose(ret, (1, 0, 2))

  @staticmethod
  def do_extra_train_step(session, end_points, global_step):
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
      if 'sources' not in end_points:
        raise NotImplementedError('It assumes we have `sources` in the `end_points`.')

      run_list = run_list + [end_points['sources'], end_points['targets'],
                             end_points['s_prime_output'], end_points['t_prime_output'],
                             end_points['s_cycle_output'], end_points['t_cycle_output'],
                             ]
      out_list = out_list + [os.path.join(image_dir, '%d_source.png' % (current_step)),
                             os.path.join(image_dir, '%d_target.png' % (current_step)),
                             os.path.join(image_dir, '%d_s_prime.png' % (current_step)),
                             os.path.join(image_dir, '%d_t_prime.png' % (current_step)),
                             os.path.join(image_dir, '%d_s_cycle.png' % (current_step)),
                             os.path.join(image_dir, '%d_t_cycle.png' % (current_step)),
                             ]

      # Get a fixed set of images and translate them to the target domain.
      try:
        custom_sources = GanModel.get_fixed_sources(os.path.join(FLAGS.dataset_dir, FLAGS.custom_sources_np_path))
        run_list = run_list + [end_points[CUSTOM_SOURCES_INPUT_PH],
                               end_points['custom_generated_t_style_rand'], ]
        out_list = out_list + [os.path.join(image_dir, '%d_sources_ph.png' % (current_step)),
                               os.path.join(image_dir, '%d_custom_t_style_rand.png' % (current_step)), ]

        feed_dict_per_hw = [{end_points[CUSTOM_SOURCES_INPUT_PH]: custom_sources[i], } for i in range(FLAGS.log_image_n_per_hw)]
        if FLAGS.use_style_embedding:
          custom_style_embed = GanModel.get_fixed_rand_style_embed()
          feed_dict_per_hw = [
            {end_points['sources_ph']: custom_sources[i], end_points['style_embed_ph']: custom_style_embed[i]} for i
            in range(FLAGS.log_image_n_per_hw)]
          run_list = run_list + [end_points['custom_generated_t_style_ph'],
                                 end_points['custom_generated_t_style_source']]
          out_list = out_list + [os.path.join(image_dir, '%d_custom_t_style_roll.png' % (current_step)),
                                 os.path.join(image_dir, '%d_custom_t_style_source.png' % (current_step)), ]
      except IOError as e:
        feed_dict_per_hw = None
        tf.logging.log_every_n(tf.logging.WARN ,'IOError %s. Not outputting custom generated images.' %(e), 100)

      image_list = GanModel.do_extra_train_step_aux(session, run_list=run_list, out_list=out_list,
                                                    feed_dict_per_hw=feed_dict_per_hw)

      def image_to_rgb(img):
        if img.shape[-1] != 3:
          # Convert the image into one channel by summing all channels together
          img = np.sum(img, axis=-1, keepdims=True)
          img = np.repeat(img, 3, axis=-1)
        return img

      # Now combine the source and t_prime images by stacking them in the y axis.
      for source_images, prime_images, name in [(image_list[0], image_list[3], 'target_s_prime'),
                                                (image_list[1], image_list[2], 'source_t_prime')]:
        source_t_prime_combined = []

        for i in range(len(source_images)):
          source_t_prime_combined.append(image_to_rgb(source_images[i]))
          source_t_prime_combined.append(image_to_rgb(prime_images[i]))
        source_t_prime_combined = np.concatenate(source_t_prime_combined, axis=2)
        source_t_prime_combined = np.reshape(source_t_prime_combined, (
          source_t_prime_combined.shape[0] * source_t_prime_combined.shape[1], source_t_prime_combined.shape[2],
          source_t_prime_combined.shape[3]))
        util_io.save_float_image(os.path.join(image_dir, '%d_%s.png' % (current_step, name)), source_t_prime_combined)
    if FLAGS.eval_every_n_iter_in_training and current_step % FLAGS.eval_every_n_iter_in_training == 0:
      GanModel._calc_swd(session, end_points, current_step=current_step, get_swd_real_fake=GanModel._get_swd_real_fake)
  ########
  # Eval #
  ########
  @staticmethod
  def _define_outputs(end_points, data_batched):
    if FLAGS.output_single_file:
      # Outputs latent embedding of one dataset to a csv file.
      do_a = (FLAGS.dataset_dir != '')
      if do_a:
        return [
          ('sources_filename', False, data_batched.get('a_filename') if do_a else data_batched.get('b_filename')),
          ('encoded_sources', False, end_points['encoded_source_content_before_classification'] if do_a else end_points[
            'encoded_target_content_before_classification']),
        ]
      else:
        return [
          ('sources_filename', False, data_batched.get('a_filename') if do_a else data_batched.get('b_filename')),
          ('encoded_sources', False, end_points['encoded_source_content_before_classification'] if do_a else end_points[
            'encoded_target_content_before_classification']),
          ('sources', True, data_batched.get('b_source')),
        ]
    else:
      raise NotImplementedError

  @staticmethod
  def _write_outputs(output_results, output_ops, ):
    # Outputs latent embedding of one dataset to a csv file.
    save_dir = FLAGS.eval_dir
    if FLAGS.output_single_file:
      single_file_name = os.path.join(save_dir, FLAGS.output_single_file_name)
      # Flatten the numpy arrays.
      output_results[1] = output_results[1].reshape([output_results[1].shape[0], -1])
      if len(output_results) == 3:
        if output_results[2].dtype != np.float32:
          output_results[2] = np.array([base64.b64encode(encoded_image) for encoded_image in output_results[2]])
        output_results[2] = output_results[2].reshape([output_results[2].shape[0], -1])
      output_results = [item.tolist() for item in output_results]

      with open(single_file_name, 'ab') as f:
        writer = csv.writer(f)
        # ugly implementation but works for now.
        if len(output_results) == 2:
          writer.writerows([[output_results[0][i]] + output_results[1][i]
                            for i in range(len(output_results[0]))])
        elif len(output_results) == 3:
          writer.writerows([[output_results[0][i]] + output_results[1][i] + output_results[2][i]
                            for i in range(len(output_results[0]))])
        else:
          raise NotImplementedError

  def get_items_to_encode(self, end_points, data_batched):
    """Outputs a list with format (name, is_image, tensor)"""
    items_to_encode = []
    end_points_to_encode = ['sources', 'targets', 't_prime_output', 'discriminator_t_prime_prediction',
                            'discriminator_real_t_prediction',
                            'encoded_%s_classification_prediction' % ('source'),
                            'encoded_%s_classification_prediction' % ('target'), ]
    for end_point in end_points_to_encode:
      if end_point in end_points:
        end_point_tensor = end_points.get(end_point)
        assert end_point_tensor is not None
        is_image = self._maybe_is_image(end_point_tensor)
        if is_image:
          items_to_encode.append((end_point, is_image, self._post_process_image(end_point_tensor)))
        else:
          items_to_encode.append((end_point, is_image, end_point_tensor))
    items_to_encode.extend([('b_label_text', False, data_batched['b_label_text'])])

    targets = end_points['targets']
    t_prime = end_points['t_prime_output']
    t_prime_prediction = end_points['discriminator_t_prime_prediction']
    real_t_prediction = end_points['discriminator_real_t_prediction']

    best_t_prime_i = tf.argmax(tf.squeeze(t_prime_prediction, axis=1))
    worst_real_target_i = tf.argmin(tf.squeeze(real_t_prediction, axis=1))

    items_to_encode.append(('best_generated_target', True, self._post_process_image(t_prime[best_t_prime_i])))
    items_to_encode.append(('worst_real_target', True, self._post_process_image(targets[worst_real_target_i])))
    return items_to_encode

  @staticmethod
  def _get_swd_real_fake(end_points):
    return end_points['targets'], end_points['t_prime_output']

  def _do_extra_eval_actions(self, session, extra_eval):
    if FLAGS.calc_inception_score:
      predictions, end_points, saver = extra_eval
      self.calc_inception_score(predictions, saver, FLAGS.incep_classifier_path, session, )
    elif FLAGS.calc_swd:
     (end_points, data_batched) = extra_eval
     self._calc_swd(session, end_points, data_batched, get_swd_real_fake=self._get_swd_real_fake)

  ##########
  # Export #
  ##########

  @staticmethod
  def _build_signature_def_map(end_points, data_batched):
    sources = tf.saved_model.utils.build_tensor_info(
      end_points[CUSTOM_SOURCES_INPUT_PH])
    targets = tf.saved_model.utils.build_tensor_info(
      end_points[CUSTOM_TARGETS_INPUT_PH])
    output_sources = tf.saved_model.utils.build_tensor_info(
      end_points[CUSTOM_GENERATED_SOURCES])
    output_targets = tf.saved_model.utils.build_tensor_info(
      end_points[CUSTOM_GENERATED_TARGETS])

    # By default the input is source domain and the output is target domain.
    # Change the following to go the other way around for target->source.
    domain_transfer_signature = (
      tf.saved_model.signature_def_utils.build_signature_def(
        inputs={
          tf.saved_model.signature_constants.PREDICT_INPUTS:
            sources,
        },
        outputs={
          tf.saved_model.signature_constants.PREDICT_OUTPUTS:
            output_targets,
        },
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))
    ret = {
      tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
        domain_transfer_signature,
    }
    return ret

  @staticmethod
  def _build_assets_collection(end_points, data_batched):
    if os.path.exists(FLAGS.tags_id_lookup_file):
      asset_file = tf.constant(FLAGS.tags_id_lookup_file, name="tags_id_lookup_file")
      tf.add_to_collection(tf.GraphKeys.ASSET_FILEPATHS, asset_file)
    assets_collection = tf.get_collection(tf.GraphKeys.ASSET_FILEPATHS)
    return assets_collection


  #############################
  # PGGAN specific functions. #
  #############################

  @staticmethod
  def _get_generator_arg_scope_fn():
    """Wrapper function specifying norm type for the generator."""
    return functools.partial(
      pggan.conditional_progressive_gan_generator_arg_scope, norm_type=FLAGS.generator_norm_type
    )

  @staticmethod
  def get_growing_source_and_target(data_batched, global_step):
    # TODO: do I need to grow source and target?
    sources = data_batched.get('a_source')
    targets = data_batched.get('b_source')
    if FLAGS.is_growing:
      with tf.variable_scope('alpha_grow'):
        alpha_grow = tf.cast(global_step - FLAGS.grow_start_number_of_steps, targets.dtype) / (
            FLAGS.max_number_of_steps - FLAGS.grow_start_number_of_steps)
        sources = GanModel.get_growing_image(sources, alpha_grow, name_postfix='sources')
        targets = GanModel.get_growing_image(targets, alpha_grow, name_postfix='targets')
    else:
      alpha_grow = 0.0
    return sources, targets, alpha_grow

  @staticmethod
  def _add_pggan_kwargs(data_batched, sources, targets, alpha_grow, generator_kwargs, discriminator_kwargs):
    """Adds pggan related function parameters to generator, encoder, and discriminator kwargs."""

    additional_kwargs = {'is_growing': FLAGS.is_growing, 'alpha_grow': alpha_grow,
                         'do_self_attention': FLAGS.do_self_attention, 'self_attention_hw': FLAGS.self_attention_hw}
    generator_kwargs.update(**additional_kwargs)
    generator_kwargs['do_pixel_norm'] = FLAGS.do_pixel_norm
    assert targets.dtype == sources.dtype, 'Source and target dtype should be the same.'
    generator_kwargs['dtype'] = targets.dtype if targets is not None else None

    generator_source_kwargs = copy.copy(generator_kwargs)
    generator_source_kwargs['target_shape'] = sources.shape
    generator_target_kwargs = copy.copy(generator_kwargs)
    generator_target_kwargs['target_shape'] = targets.shape

    encoder_kwargs = copy.copy(generator_kwargs)

    discriminator_kwargs.update(**additional_kwargs)
    if FLAGS.use_gdrop:
      discriminator_kwargs[GDROP_STRENGTH_VAR_NAME] = slim.model_variable(GDROP_STRENGTH_VAR_NAME, shape=[],
                                                                          dtype=targets.dtype,
                                                                          initializer=tf.zeros_initializer,
                                                                          trainable=False)
    else:
      discriminator_kwargs['do_dgrop'] = False

    discriminator_source_kwargs = copy.copy(discriminator_kwargs)
    discriminator_target_kwargs = copy.copy(discriminator_kwargs)
    if FLAGS.use_conditional_labels:
      raise NotImplementedError('TwinGAN does not support `use_conditional_labels` flag yet.')
    return (encoder_kwargs, generator_source_kwargs, generator_target_kwargs,
            discriminator_source_kwargs, discriminator_target_kwargs)

  @staticmethod
  def _copy_kwargs(old_kwargs, scope_fn_postfix='', scope_fn_cond=None, **kwargs):
    """Convenience function to create a new set of kwargs with some fields reset."""
    new_kwargs = copy.copy(old_kwargs)
    arg_scope_fn_kwargs = {}
    if scope_fn_postfix or scope_fn_cond is not None:
      if scope_fn_postfix:
        arg_scope_fn_kwargs['conditional_layer_var_scope_postfix']=scope_fn_postfix
      if scope_fn_cond is not None:
        arg_scope_fn_kwargs['conditional_layer']=scope_fn_cond
      new_kwargs['arg_scope_fn'] = functools.partial(
        new_kwargs.get('arg_scope_fn', GanModel._get_generator_arg_scope_fn()), **arg_scope_fn_kwargs)

    for key in kwargs:
      new_kwargs[key] = kwargs[key]
    return new_kwargs

  def main(self):
    super(GanModel, self).main()


def main(_):
  print('FLAGS.train_encoder is %s' % (str(FLAGS.train_encoder)))
  model = GanModel()
  model.main()


if __name__ == '__main__':
  tf.app.run()
