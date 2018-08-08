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

"""The parent class of all trainers. Taken from slim object detection."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools
import math
import os
import time
import urllib

import tensorflow as tf
import tensorflow.contrib.slim as slim

import util_io
import util_misc
from datasets import dataset_factory
from deployment import model_deploy
from preprocessing import preprocessing_factory

#########################
# Overall Trainer Flags #
#########################
tf.flags.DEFINE_string(
  'train_dir', 'checkpoints/',
  'Directory where checkpoints and event logs are written to.')

tf.flags.DEFINE_boolean(
  'is_training', False,
  'Train when set to true. Do eval otherwise.')

tf.flags.DEFINE_boolean(
  'clone_on_cpu', False,
  'Use CPUs to deploy clones. Default is to use GPU.')

tf.flags.DEFINE_float(
  'moving_average_decay', None,
  'The decay to use for using moving average of variables in SyncReplicasOptimizer.'
  'If left as None, then moving averages are not used.')

tf.flags.DEFINE_integer(
  'num_readers', 4,
  'The number of parallel readers that read data from the dataset.')

tf.flags.DEFINE_integer(
  'num_preprocessing_threads', 4,
  'The number of threads used to create the batches.')

tf.flags.DEFINE_integer(
  'log_every_n_steps', 10,
  'The frequency with which logs are print.')

tf.flags.DEFINE_integer(
  'save_summaries_secs', 120,
  'The frequency with which summaries are saved, in seconds.')

tf.flags.DEFINE_integer(
  'save_interval_secs', 600,
  'The frequency with which the model is saved, in seconds.')

###################
# Multi-GPU Flags #
###################
# Note that multi-gpu setting is taken from slim without testing. You may have to modify the code to get it working.
tf.flags.DEFINE_string(
  'master', '',
  'The address of the TensorFlow master to use.')

tf.flags.DEFINE_integer('num_clones', 1,
                        'Number of model clones to deploy. Increase beyond 1 for multi-gpu training.')

tf.flags.DEFINE_integer('worker_replicas', 1,
                        'Number of worker replicas. Increase beyond 1 for multi-gpu training.')

tf.flags.DEFINE_bool(
  'sync_replicas', False,
  'Whether or not to synchronize the replicas during multi-gpu training.')

tf.flags.DEFINE_integer(
  'replicas_to_aggregate', 1,
  'The Number of gradients to collect before updating params.')

tf.flags.DEFINE_integer(
  'num_ps_tasks', 0,
  'The number of parameter servers. If the value is 0, then the parameters '
  'are handled locally by the worker.')

tf.flags.DEFINE_integer(
  'task', 0,
  'Task id of the replica running the training.')

##########################
# Network and loss Flags #
##########################

# To be defined by child classes.

######################
# Optimization Flags #
######################

tf.flags.DEFINE_float(
  'weight_decay', 0.00000004, 'The weight decay on the model weights.')

tf.flags.DEFINE_string(
  'optimizer', 'adam',
  'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
  '"ftrl", "momentum", "sgd" or "rmsprop".')

tf.flags.DEFINE_float(
  'adadelta_rho', 0.95,
  'The decay rate for adadelta.')

tf.flags.DEFINE_float(
  'adagrad_initial_accumulator_value', 0.1,
  'Starting value for the AdaGrad accumulators.')

tf.flags.DEFINE_float(
  'adam_beta1', 0.5,
  'The exponential decay rate for the 1st moment estimates.')

tf.flags.DEFINE_float(
  'adam_beta2', 0.99,
  'The exponential decay rate for the 2nd moment estimates.')

tf.flags.DEFINE_float('opt_epsilon', 1e-8,
                      'Epsilon term for the optimizer.')

tf.flags.DEFINE_float('ftrl_learning_rate_power', -0.5,
                      'The learning rate power.')

tf.flags.DEFINE_float(
  'ftrl_initial_accumulator_value', 0.1,
  'Starting value for the FTRL accumulators.')

tf.flags.DEFINE_float(
  'ftrl_l1', 0.0, 'The FTRL l1 regularization strength.')

tf.flags.DEFINE_float(
  'ftrl_l2', 0.0, 'The FTRL l2 regularization strength.')

tf.flags.DEFINE_float(
  'momentum', 0.9,
  'The momentum for the MomentumOptimizer and RMSPropOptimizer.')

tf.flags.DEFINE_float('rmsprop_momentum', 0.9, 'Momentum.')

tf.flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')

#######################
# Learning Rate Flags #
#######################

tf.flags.DEFINE_float('learning_rate', 0.005, 'Initial learning rate.')

tf.flags.DEFINE_string(
  'learning_rate_decay_type',
  'exponential',
  'Specifies how the learning rate is decayed. One of "fixed", "exponential",'
  ' or "polynomial"')

tf.flags.DEFINE_float(
  'end_learning_rate', 0.0001,
  'The minimal end learning rate used by a polynomial decay learning rate.')

tf.flags.DEFINE_float(
  'learning_rate_decay_factor', 0.94, 'Learning rate decay factor.')

tf.flags.DEFINE_float(
  'num_epochs_per_decay', 2.0,
  'Number of epochs after which learning rate decays.')

#################
# Dataset Flags #
#################

tf.flags.DEFINE_string(
  'dataset_name', '', 'The name of the dataset to load.')

tf.flags.DEFINE_string(
  'dataset_split_name', 'train', 'The name of the train/test split.')

tf.flags.DEFINE_string(
  'dataset_dir', '', 'The directory where the dataset files are stored. e.g. A folder containing tfrecords.')

tf.flags.DEFINE_string(
  'unpaired_target_dataset_name', '',
  '(Optional) The name of the target dataset to load. This means that the source dataset with name '
  '`FLAGS.dataset_name` will be in a separate batch and there is no pairing between source and target.')

tf.flags.DEFINE_string(
  'unpaired_target_dataset_dir', '',
  'The directory where the target dataset files are stored.')

tf.flags.DEFINE_integer(
  'labels_offset', 0,
  'An offset for the labels in the dataset. This flag is primarily used to '
  'evaluate the VGG and ResNet architectures which do not use a background '
  'class for the ImageNet dataset.')

tf.flags.DEFINE_integer(
  'batch_size', 32, 'The number of samples in each batch.')

tf.flags.DEFINE_integer(
  'train_image_size', None, 'Train image size')

tf.flags.DEFINE_boolean(
  'do_random_cropping', False, 'If true, randomly crop the input images to 0.8 times of its original size for data '
                               'augmentation. Note that this is done BEFORE resizing in `resize_mode`.')

tf.flags.DEFINE_integer('max_number_of_steps', None,
                        'The maximum number of training steps.')

tf.flags.DEFINE_string(
  'preprocessing_name', None, 'The name of the preprocessing to use. If left '
                              'as `None`, then the generator_network flag is used.')
tf.flags.DEFINE_boolean(
  'preprocess_on_clone', False,
  'If true, preprocess is allocated on the first clone device (usually a gpu). Otherwise it is allocated on the input'
  'device.')

tf.flags.DEFINE_string(
  'color_space', 'rgb', 'The color space that the image will be processsed in. Currently supports rgb and yiq')

tf.flags.DEFINE_boolean(
  'subtract_mean', False, 'If true, subtract the mean of the images. Used for vgg. You may need to change the mean'
                          'values to that of your dataset.')

tf.flags.DEFINE_string(
  'resize_mode', 'PAD', 'One of PAD, CROP, RESHAPE, or NONE as specified in preprocessing_util.py.')

##############################
# Fine-Tuning and Eval Flags #
##############################

tf.flags.DEFINE_string(
  'checkpoint_path', None,
  'The path to a checkpoint from which to fine-tune.')

tf.flags.DEFINE_string(
  'eval_dir', None,
  'The path write eval results.')
tf.flags.DEFINE_boolean(
  'do_custom_eval', False,
  'If true, instead of outputting the losses on the test set, run custom evaluation code.')
tf.flags.DEFINE_boolean(
  'do_eval_debug', False,
  'If true, instead of normal evaluation, output debug information.')

tf.flags.DEFINE_boolean(
  'do_output', False,
  'If true, write model outputs (No eval and no training).')

tf.flags.DEFINE_boolean(
  'do_export', False,
  'If true, freeze and export the model for future inference (No eval and no training).')
tf.flags.DEFINE_string(
  'export_path', 'export/default/',
  'The path to export the model.')

tf.flags.DEFINE_string(
  'checkpoint_exclude_scopes', None,
  'Comma-separated list of scopes of variables to exclude when restoring '
  'from a checkpoint.')

tf.flags.DEFINE_string(
  'trainable_scopes', None,
  'Comma-separated list of scopes to filter the set of variables to train.'
  'By default, None would train all the variables.')

tf.flags.DEFINE_boolean(
  'ignore_missing_vars', False,
  'When restoring a checkpoint would ignore missing variables.')

############################
# Mixed Precision Training #
############################
tf.flags.DEFINE_string(
  'dataset_dtype', 'float32',
  'The dtype for dataset. All applicable data and most operations will be using this dtype.')
tf.flags.DEFINE_string(
  'variable_dtype', 'float32',
  'The dtype for the variables. If it is different from dataset_dtype, mixed_precision training will be used. '
  'See Nvidia\'s blog on this for more details.')
tf.flags.DEFINE_float(
  'mix_precision_loss_scale', 128.0,
  'The loss scale to scale the gradients during computation. Used for mixed precision training.')

FLAGS = tf.flags.FLAGS
FLAGS_FILE_NAME = 'flags.txt'

# Used to output tensor values to webpages, console, etc.
OutputTensor = collections.namedtuple(
  'OutputTensor', [
    'name', 'is_image', 'tensor'
  ])


class GeneralModel(object):
  #######################
  # Select the dataset  #
  #######################
  def _select_dataset(self):
    """Selects and returns the dataset used for training/eval.

    :return: One ore more slim.dataset.Dataset.
    """
    dataset = dataset_factory.get_dataset(
      FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)
    assert dataset.num_samples >= FLAGS.batch_size
    self.num_samples = dataset.num_samples
    if hasattr(dataset, 'num_classes'):
      self.num_classes = dataset.num_classes
    else:
      self.num_classes = 0
    tf.logging.info('dataset %s number of classes:%d ,number of samples:%d'
                    % (FLAGS.dataset_name, self.num_classes, self.num_samples))
    return dataset

  ######################
  # Select the network #
  ######################
  def _select_network(self):
    raise NotImplementedError('To be implemented by child class.')

  ####################
  # Define the model #
  ####################
  @staticmethod
  def _clone_fn(networks, batch_queue, batch_names, data_batched=None, is_training=False, **kwargs):
    """Allows data parallelism by creating multiple clones of network_fn."""
    raise NotImplementedError('This should be implemented by the child class.')

  ####################
  # Define the loss  #
  ####################
  @staticmethod
  def add_loss(data_batched, end_points, discriminator_network_fn=None):
    raise NotImplementedError('This should be implemented by the child class.')

  def _add_optimization(self, clones, optimizer, summaries, update_ops, global_step):
    raise NotImplementedError('Implemented by the child class.')

  #################################
  # Define Extra Op When Training #
  #################################
  @staticmethod
  def do_extra_train_step(session, end_points, global_step):
    """This optional function is hooked up to the main training step function and is executed after each step."""
    pass

  ###################################
  # END OF THE IMPORTANT FUNCTIONS  #
  ###################################

  ######################
  # Dataset Utilities  #
  ######################

  @staticmethod
  def _get_batch(data):
    """Batch the input. Notice that inputs need to have known dimensions.

    Args:
      data: a dictionary.
    Returns:
      A dictionary with the same keys as input and batched values.
    """
    names = data.keys()
    batch_input = data.values()
    try:
      batch = tf.train.batch(
        batch_input,
        batch_size=FLAGS.batch_size,
        enqueue_many=False,
        num_threads=FLAGS.num_preprocessing_threads,
        capacity=5 * FLAGS.batch_size)
      if len(batch_input) == 1:
        batch = [batch]
    except ValueError:
      tf.logging.warning('Cannot create batch.')
      batch = [tf.expand_dims(current_input, 0) for current_input in batch_input]
    named_batch_data = {name: batch[i] for i, name in enumerate(names)}
    return named_batch_data

  @staticmethod
  def _select_image_preprocessing_fn():
    """A wrapper around preprocessing_factory.get_preprocessing()"""
    preprocessing_name = FLAGS.preprocessing_name or FLAGS.generator_network
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
      preprocessing_name,
      is_training=FLAGS.is_training, )
    if image_preprocessing_fn is not None:
      # TODO: this is convoluted. Perhaps combine this into the preprocessing factory.
      image_preprocessing_fn = functools.partial(image_preprocessing_fn,
                                                 dtype=GeneralModel._dtype_string_to_dtype(FLAGS.dataset_dtype),
                                                 color_space=FLAGS.color_space,
                                                 subtract_mean=FLAGS.subtract_mean,
                                                 resize_mode=FLAGS.resize_mode,
                                                 )
    return image_preprocessing_fn

  @staticmethod
  def _get_data_batched(batch_queue, batch_names, data_batched=None):
    """Returns data_batched from either dequeueing batch_queue or directly from `data_batched`.

    Returns:
      A dictionary with keys = `batch_names` and values = dequeued items.
    """
    ret = {}
    if batch_queue:
      assert data_batched is None, '`batch_queue` and `data_batched` are mutually exclusive.'
      with tf.device('cpu'):
        if isinstance(batch_queue, list):
          dequeued = batch_queue  # Failed to create batch_queue due to unknown tensor shape.
        else:
          dequeued = batch_queue.dequeue()
      # This deals with batch queue having only one input.
      if not isinstance(dequeued, list):
        dequeued = [dequeued]
      for i in range(len(dequeued)):
        ret[batch_names[i]] = dequeued[i]
      return ret
    else:
      assert data_batched is not None, '`batch_queue` and `data_batched` are mutually exclusive.'
      return data_batched

  @staticmethod
  def _do_preprocessing(tensor, image_preprocessing_fn, dataset, name=''):
    """Preprocess the given tensor depending on its variable type."""
    train_image_size = FLAGS.train_image_size
    do_random_cropping = FLAGS.do_random_cropping
    if image_preprocessing_fn and tensor is not None:
      # Note: variable casting to the dtype specified in the `dataset_dtype` flag occurs after preprocessing.
      if GeneralModel._maybe_is_image(tensor):
        tensor = image_preprocessing_fn(tensor, train_image_size, train_image_size,
                                        do_random_cropping=do_random_cropping, summary_prefix=name + '_')
      elif tensor.dtype.is_integer:
        tf.logging.info('Preprocessing integer tensor %s into one hot embedding.', name)
        tensor = util_misc.safe_one_hot_encoding(tensor, dataset.num_classes - FLAGS.labels_offset,
                                                 dtype=GeneralModel._dtype_string_to_dtype(FLAGS.dataset_dtype))
      elif tensor.dtype == tf.string:
        # Hard code last dimension to 3 for now.
        tensor = tf.reshape(tf.decode_raw(tensor, tf.uint8), (FLAGS.train_image_size, FLAGS.train_image_size, 3))
      else:
        tf.logging.info('Not preprocessing %s.', name)
    return tensor

  ###########################
  # Optimization Utilities  #
  ###########################

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
      ValueError: if the flag `learning_rate_decay_type` is not supported.

    """
    if start_learning_rate is None:
      start_learning_rate = FLAGS.learning_rate
    decay_steps = int(num_samples_per_epoch / FLAGS.batch_size *
                      FLAGS.num_epochs_per_decay)
    if FLAGS.sync_replicas:
      decay_steps /= FLAGS.replicas_to_aggregate  # Batch size is per replica.

    if FLAGS.learning_rate_decay_type == 'exponential':
      return tf.train.exponential_decay(start_learning_rate,
                                        global_step,
                                        decay_steps,
                                        FLAGS.learning_rate_decay_factor,
                                        staircase=True,
                                        name='exponential_decay_learning_rate')
    elif FLAGS.learning_rate_decay_type == 'fixed':
      return tf.constant(start_learning_rate, name='fixed_learning_rate')
    elif FLAGS.learning_rate_decay_type == 'polynomial':
      return tf.train.polynomial_decay(start_learning_rate,
                                       global_step,
                                       decay_steps,
                                       FLAGS.end_learning_rate,
                                       power=1.0,
                                       cycle=False,
                                       name='polynomial_decay_learning_rate')
    else:
      raise ValueError('learning_rate_decay_type [%s] was not recognized',
                       FLAGS.learning_rate_decay_type)

  @staticmethod
  def _configure_optimizer(learning_rate):
    """Configures the optimizer used for training.

    Args:
      learning_rate: A scalar or `Tensor` learning rate.

    Returns:
      An instance of an optimizer.

    Raises:
      ValueError: if FLAGS.optimizer is not recognized.
    """
    if FLAGS.optimizer == 'adadelta':
      optimizer = tf.train.AdadeltaOptimizer(
        learning_rate,
        rho=FLAGS.adadelta_rho,
        epsilon=FLAGS.opt_epsilon)
    elif FLAGS.optimizer == 'adagrad':
      optimizer = tf.train.AdagradOptimizer(
        learning_rate,
        initial_accumulator_value=FLAGS.adagrad_initial_accumulator_value)
    elif FLAGS.optimizer == 'adam':
      optimizer = tf.train.AdamOptimizer(
        learning_rate,
        beta1=FLAGS.adam_beta1,
        beta2=FLAGS.adam_beta2,
        epsilon=FLAGS.opt_epsilon)
    elif FLAGS.optimizer == 'ftrl':
      optimizer = tf.train.FtrlOptimizer(
        learning_rate,
        learning_rate_power=FLAGS.ftrl_learning_rate_power,
        initial_accumulator_value=FLAGS.ftrl_initial_accumulator_value,
        l1_regularization_strength=FLAGS.ftrl_l1,
        l2_regularization_strength=FLAGS.ftrl_l2)
    elif FLAGS.optimizer == 'momentum':
      optimizer = tf.train.MomentumOptimizer(
        learning_rate,
        momentum=FLAGS.momentum,
        name='Momentum')
    elif FLAGS.optimizer == 'rmsprop':
      optimizer = tf.train.RMSPropOptimizer(
        learning_rate,
        decay=FLAGS.rmsprop_decay,
        momentum=FLAGS.rmsprop_momentum,
        epsilon=FLAGS.opt_epsilon)
    elif FLAGS.optimizer == 'sgd':
      optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    else:
      raise ValueError('Optimizer [%s] was not recognized', FLAGS.optimizer)
    return optimizer

  @staticmethod
  def _get_gradient_scale():
    """Scale the gradients for mixed-precision training."""
    return 1.0 if FLAGS.dataset_dtype == FLAGS.variable_dtype else FLAGS.mix_precision_loss_scale

  #################
  # Save/Restore  #
  #################
  @staticmethod
  def _get_init_fn(checkpoint_path, checkpoint_exclude_scopes, use_dict=False, strip_scope=''):
    """Returns a function run by the chief worker to warm-start the training.

    Note that the init_fn is only run when initializing the model during the very
    first global step.

    Args
      checkpoint_path: string, path to checkpoint or it's directory. Ignored if the flag `train_dir` exists.
      checkpoint_exclude_scopes: string, A comma separated list of scopes to exclude.
      use_dict: If true, pass a dictionary instead of a list to the saver.
      strip_scope: If non-empty, the specified scope will be stripped in the dictionary keys.

    Returns:
      An init function run by the supervisor.
    """
    if checkpoint_path is None:
      return None

    if strip_scope and not use_dict:
      raise ValueError('strip_scope has to be used along with use_dict.')

    # Warn the user if a checkpoint exists in the train_dir. Then we'll be
    # ignoring the checkpoint anyway.
    if tf.train.latest_checkpoint(FLAGS.train_dir):
      tf.logging.info(
        'Ignoring --checkpoint_path because a checkpoint already exists in %s'
        % FLAGS.train_dir)
      return None

    exclusions = []
    if checkpoint_exclude_scopes:
      exclusions = [scope.strip()
                    for scope in checkpoint_exclude_scopes.split(',')]

    variables_to_restore = {} if use_dict else []
    vars = slim.get_model_variables()
    if not vars:
      vars = slim.get_variables()
    for var in vars:
      excluded = False
      for exclusion in exclusions:
        if var.op.name.startswith(exclusion):
          excluded = True
          break
      if excluded:
        continue

      if use_dict:
        var_name = var.name.rstrip(':0')  # No :0 in the checkpoint so stripping that.
        # If strip_scope is set, do not add variables that does not belong to the scope.
        if strip_scope and var_name.startswith(strip_scope):
          var_name = var_name.lstrip(strip_scope)
        else:
          continue
        variables_to_restore[var_name] = var
      else:
        variables_to_restore.append(var)

    if tf.gfile.IsDirectory(checkpoint_path):
      restore_from_path = tf.train.latest_checkpoint(checkpoint_path)
    else:
      restore_from_path = checkpoint_path

    tf.logging.info('Fine-tuning from %s. # Variables restored: %d.' % (restore_from_path, len(variables_to_restore)))

    return slim.assign_from_checkpoint_fn(
      restore_from_path,
      variables_to_restore,
      ignore_missing_vars=FLAGS.ignore_missing_vars)

  @staticmethod
  def _get_variables_to_train(trainable_scopes=None):
    """Returns a list of variables to train.

    Args:
      trainable_scopes: string, a comma separated list of variable scopes.

    Returns:
      A list of variables to train by the optimizer.
    """
    if trainable_scopes is None:
      ret = tf.trainable_variables()
    else:
      scopes = [scope.strip() for scope in trainable_scopes.split(',')]

      variables_to_train = []
      for scope in scopes:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        variables_to_train.extend(variables)
      ret = variables_to_train
    # For each variable, print their name and the number of trainable parameters it has.
    total_num_param = 0
    for var in ret:
      num_param = reduce(lambda x, y: x * y, var.shape)
      total_num_param += int(num_param)
      tf.logging.info('Variable %s %s of shape %s #param: %d', var.name, str(var.dtype), var.shape, num_param)
    tf.logging.info('Total number of trainable parameters: %d', total_num_param)
    return ret

  #########################
  # Tensorflow Utilities  #
  #########################

  @staticmethod
  def _dtype_string_to_dtype(dtype):
    return getattr(tf, dtype)

  @staticmethod
  def _maybe_cast(tensor, dtype):
    if tensor.dtype != dtype:
      return tf.cast(tensor, dtype)
    else:
      return tensor

  @staticmethod
  def _safe_one_hot_encoding(tensor, num_classes, dtype=None):
    """Given a (possibly out of range) vector of labels, transform them into one-hot encoding."""
    one_hot_encoded = slim.one_hot_encoding(
      tensor, num_classes, on_value=tf.constant(1, tf.int64),
      off_value=tf.constant(0, tf.int64))

    # This makes sure that when there are no labels, reduce_max will not output -inf but output 0s instead.
    stacked = tf.concat((tf.zeros((1, num_classes), dtype=one_hot_encoded.dtype),
                         one_hot_encoded), axis=0)
    tensor = tf.reduce_max(stacked, 0)
    if dtype is not None:
      tensor = tf.cast(tensor, dtype)  # GeneralModel._dtype_string_to_dtype(FLAGS.dataset_dtype)
    return tensor

  @staticmethod
  def _embed_one_hot(one_hot, embed_dim, name='one_hot_embedding'):
    """Assign each class a random vector and embed the one-hot tensor as the sum of its embeddings."""
    vocab_size = int(one_hot.shape[-1])
    embedding_lookup_matrix = tf.get_variable(name + '_lookup_matrix', shape=[vocab_size, embed_dim],
                                              dtype=one_hot.dtype,
                                              initializer=tf.random_uniform_initializer, trainable=False,
                                              collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.MODEL_VARIABLES])
    embedding = tf.matmul(one_hot, embedding_lookup_matrix)
    return embedding

  @staticmethod
  def _maybe_is_image(tensor):
    return len(tensor.shape) >= 3

  @staticmethod
  def _add_end_point_summaries(end_points, summaries):
    for end_point in end_points:
      x = end_points[end_point]
      summaries.add(tf.summary.histogram('activations/' + end_point, x))
      summaries.add(tf.summary.scalar('sparsity/' + end_point,
                                      tf.nn.zero_fraction(x)))

  def _add_one_image_summary(self, name, tensor, **kwargs):
    tf.summary.image(name, self._post_process_image(tensor), **kwargs)

  @staticmethod
  def _post_process_image(image):
    """Undo some preprocessing steps such as mean subtraction, dtype change etc."""
    # preprocessing and postprocessing shares the same name for convenience.
    preprocessing_name = FLAGS.preprocessing_name or FLAGS.generator_network
    postprocessing_fn = preprocessing_factory.get_postprocessing(preprocessing_name)
    postprocessing_fn = functools.partial(postprocessing_fn,
                                          color_space=FLAGS.color_space,
                                          subtract_mean=FLAGS.subtract_mean,
                                          )
    return postprocessing_fn(image)

  def _add_image_summaries(self, end_points, summaries):
    raise NotImplementedError('The child class needs to decide which end point to add to summary.')

  @staticmethod
  def _add_loss_summaries(first_clone_scope, summaries, end_points):
    for loss in tf.get_collection(tf.GraphKeys.LOSSES, first_clone_scope):
      summaries.add(tf.summary.scalar('losses/%s' % loss.op.name, loss))

  @staticmethod
  def _add_pr_curve(prediction, target_labels, summaries, name=''):
    """Adds a precision recall curve for classifications."""
    auc, auc_op = tf.metrics.auc(target_labels, tf.clip_by_value(prediction, 0, 1),
                                 updates_collections=tf.GraphKeys.UPDATE_OPS)
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, auc_op)
    summaries.add(tf.summary.scalar('losses/auc_metric%s' % name, auc))
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    precision, precision_op = tf.metrics.precision_at_thresholds(target_labels,
                                                                 tf.clip_by_value(prediction, 0, 1),
                                                                 thresholds,
                                                                 updates_collections=tf.GraphKeys.UPDATE_OPS)
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, precision_op)
    for i, threshold in enumerate(thresholds):
      summaries.add(tf.summary.scalar('losses/precision_at_%.2f_metric%s' % (threshold, name), precision[i]))
    recall, recall_op = tf.metrics.recall_at_thresholds(target_labels, tf.clip_by_value(prediction, 0, 1),
                                                        thresholds, updates_collections=tf.GraphKeys.UPDATE_OPS)
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, recall_op)
    for i, threshold in enumerate(thresholds):
      summaries.add(tf.summary.scalar('losses/recall_at_%.2f_metric%s' % (threshold, name), recall[i]))

  ########
  # Eval #
  ########

  @staticmethod
  def _define_eval_metrics(end_points, data_batched):
    """See slim.metrics.aggregate_metric_map()."""
    metric_map = collections.OrderedDict()
    losses = tf.get_collection(tf.GraphKeys.LOSSES, )
    for loss in losses:
      metric_map[loss.name.strip('/value:0')] = slim.metrics.streaming_mean(loss)
    return metric_map

  @staticmethod
  def _create_synthetic_data(dataset):
    """Create synthetic data for testing processing speed. Child class is responsible for changing this if necessary."""
    source = tf.truncated_normal(
      (FLAGS.train_image_size, FLAGS.train_image_size, 3),
      dtype=GeneralModel._dtype_string_to_dtype(FLAGS.dataset_dtype),
      stddev=1e-1,
      name='synthetic_images')
    target = tf.random_uniform(
      [FLAGS.batch_size],
      minval=0,
      maxval=dataset.num_classes - 1,
      dtype=tf.int32,
      name='synthetic_labels')
    return {'source': source, 'target': target}

  def _prepare_data_aux(self, dataset, image_preprocessing_fn, deploy_config, is_synthetic=False, name=''):
    """Auxiliary function for preparing one single dataset."""
    with tf.device(deploy_config.inputs_device()):
      ret_raw = {}
      if not is_synthetic:
        provider = slim.dataset_data_provider.DatasetDataProvider(
          dataset,
          num_readers=FLAGS.num_readers,
          shuffle=False if FLAGS.do_output else True,
          num_epochs=1 if FLAGS.do_output else None,  # If `do_output` is true, go through input only once.
          common_queue_capacity=20 * FLAGS.batch_size,
          common_queue_min=0 if FLAGS.do_output else 10 * FLAGS.batch_size)

        for item in provider.list_items():
          [item_tensor] = provider.get([item])
          ret_raw[item] = item_tensor
      else:
        tf.logging.warning('Flag `dataset_dir` is empty! Using synthetic data to test training speed.')
        ret_raw = self._create_synthetic_data(dataset)

    if FLAGS.preprocess_on_clone:
      preprocess_device = deploy_config.clone_device(0)
    else:
      preprocess_device = deploy_config.inputs_device()

    ret = {}
    if hasattr(dataset, 'items_used'):
      items_used = dataset.items_used
    else:
      items_used = ['source', 'target']
    if hasattr(dataset, 'items_need_preprocessing'):
      items_need_preprocessing = dataset.items_need_preprocessing
    else:
      items_need_preprocessing = ['source', 'target']

    with tf.device(preprocess_device):
      for key, val in ret_raw.iteritems():
        if key in items_used:
          if key in items_need_preprocessing and image_preprocessing_fn is not None:
            ret[key] = self._do_preprocessing(val, image_preprocessing_fn, dataset, name=name + key)
          else:
            ret[key] = val
    return ret

  def _prepare_data(self, dataset, image_preprocessing_fn, deploy_config, ):
    """Does preprocessing on input datasets."""
    if FLAGS.unpaired_target_dataset_name:
      assert len(dataset) == 2, 'If `unpaired_target_dataset_name` is on, there must be two datasets.'
      data_a = self._prepare_data_aux(dataset[0], image_preprocessing_fn, deploy_config,
                                      is_synthetic=(not FLAGS.dataset_dir), name='a')
      data_b = self._prepare_data_aux(dataset[1], image_preprocessing_fn, deploy_config,
                                      is_synthetic=(not FLAGS.unpaired_target_dataset_dir), name='b')
      return util_misc.combine_dicts({'a': data_a, 'b': data_b})
    else:
      # Do the default if the dataset is paired.
      return self._prepare_data_aux(dataset, image_preprocessing_fn, deploy_config, )

  @staticmethod
  def _maybe_encode_output_tensor(output_tensors):
    """Given a list of output tensors, encode the images for output."""
    ret = []
    for output_tensor in output_tensors:
      if isinstance(output_tensor, list) or isinstance(output_tensor, tuple):
        output_tensor = OutputTensor(*output_tensor)
      if output_tensor.is_image:
        if len(output_tensor.tensor.shape) == 3:
          image = tf.expand_dims(output_tensor.tensor, 0)
        elif len(output_tensor.tensor.shape) == 4:
          image = output_tensor.tensor
        else:
          raise ValueError(
            'Image must be of dimension 3 or 4. The input shape is instead: %s' % (str(output_tensor.tensor.shape)))

        ret.append(OutputTensor(name=output_tensor.name, is_image=output_tensor.is_image,
                                tensor=tf.map_fn(tf.image.encode_jpeg, tf.image.convert_image_dtype(image, tf.uint8),
                                                 dtype=tf.string, name=output_tensor.name + '_encoded_jpeg')))
      else:
        ret.append(OutputTensor(name=output_tensor.name, is_image=output_tensor.is_image,
                                tensor=tf.map_fn(tf.identity, output_tensor.tensor,
                                                 name=output_tensor.name + '_identity')))
    return ret

  def _define_extra_eval_actions(self, end_points, data_batched):
    pass

  def _do_extra_eval_actions(self, session, extra_eval):
    pass

  def get_items_to_encode(self, end_points, data_batched):
    """Outputs a list with format OutputTensor(name, is_image, tensor)"""
    return NotImplementedError('To be implemented by child class.')

  def _get_encode_op_feed_dict(self, end_points, encode_ops, i):
    return None

  @staticmethod
  def save_images(fetches, image_dir):
    """Given a list of `OutputTensor`s, save the images to `image_dir`."""
    if not os.path.isdir(image_dir):
      util_io.touch_folder(image_dir)
    filesets = []
    now = str(int(time.time() * 1000))

    for name, is_image, vals in fetches:
      if is_image:
        image_names = []
        filesets.append((name, is_image, image_names))
        for i, val in enumerate(vals):
          filename = name + '_' + now + '_' + str(i) + '.jpg'
          image_names.append(filename)
          out_path = os.path.join(image_dir, filename)
          with open(out_path, 'w') as f:
            f.write(val)
      else:
        filesets.append((name, is_image, vals))

    return filesets

  @staticmethod
  def to_human_friendly(eval_items):
    raise NotImplementedError

  def _write_eval_html(self, filesets):
    index_path = os.path.join(FLAGS.eval_dir, 'index.html')
    if len(filesets) == 0:
      return
    num_items = len(filesets[0][-1])

    if os.path.exists(index_path):
      index = open(index_path, 'a')
    else:
      index = open(index_path, 'w')
      index.write(
        '<html><meta content=\'text/html;charset=utf-8\' http-equiv=\'Content-Type\'>\n'
        '<meta content=\'utf-8\' http-equiv=\'encoding\'>\n'
        '<body>\n'
        '  <table>\n'
        '     <tr>')

      for item in filesets:
        index.write('<th>%s</th>' % item[0])  # Write column name.
      index.write('</tr>\n')

    for i in range(num_items):
      index.write('<tr>')
      for key_i in range(len(filesets)):
        # In case the number of items are different.
        if i < len(filesets[key_i][2]):
          is_image = filesets[key_i][1]
          if is_image:
            index.write('<td><img src=\'images/%s\'></td>' % urllib.quote(filesets[key_i][2][i]))
          else:
            index.write('<td>%s</td>' % str(filesets[key_i][2][i]))
        else:
          index.write('<td></td>')
      index.write('</tr>\n')
    return index_path

  ########################
  # Export Trained Model #
  ########################
  @staticmethod
  def _define_outputs(end_points, data_batched):
    raise NotImplementedError('To be implemented by child classes.')

  @staticmethod
  def _write_outputs(output_results, output_ops, ):
    raise NotImplementedError('To be implemented by child classes.')

  @staticmethod
  def _build_signature_def_map(end_points, data_batched):
    raise NotImplementedError('To be implemented by child classes.')

  @staticmethod
  def _build_assets_collection(end_points, data_batched):
    raise NotImplementedError('To be implemented by child classes.')

  ########
  # Main #
  ########

  def main(self):
    tf.logging.set_verbosity(tf.logging.INFO)
    # Set session_config to allow some operations to be run on cpu.
    session_config = tf.ConfigProto(allow_soft_placement=True, )

    with tf.Graph().as_default():
      ######################
      # Select the dataset #
      ######################
      dataset = self._select_dataset()

      ######################
      # Select the network #
      ######################
      networks = self._select_network()

      #####################################
      # Select the preprocessing function #
      #####################################
      image_preprocessing_fn = self._select_image_preprocessing_fn()

      #######################
      # Config model_deploy #
      #######################
      deploy_config = model_deploy.DeploymentConfig(
        num_clones=FLAGS.num_clones,
        clone_on_cpu=FLAGS.clone_on_cpu,
        replica_id=FLAGS.task,
        num_replicas=FLAGS.worker_replicas,
        num_ps_tasks=FLAGS.num_ps_tasks)

      global_step = slim.create_global_step()

      ##############################################################
      # Create a dataset provider that loads data from the dataset #
      ##############################################################
      data = self._prepare_data(dataset, image_preprocessing_fn, deploy_config, )
      data_batched = self._get_batch(data)
      batch_names = data_batched.keys()
      batch = data_batched.values()

      ###############
      # Is Training #
      ###############
      if FLAGS.is_training:
        if not os.path.isdir(FLAGS.train_dir):
          util_io.touch_folder(FLAGS.train_dir)
        if not os.path.exists(os.path.join(FLAGS.train_dir, FLAGS_FILE_NAME)):
          FLAGS.append_flags_into_file(os.path.join(FLAGS.train_dir, FLAGS_FILE_NAME))

        try:
          batch_queue = slim.prefetch_queue.prefetch_queue(
            batch, capacity=4 * deploy_config.num_clones)
        except ValueError as e:
          tf.logging.warning('Cannot use batch_queue due to error %s', e)
          batch_queue = batch
        # Gather initial summaries.
        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

        clones = model_deploy.create_clones(deploy_config, self._clone_fn,
                                            GeneralModel._dtype_string_to_dtype(FLAGS.variable_dtype),
                                            [networks, batch_queue, batch_names],
                                            {'global_step': global_step,
                                             'is_training': FLAGS.is_training})
        first_clone_scope = deploy_config.clone_scope(0)

        # Add summaries for end_points.
        end_points = clones[0].outputs
        self._end_points_for_debugging = end_points
        self._add_end_point_summaries(end_points, summaries)
        # Add summaries for images, if there are any.
        self._add_image_summaries(end_points, summaries)
        # Add summaries for losses.
        self._add_loss_summaries(first_clone_scope, summaries, end_points)
        # Add summaries for variables.
        for variable in slim.get_model_variables():
          summaries.add(tf.summary.histogram(variable.op.name, variable))

        #################################
        # Configure the moving averages #
        #################################
        if FLAGS.moving_average_decay:
          moving_average_variables = slim.get_model_variables()
          variable_averages = tf.train.ExponentialMovingAverage(
            FLAGS.moving_average_decay, global_step)
        else:
          moving_average_variables, variable_averages = None, None

        #########################################
        # Configure the optimization procedure. #
        #########################################
        # Gather update_ops from the first clone. These contain, for example,
        # the updates for the batch_norm variables created by generator_network_fn.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone_scope)

        with tf.device(deploy_config.optimizer_device()):
          learning_rate = self._configure_learning_rate(self.num_samples, global_step)
          optimizer = self._configure_optimizer(learning_rate)

        if FLAGS.sync_replicas:
          # If sync_replicas is enabled, the averaging will be done in the chief
          # queue runner.
          optimizer = tf.train.SyncReplicasOptimizer(
            opt=optimizer,
            replicas_to_aggregate=FLAGS.replicas_to_aggregate,
            total_num_replicas=FLAGS.worker_replicas,
            variable_averages=variable_averages,
            variables_to_average=moving_average_variables)
        elif FLAGS.moving_average_decay:
          # Update ops executed locally by trainer.
          update_ops.append(variable_averages.apply(moving_average_variables))

        summaries.add(tf.summary.scalar('learning_rate', learning_rate))
        # Define optimization process.
        train_tensor = self._add_optimization(clones, optimizer, summaries, update_ops, global_step)

        # Add the summaries from the first clone. These contain the summaries
        # created by model_fn and either optimize_clones() or _gather_clone_loss().
        summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES,
                                           first_clone_scope))

        # Merge all summaries together.
        summary_op = tf.summary.merge(list(summaries), name='summary_op')

        # Define train_step with eval every `eval_every_n_steps`.
        def train_step_fn(session, *args, **kwargs):
          self.do_extra_train_step(session, end_points, global_step)
          total_loss, should_stop = slim.learning.train_step(session, *args, **kwargs)
          return [total_loss, should_stop]

        ###########################
        # Kicks off the training. #
        ###########################
        slim.learning.train(
          train_tensor,
          train_step_fn=train_step_fn,
          logdir=FLAGS.train_dir,
          master=FLAGS.master,
          is_chief=(FLAGS.task == 0),
          init_fn=self._get_init_fn(FLAGS.checkpoint_path, FLAGS.checkpoint_exclude_scopes),
          summary_op=summary_op,
          number_of_steps=FLAGS.max_number_of_steps,
          log_every_n_steps=FLAGS.log_every_n_steps,
          save_summaries_secs=FLAGS.save_summaries_secs,
          save_interval_secs=FLAGS.save_interval_secs,
          sync_optimizer=optimizer if FLAGS.sync_replicas else None,
          session_config=session_config)
      ##########################
      # Eval, Export or Output #
      ##########################
      else:
        # Write flags file.
        if not os.path.isdir(FLAGS.eval_dir):
          util_io.touch_folder(FLAGS.eval_dir)
        if not os.path.exists(os.path.join(FLAGS.eval_dir, FLAGS_FILE_NAME)):
          FLAGS.append_flags_into_file(os.path.join(FLAGS.eval_dir, FLAGS_FILE_NAME))

        with tf.variable_scope(tf.get_variable_scope(),
                               custom_getter=model_deploy.get_custom_getter(
                                 GeneralModel._dtype_string_to_dtype(FLAGS.variable_dtype)),
                               reuse=False):
          end_points = self._clone_fn(networks, batch_queue=None, batch_names=batch_names, data_batched=data_batched,
                                      is_training=False, global_step=global_step)

        num_batches = int(math.ceil(self.num_samples / float(FLAGS.batch_size)))

        checkpoint_path = util_misc.get_latest_checkpoint_path(FLAGS.checkpoint_path)

        if FLAGS.moving_average_decay:
          variable_averages = tf.train.ExponentialMovingAverage(
            FLAGS.moving_average_decay, global_step)
          variables_to_restore = variable_averages.variables_to_restore(
            slim.get_model_variables())
          variables_to_restore[global_step.op.name] = global_step
        else:
          variables_to_restore = slim.get_variables_to_restore()

        saver = None
        if variables_to_restore is not None:
          saver = tf.train.Saver(variables_to_restore)

        session_creator = tf.train.ChiefSessionCreator(
          scaffold=tf.train.Scaffold(saver=saver),
          checkpoint_filename_with_path=checkpoint_path,
          master=FLAGS.master,
          config=session_config)

        ##########
        # Output #
        ##########
        if FLAGS.do_output:
          tf.logging.info('Output mode.')
          output_ops = self._maybe_encode_output_tensor(self._define_outputs(end_points, data_batched))
          start_time = time.time()
          with tf.train.MonitoredSession(
              session_creator=session_creator) as session:
            for i in range(num_batches):
              output_results = session.run([item[-1] for item in output_ops])
              self._write_outputs(output_results, output_ops)
              if i % FLAGS.log_every_n_steps == 0:
                current_time = time.time()
                speed = (current_time - start_time) / (i + 1)
                time_left = speed * (num_batches - i + 1)
                tf.logging.info('%d / %d done. Time left: %f', i + 1, num_batches, time_left)


        ################
        # Export Model #
        ################
        elif FLAGS.do_export:
          tf.logging.info('Exporting trained model to %s', FLAGS.export_path)
          with tf.Session(config=session_config) as session:
            saver.restore(session, checkpoint_path)
            builder = tf.saved_model.builder.SavedModelBuilder(FLAGS.export_path)
            signature_def_map = self._build_signature_def_map(end_points, data_batched)
            assets_collection = self._build_assets_collection(end_points, data_batched)
            legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
            builder.add_meta_graph_and_variables(
              session, [tf.saved_model.tag_constants.SERVING],
              signature_def_map=signature_def_map,
              legacy_init_op=legacy_init_op,
              assets_collection=assets_collection,
            )
          builder.save()
          tf.logging.info('Done exporting!')

        ########
        # Eval #
        ########
        else:
          tf.logging.info('Eval mode.')
          # Add summaries for images, if there are any.
          self._add_image_summaries(end_points, None)

          # Define the metrics:
          metric_map = self._define_eval_metrics(end_points, data_batched)

          names_to_values, names_to_updates = slim.metrics.aggregate_metric_map(metric_map)
          names_to_values = collections.OrderedDict(**names_to_values)
          names_to_updates = collections.OrderedDict(**names_to_updates)

          # Print the summaries to screen.
          for name, value in names_to_values.items():
            summary_name = 'eval/%s' % name
            if len(value.shape):
              op = tf.summary.tensor_summary(summary_name, value, collections=[])
            else:
              op = tf.summary.scalar(summary_name, value, collections=[])
            op = tf.Print(op, [value], summary_name)
            tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

          if not (FLAGS.do_eval_debug or FLAGS.do_custom_eval):
            tf.logging.info('Evaluating %s' % checkpoint_path)

            slim.evaluation.evaluate_once(
              master=FLAGS.master,
              checkpoint_path=checkpoint_path,
              logdir=FLAGS.eval_dir,
              num_evals=num_batches,
              eval_op=list(names_to_updates.values()),
              variables_to_restore=variables_to_restore,
              session_config=session_config)
            return

          ################################
          # `do_eval_debug` flag is true.#
          ################################
          if FLAGS.do_eval_debug:
            eval_ops = list(names_to_updates.values())
            eval_names = list(names_to_updates.keys())

            # Items to write to a html page.
            encode_ops = self._maybe_encode_output_tensor(self.get_items_to_encode(end_points, data_batched))

            with tf.train.MonitoredSession(session_creator=session_creator) as session:
              if eval_ops is not None:
                for i in range(num_batches):
                  eval_result = session.run(eval_ops, None)
                  print('; '.join(('%s:%s' % (name, str(eval_result[i])) for i, name in enumerate(eval_names))))

              # Write to HTML
              if encode_ops:
                for i in range(num_batches):
                  encode_ops_feed_dict = self._get_encode_op_feed_dict(end_points, encode_ops, i)
                  encoded_items = session.run([item[-1] for item in encode_ops], encode_ops_feed_dict)
                  encoded_list = []
                  for j in range(len(encoded_items)):
                    encoded_list.append((encode_ops[j][0], encode_ops[j][1], encoded_items[j].tolist()))

                  eval_items = self.save_images(encoded_list, os.path.join(FLAGS.eval_dir, 'images'))
                  eval_items = self.to_human_friendly(eval_items, )
                  self._write_eval_html(eval_items)
                  if i % 10 == 0:
                    tf.logging.info('%d/%d' % (i, num_batches))
          if FLAGS.do_custom_eval:
            extra_eval = self._define_extra_eval_actions(end_points, data_batched)
            with tf.train.MonitoredSession(session_creator=session_creator) as session:
              self._do_extra_eval_actions(session, extra_eval)


def main(_):
  raise NotImplementedError()


if __name__ == '__main__':
  tf.app.run()
