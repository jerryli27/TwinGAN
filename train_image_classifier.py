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
# This is a copy of slim/train_image_classifier with slight modifications
# to allow multiple labels for the same image.
# ==============================================================================
"""Generic training script that trains a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

import util_misc
from datasets import convert_danbooru_data
from deployment import model_deploy
from model import model_inheritor
from nets import grad_cam
from nets import nets_factory

##########################
# Network and loss Flags #
##########################
tf.flags.DEFINE_string(
  'model_name', 'inception_v3',
  'The name of the image classification architecture used.')
tf.flags.DEFINE_float(
  'classification_threshold', 0.25,
  'Labels are considered to be present if the classification value crosses this threshold.'
  'Currently used only when `do_eval_debug` flag is set.')
tf.flags.DEFINE_boolean(
  'predict_multilabel', True,
  'If true, we predict a single 0~1 score for each class. Otherwise the classes as mutually exclusive.'
)

tf.flags.DEFINE_boolean(
  'output_single_file', False,
  'If true, the output mode (where it outputs the predicted labels for each image) will only output to one file.')
tf.flags.DEFINE_string(
  'output_single_file_name', 'output.csv',
  'Name of the output file.')

FLAGS = tf.flags.FLAGS

PRELOGITS_LAYER_NAME = 'PreLogits'


class ClassifierModel(model_inheritor.GeneralModel):
  """This class has not yet been refactored."""
  ######################
  # Select the network #
  ######################
  def _select_network(self):
    network_fn = nets_factory.get_network_fn(
      FLAGS.model_name,
      num_classes=(self.num_classes - FLAGS.labels_offset),
      weight_decay=FLAGS.weight_decay,
      is_training=FLAGS.is_training,
    )
    return network_fn

  ####################
  # Define the model #
  ####################
  @staticmethod
  def _clone_fn(networks, batch_queue, batch_names, data_batched=None, is_training=False, **kwargs):
    """Allows data parallelism by creating multiple clones of network_fn."""
    data_batched = super(ClassifierModel, ClassifierModel)._get_data_batched(batch_queue, batch_names,
                                                                             data_batched)
    images = data_batched.get('source')
    labels = data_batched.get('target')
    if labels is None or images is None:
      raise ValueError('images and labels have to be available in the dataset.')
    network_fn = networks
    try:
      logits, end_points = network_fn(images, prediction_fn=tf.sigmoid, create_aux_logits=False)
    except TypeError:
      tf.logging.warning('Cannot set prediction_fn to sigmoid, or create_aux_logits to False!')
      logits, end_points = network_fn(images, )
    if FLAGS.dataset_dtype == 'float16' and 'AuxLogits' in end_points:
      end_points['AuxLogits'] = tf.cast(end_points['AuxLogits'], tf.float32)
    end_points['Logits'] = tf.cast(end_points['Logits'], tf.float32)

    end_points['images'] = images
    end_points['labels'] = labels
    ClassifierModel.add_loss(data_batched, end_points)

    return end_points

  ####################
  # Define the loss  #
  ####################
  @staticmethod
  def add_loss(data_batched, end_points, discriminator_network_fn=None):
    targets = data_batched.get('target')
    loss_fn = tf.losses.sigmoid_cross_entropy if FLAGS.predict_multilabel else tf.losses.softmax_cross_entropy

    if 'AuxLogits' in end_points:
      loss_fn(targets, end_points['AuxLogits'], weights=0.4, scope='aux_loss')
    loss_fn(targets, end_points['Logits'], weights=1.0)

  def _add_optimization(self, clones, optimizer, summaries, update_ops, global_step):
    # Variables to train.
    variables_to_train = self._get_variables_to_train()

    #  and returns a train_tensor and summary_op
    total_loss, clones_gradients = model_deploy.optimize_clones(
      clones,
      optimizer,
      gradient_scale=self._get_gradient_scale(),
      var_list=variables_to_train)
    # Add total_loss to summary.
    summaries.add(tf.summary.scalar('total_loss', total_loss))
    if clones_gradients:
      # Add summaries to the gradients.
      summaries |= set(model_deploy.add_gradients_summaries(clones_gradients))

    # Create gradient updates.
    grad_updates = optimizer.apply_gradients(clones_gradients,
                                             global_step=global_step)
    update_ops.append(grad_updates)

    update_op = tf.group(*update_ops)
    with tf.control_dependencies([update_op]):
      train_tensor = tf.identity(total_loss, name='train_op')
    return train_tensor

  #############
  # Summaries #
  #############

  @staticmethod
  def _define_eval_metrics(end_points, data_batched):
    metric_map = super(ClassifierModel, ClassifierModel)._define_eval_metrics(end_points, data_batched)
    # Define the metrics:
    # streaming_auc requires inputs to be within [0,1]
    targets = data_batched.get('target')
    clipped_predictions = tf.clip_by_value(end_points['Predictions'], 0, 1)
    metric_map['AUC'] = tf.metrics.auc(targets, clipped_predictions)
    metric_map['mean_squared_error'] = slim.metrics.streaming_mean_squared_error(end_points['Predictions'], targets)
    metric_map['precision_at_thresholds'] = tf.metrics.precision_at_thresholds(targets, clipped_predictions,
                                                                               [i / 10.0 for i in range(0, 11)])
    metric_map['recall_at_thresholds'] = tf.metrics.recall_at_thresholds(targets, clipped_predictions,
                                                                         [i / 10.0 for i in range(0, 11)])
    return metric_map

  def _add_image_summaries(self, end_points, summaries):
    # Add summaries for images, if there are any.
    if self._maybe_is_image(end_points['images']):
      self._add_one_image_summary('images', end_points['images'])

  @staticmethod
  def _add_loss_summaries(first_clone_scope, summaries, end_points):
    super(ClassifierModel, ClassifierModel)._add_loss_summaries(first_clone_scope, summaries, end_points)
    # Adds loss metrics.
    if 'Predictions' in end_points:
      auc, auc_op = tf.metrics.auc(end_points['labels'], tf.clip_by_value(end_points['Predictions'], 0, 1),
                                   updates_collections=tf.GraphKeys.UPDATE_OPS)
      summaries.add(tf.summary.scalar('losses/auc_metric', auc))
    else:
      tf.logging.warning('Cannot calculate the auc because there is no endpoint called "Predictions".')

  ###########################
  # Eval and output results #
  ###########################

  def get_items_to_encode(self, end_points, data_batched):
    """Outputs a list with format (name, is_image, tensor)"""
    targets = data_batched.get('target')
    items_to_encode = [
      ('sources', True, self._post_process_image(data_batched.get('source'))),
      ('targets', False, targets),
      ('predictions', False, end_points['Predictions']),
    ]
    for class_i in range(10):
      grad_cam_mask_class_i = grad_cam.grad_cam(FLAGS.model_name, end_points, class_i)
      masked_source_class_i = grad_cam.impose_mask_on_image(grad_cam_mask_class_i, data_batched)
      one_hot_class = tf.one_hot([class_i for _ in range(targets.shape[0])], targets.shape[-1])
      items_to_encode.append(('class_%d_name' % (class_i), False, one_hot_class))
      items_to_encode.append(('masked_source_class_%d' % (class_i), True,
                              self._post_process_image(masked_source_class_i)), )

    return items_to_encode

  @staticmethod
  def to_human_friendly(eval_items, delimiter=' '):
    ret = []
    labels_dict = util_misc.get_tags_dict(FLAGS.tags_id_lookup_file, FLAGS.tags_key_column_index,
                                          FLAGS.tags_value_column_index)

    for name, is_image, vals in eval_items:
      if is_image:
        ret.append((name, is_image, vals))
      else:
        human_readable_vals = []
        for val in vals:
          if isinstance(val, str):
            human_readable_vals = vals
            break
          human_readable_val = []
          if FLAGS.process_mutually_exclusive_labels:
            val = util_misc.process_anime_face_labels(val, FLAGS.classification_threshold)

          for i, item in enumerate(val):
            # The best way is to get the threshold from an AUC eval.
            if item >= FLAGS.classification_threshold:
              human_readable_val.append(labels_dict.get(i, 'UKNOWN_LABEL'))
          human_readable_vals.append(' '.join(human_readable_val))
        ret.append((name, is_image, human_readable_vals))
    return ret

  @staticmethod
  def _define_outputs(end_points, data_batched):
    """Output label predictions for each image."""
    if FLAGS.output_single_file:
      return [
        ('prelogits', False, end_points[PRELOGITS_LAYER_NAME]),
        ('filename', False, data_batched.get('filename')),
        ('predictions', False, end_points['Predictions']),
      ]

    else:
      return [
        ('sources', True, ClassifierModel._post_process_image(data_batched.get('source'))),
        ('filename', False, data_batched.get('filename')),
        ('predictions', False, end_points['Predictions']),
      ]

  @staticmethod
  def _write_outputs(output_results, output_ops, ):
    save_dir = FLAGS.eval_dir
    if FLAGS.output_single_file:
      single_file_name = os.path.join(save_dir, FLAGS.output_single_file_name)
      # Flatten the prelogits
      output_results[0] = np.reshape(output_results[0], [output_results[0].shape[0], output_results[0].shape[-1]])
      output_results = [item.tolist() for item in output_results]

      with open(single_file_name, 'ab') as f:
        writer = csv.writer(f)
        writer.writerows([[output_results[1][i]] + output_results[0][i] + output_results[2][i]
                          for i in range(len(output_results[0]))])
    else:
      encoded_list = []
      for j in range(len(output_results)):
        encoded_list.append(output_ops[j][:-1] + (output_results[j].tolist(),))
      items = ClassifierModel.save_images(encoded_list, save_dir)
      human_friendly_results = ClassifierModel.to_human_friendly(items,
                                                                 delimiter=convert_danbooru_data.TAG_TEXT_DELIMITER)

      num_labels_written = 0
      for i, predictions in enumerate(human_friendly_results[-1][-1]):
        if FLAGS.process_mutually_exclusive_labels:
          if not predictions:
            try:
              tf.gfile.Remove(os.path.join(save_dir, human_friendly_results[0][2][i]))
            except tf.errors.OpError as e:
              tf.logging.warning(e)
            continue  # Skip empty predictions. (The image will still be written but there will be no label).

        image_name = human_friendly_results[1][2][i]
        try:
          tf.gfile.Rename(os.path.join(save_dir, human_friendly_results[0][2][i]), os.path.join(save_dir, image_name))
        except tf.errors.OpError as e:
          tf.logging.warning(e)
        tags_file_path = os.path.join(save_dir, image_name + '.txt')
        with open(tags_file_path, 'w') as f:
          f.write(predictions)
        num_labels_written += 1

      tf.logging.info('%d label files are written.' % num_labels_written)

  def main(self):
    super(ClassifierModel, self).main()


def main(_):
  model = ClassifierModel()
  model.main()


if __name__ == '__main__':
  tf.app.run()
