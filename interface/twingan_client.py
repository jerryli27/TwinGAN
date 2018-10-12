# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# !/usr/bin/env python2.7

"""A client that talks to tensorflow_model_server loaded with the TwinGAN model."""

from __future__ import print_function

import os
import sys
import time

import numpy as np
import scipy.misc
import tensorflow as tf
from grpc.beta import implementations  # pip install grpcio
from tensorflow_serving.apis import predict_pb2  # pip install tensorflow-serving-api
from tensorflow_serving.apis import prediction_service_pb2

import util_io

# This is a placeholder for a Google-internal import.

tf.flags.DEFINE_integer('concurrency', 1,
                        'maximum number of concurrent inference requests')
tf.flags.DEFINE_integer('image_hw', 4, 'Height and width to resize the input image to.')
tf.flags.DEFINE_string('twingan_server', None, 'PredictionService host:port')
tf.flags.mark_flag_as_required('twingan_server')
FLAGS = tf.flags.FLAGS


class TwinGANClient(object):

  def __init__(self, hostport, image_hw, model_spec_name='test', model_spec_signature_name='serving_default',
               concurrency=1, ):
    """

    Args:
      hostport: Host:port address of the PredictionService.
      image_hw: Width and height of the input image.
      model_spec_name:
      model_spec_signature_name:
      concurrency: Maximum number of concurrent requests.
    """
    self.hostport = hostport
    self.image_hw = image_hw
    self.model_spec_name = model_spec_name
    self.model_spec_signature_name = model_spec_signature_name
    self.concurrency = concurrency

    try:
      host, port = hostport.split(':')
      self.channel = implementations.insecure_channel(host, int(port))
    except ValueError as e:
      tf.logging.error('Cannot parse hostport %s' % hostport)
    self.request_template = predict_pb2.PredictRequest()
    self.request_template.model_spec.name = model_spec_name
    self.request_template.model_spec.signature_name = model_spec_signature_name

  @staticmethod
  def _request_set_input_image(request, input_image):
    request.inputs['inputs'].CopyFrom(tf.contrib.util.make_tensor_proto(input_image))

  @staticmethod
  def _create_rpc_callback(output_path):
    """Creates RPC callback function.

    Args:
      output_path: path to save returned images in callback.
    Returns:
      The callback function.
    """

    def _callback(result_future):
      """Callback function.

      Calculates the statistics for the prediction result.

      Args:
        result_future: Result future of the RPC.
      """
      exception = result_future.exception()
      if exception:
        print(exception)
      else:
        sys.stdout.write('.')
        sys.stdout.flush()
        # TODO: do post-processing using another function.
        response_images = np.reshape(np.array(
          result_future.result().outputs['outputs'].float_val),
          [dim.size for dim in result_future.result().outputs['outputs'].tensor_shape.dim]) * 255.0

        util_io.imsave(output_path, response_images[0])

    return _callback

  def do_inference(self, output_dir, image_path=None, image_np=None):
    """Tests PredictionService with concurrent requests.

    Args:
      output_dir: Directory to output image.
      image_path: Path to image.
      image_np: Image in np format. Ignored when image_path is set.

    Returns:
      `output_dir`.
    """
    if image_path is None and image_np is None:
      raise ValueError('Either `image_np` or `image_path` must be specified.')

    if image_path:
      image_resized = util_io.imread(image_path, (self.image_hw, self.image_hw))
    else:
      image_resized = scipy.misc.imresize(image_np, (self.image_hw, self.image_hw))
    # TODO: do preprocessing in a separate function. Check whether image has already been preprocessed.
    image = np.expand_dims(image_resized / np.float32(255.0), 0)

    stub = prediction_service_pb2.beta_create_PredictionService_stub(self.channel)
    request = predict_pb2.PredictRequest()
    request.CopyFrom(self.request_template)
    self._request_set_input_image(request, image)
    result_future = stub.Predict.future(request, 5.0)  # 5 seconds
    result_future.add_done_callback(self._create_rpc_callback(output_dir))
    return output_dir

  def block_on_callback(self, output_dir):
    while not os.path.exists(output_dir):
      time.sleep(0.001)


class MockTwinGANClient(TwinGANClient):
  """Returns the same image regardless of the input. Used for debugging."""
  def __init__(self, hostport, image_hw, **kwargs):
    super(MockTwinGANClient, self).__init__(hostport, image_hw,**kwargs)
    self.mock_output_image = util_io.imread('static/images/mock/mock_twingan_output.png',
                                            shape=(image_hw,image_hw))

  def do_inference(self, output_dir, image_path=None, image_np=None):
    util_io.imsave(output_dir, self.mock_output_image)
    return output_dir



def main(_):
  print("""Another way to test the inference model: 
        saved_model_cli run --dir 'path/to/export/model' \
        --tag_set serve  --signature_def serving_default --input_exprs 'inputs=np.ones((1,4,4,3))'""")
  if not FLAGS.twingan_server:
    print('please specify twingan_server host:port')
    return
  util_io.touch_folder(FLAGS.output_dir)
  img_basename = os.path.basename(FLAGS.image_path)

  client = TwinGANClient(FLAGS.twingan_server, FLAGS.image_hw, concurrency=FLAGS.concurrency)
  output_dir = os.path.join(FLAGS.output_dir, img_basename)
  client.do_inference(image_path=FLAGS.image_path, output_dir=output_dir)
  client.block_on_callback(output_dir)
  print('\nDone')


if __name__ == '__main__':
  # Sanity check.
  # cd TwinGAN/interface
  # python twingan_client.py
  tf.flags.DEFINE_string('image_path', '../demo/web_interface_input/', 'Path to the image to be translated.')
  tf.flags.DEFINE_string('output_dir', '../demo/face_translated/', 'Path to output directory for translated images.')
  tf.app.run()
