#!/usr/bin/env python
"""

Before running this please run
tensorflow_model_server --port=9000 --model_name=test --model_base_path=/PATH/TO/MODEL/export/

cd TwinGAN/interface
python server.py --twingan_server=localhost:9000 --image_hw=256 --gpu=1

"""

from __future__ import absolute_import
from __future__ import unicode_literals

import BaseHTTPServer
import CGIHTTPServer
import functools
import json
import os
import shutil
import time
from cgi import parse_header, parse_multipart
from urlparse import parse_qs

import numpy as np
import scipy.misc
import tensorflow as tf

import util_io
from interface import face_detection_util
from interface import interface_utils
from interface import twingan_client
from interface import waifu2x_interface

tf.flags.DEFINE_integer('gpu', -1,
                        'GPU ID (negative value indicates CPU)')
tf.flags.DEFINE_integer('port', 8000, 'Port of this server.')
tf.flags.DEFINE_string('host', '', 'Host of this server. Defaults to localhost')
tf.flags.DEFINE_integer('max_num_faces', 4, 'Maximum number of faces to detect and translate.')
tf.flags.DEFINE_boolean('debug', False, 'If true, use the mock twingan client for debugging.')

FLAGS = tf.flags.FLAGS

FACE_DETECTOR = face_detection_util.FaceDetector()
DT_CLIENT = twingan_client.TwinGANClient('', None)
WAIFU2X = waifu2x_interface.Waifu2x()


class MyHandler(CGIHTTPServer.CGIHTTPRequestHandler):
  def __init__(self, req, client_addr, server):
    CGIHTTPServer.CGIHTTPRequestHandler.__init__(self, req, client_addr, server)

  def parse_POST(self):
    ctype, pdict = parse_header(self.headers['content-type'])
    pdict['boundary'] = str(pdict['boundary']).encode('utf-8')
    if ctype == 'multipart/form-data':
      postvars = parse_multipart(self.rfile, pdict)
    elif ctype == 'application/x-www-form-urlencoded':
      length = int(self.headers['content-length'])
      postvars = parse_qs(
        self.rfile.read(length),
        keep_blank_values=1)
    else:
      postvars = {}
    return postvars

  def do_POST(self):
    # TODO: refactor this file. A lot of asynchronous functions are hacky and ugly.
    form = self.parse_POST()

    if 'id' in form:
      id_str = form['id'][0]
      id_str = id_str.decode()
    else:
      id_str = 'test'

    if 'register_download' in form and form['register_download']:
      if 'subid' in form:
        subid_str = form['subid'][0]
        subid_str = subid_str.decode()
      else:
        subid_str = '0'
      print('TODO: do action for register download for id: %s and subid: %s' % (id_str, subid_str))
      self.post_success(id_str, )
      return

    if 'detectFace' in form and form['detectFace']:
      if 'image' not in form:
        self.post_server_internal_error('Missing image for detectFace mode.', id_str, {})
        return
      bin1 = form['image'][0]
      image_np = interface_utils.base64_to_numpy(bin1,contains_format=True)
      image_face_marked, face_found = FACE_DETECTOR.mark_face(image_path=None, image_np=image_np)
      self.post_success(id_str, {'image': interface_utils.numpu_to_base64(image_face_marked), 'face_found': face_found})
      return


    elif 'image' in form:
      bin1 = form['image'][0]
      input_image_path = interface_utils.save_encoded_image(bin1, './static/images/inputs/' + id_str)

      cropped_image_pattern = './static/images/cropped_faces/' + id_str + '_%d.png'
      faces = FACE_DETECTOR.crop_face_and_save(input_image_path, cropped_image_pattern)
      num_faces = len(faces)
      if num_faces > FLAGS.max_num_faces:
        faces = faces[FLAGS.max_num_faces]
        num_faces = FLAGS.max_num_faces
      if num_faces == 0:
        if 'failOnMissingFace' in form and form['failOnMissingFace']:
          self.post_success(id_str, {'face_found': False})
          return
        shutil.copy(input_image_path, './static/images/cropped_faces/' + id_str + '_%d.png' % 0)
        faces = [util_io.imread(input_image_path, (FLAGS.image_hw, FLAGS.image_hw))]
        num_faces = len(faces)

      transferred_image_file_format = './static/images/transferred_faces/' + id_str + '_%d.png'
      succeed, transferred_image_files = self.automatic_retry(functools.partial(
        self.domain_transfer, transferred_image_file_format=transferred_image_file_format, images=faces))
      if not succeed:
        self.post_server_internal_error('Domain transfer failed', id_str, {'num_faces': num_faces})
        return

      if 'do_waifu2x' in form:
        do_waifu2x = form['do_waifu2x'][0] == 'true'
      else:
        do_waifu2x = False
      if do_waifu2x:
        transferred_2x_image_file_format = './static/images/transferred_faces_2x/' + id_str + '_%d.png'
        succeed, transferred_2x_image_files = self.automatic_retry(functools.partial(
          self.call_waifu2x, transferred_image_file_format=transferred_image_file_format,
          transferred_2x_image_file_format=transferred_2x_image_file_format, num_images=num_faces))
        if not succeed:
          self.post_server_internal_error('Waifu2x failed', id_str, {'num_faces': num_faces})
          return
        transferred_image_to_be_combined_format = transferred_2x_image_file_format
      else:
        transferred_image_to_be_combined_format = transferred_image_file_format

      combined_image_pattern = './static/images/combined/' + id_str + '_%d.png'
      succeed, combined_images = self.automatic_retry(functools.partial(
        self.combine_original_and_transferred, images=faces, combined_image_pattern=combined_image_pattern,
        transferred_2x_image_file_format=transferred_image_to_be_combined_format))
      if not succeed:
        self.post_server_internal_error('Combine original and transferred failed.', id_str, {'num_faces': num_faces})
        return

      self.post_success(id_str, {'num_faces': num_faces, 'face_found': True})
    else:
      self.post_bad_request('Post request must contain image.', id_str)
    return

  def post_success(self, id_str, extra_kwargs=None):
    content = {
      'message': 'The command Completed Successfully',
      'Status': '200 OK',
      'success': True,
      'gpu': FLAGS.gpu,
      'id_str': id_str,
    }
    if extra_kwargs:
      content.update(extra_kwargs)
    content = json.dumps(content)
    self.send_response(200)
    self.send_header('Content-type', 'application/json')
    self.send_header('Content-Length', len(content))
    self.end_headers()
    self.wfile.write(content)

  def post_bad_request(self, message, id_str, extra_kwargs=None):
    content = {
      'message': message,
      'Status': '400 Bad Request',
      'success': False,
      'gpu': FLAGS.gpu,
      'id_str': id_str,
    }
    if extra_kwargs:
      content.update(extra_kwargs)
    content = json.dumps(content)
    self.send_response(400)
    self.send_header('Content-type', 'application/json')
    self.send_header('Content-Length', len(content))
    self.end_headers()
    self.wfile.write(content)

  def post_server_internal_error(self, error_message, id_str, extra_kwargs=None):
    content = {
      'message': 'Some internal server error occurred: Error is "%s". Please try again.' % error_message,
      'Status': '500 Internal Server Error',
      'success': False,
      'gpu': FLAGS.gpu,
      'id_str': id_str,
    }
    if extra_kwargs:
      content.update(extra_kwargs)
    content = json.dumps(content)
    self.send_response(500)
    self.send_header('Content-type', 'application/json')
    self.send_header('Content-Length', len(content))
    self.end_headers()
    self.wfile.write(content)

  def automatic_retry(self, func, num_tries=3):
    """Automatically reruns a function.

    Takes a function that returns a list of files as output. Check whether those files exists.
    If not, run the function again."""
    succeed = False
    paths = []
    while num_tries > 0 and not succeed:
      paths = func()
      succeed = True
      for path in paths:
        if not os.path.exists(path):
          succeed = False
          break
    return succeed, paths

  @staticmethod
  def domain_transfer(transferred_image_file_format, images, skip_existing_images=False):
    ret = []
    for i, image in enumerate(images):
      image_path = transferred_image_file_format % i
      ret.append(image_path)
      if skip_existing_images and os.path.exists(image_path):
        continue
      DT_CLIENT.do_inference(image_path, image_np=image)
      DT_CLIENT.block_on_callback(image_path)
    return ret

  @staticmethod
  def call_waifu2x(transferred_image_file_format, transferred_2x_image_file_format, num_images):
    ret = []
    for i in range(num_images):
      image_path = transferred_2x_image_file_format % i
      ret.append(image_path)
      if os.path.exists(image_path):
        continue
      WAIFU2X.post_request(transferred_image_file_format % i, image_path)
    return ret

  @staticmethod
  def combine_original_and_transferred(images, transferred_2x_image_file_format, combined_image_pattern):
    ret = []
    for i in range(len(images)):
      save_image_path = combined_image_pattern % i
      ret.append(save_image_path)
      if os.path.exists(save_image_path):
        continue
      start_time = time.time()
      transferred_2x_image_file_path = transferred_2x_image_file_format % i
      while not os.path.exists(transferred_2x_image_file_path) and time.time() - start_time < 5:
        time.sleep(1)
      transferred_image = None
      while time.time() - start_time < 5:
        try:
          transferred_image = util_io.imread(transferred_2x_image_file_path, )
        except IOError:
          time.sleep(1)
      if transferred_image is None:
        raise IOError('Cannot read image file %s' % (transferred_2x_image_file_path))
      face_image = scipy.misc.imresize(images[i], (transferred_image.shape[0], transferred_image.shape[1]))
      combined_image = np.concatenate((face_image, transferred_image), axis=1)
      util_io.imsave(save_image_path, combined_image)
    return ret


def main(_):
  global DT_CLIENT
  if FLAGS.debug:
    DT_CLIENT = twingan_client.MockTwinGANClient(
      FLAGS.twingan_server, FLAGS.image_hw,
      concurrency=FLAGS.concurrency)
  else:
    DT_CLIENT = twingan_client.TwinGANClient(
      FLAGS.twingan_server, FLAGS.image_hw,
      concurrency=FLAGS.concurrency)

  print 'GPU: {}'.format(FLAGS.gpu)
  httpd = BaseHTTPServer.HTTPServer((FLAGS.host, FLAGS.port), MyHandler)
  print 'serving at', FLAGS.host, ':', FLAGS.port
  httpd.serve_forever()


if __name__ == '__main__':
  tf.app.run()
