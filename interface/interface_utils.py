"""Utility functions for web interface."""
import base64
import io
import os
import time
from io import BytesIO

import numpy as np
from PIL import Image

IMAGE_TYPE_DICT = {
  '.jpg': 'jpeg',
  '.jpeg': 'jpeg',
  '.png': 'png',
}


def get_image_encoding(path):
  ext = os.path.splitext(path)[1]
  image_type = IMAGE_TYPE_DICT[ext]
  encoding = None
  exponential_backoff = 0.05
  while not encoding and exponential_backoff < 1:
    with open(path, "rb") as image_file:
      encoding = base64.b64encode(image_file.read())
    if not encoding:
      time.sleep(exponential_backoff)
      exponential_backoff *= 2
  return 'data:image/' + image_type + ';base64,' + encoding


def save_encoded_image(encoded, path_without_extension):
  encoded = encoded.decode()
  extension = encoded.lstrip('data:image/').split(';')[0]
  encoded = encoded.split(',')[1]
  encoded = base64.b64decode(encoded.encode())
  path = path_without_extension + '.' + extension
  with open(path, 'wb') as fout1:
    fout1.write(encoded)
  return path


def base64_to_numpy(base64_text, contains_format=False):
  """if contains_format is true, the first part separated by a ',' is the image type."""
  ret = base64_text.decode()
  if contains_format:
    ret = ret.split(',')[1]
  ret = base64.b64decode(ret.encode())
  ret = np.asarray(Image.open(io.BytesIO(ret)))
  return ret

def numpu_to_base64(image_np, format="JPEG"):
  pil_img = Image.fromarray(image_np)
  buff = BytesIO()
  pil_img.save(buff, format=format)
  b64 = base64.b64encode(buff.getvalue()).decode("utf-8")
  ret = u'data:image/%s;base64,%s' %(format, b64)
  return ret