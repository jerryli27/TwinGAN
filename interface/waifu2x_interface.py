"""To be used with https://github.com/nagadomi/waifu2x"""

import requests
import PIL.Image
import io

ART_STYLE = 'art'
PHOTO_STYLE = 'photo'
NOISE_ALLOWED_VALS = (-1, 0, 1, 2, 3)
SCALE_ALLOWED_VALS = (-1, 1, 2)

class Waifu2x(object):
  def __init__(self, url='http://localhost:8812'):
    self.session = requests.Session()
    self.url = url

  def post_request(self, file_name, output_file_name, style=ART_STYLE, noise=1, scale=2, ):
    if style not in (ART_STYLE, PHOTO_STYLE):
      raise ValueError('style must be one of ART_STYLE, PHOTO_STYLE')
    if noise not in NOISE_ALLOWED_VALS:
      raise ValueError('style must belong to NOISE_ALLOWED_VALS')
    if scale not in SCALE_ALLOWED_VALS:
      raise ValueError('style must belong to SCALE_ALLOWED_VALS')

    headers = {'User-Agent': 'Mozilla/5.0'}

    with open(file_name, 'rb') as f:

      files = {
        'file': f,
      }
      data = {
        'style': style,
        'noise': noise,
        'scale': scale,

      }

      response = self.session.post(self.url + '/api', headers=headers, files=files, data=data)
      if response.status_code == requests.codes.ok:
        image = PIL.Image.open(io.BytesIO(response.content))
        image.save(output_file_name)
        return True
      else:
        print(response.content)
        return False


if __name__ == '__main__':
  # Sanity check.
  # cd TwinGAN/
  # python interface/waifu2x_interface.py
  waifu2x = Waifu2x()
  waifu2x.post_request('./demo/inference_output/anime/CKP7F1CTFJYYYF2O1P04J671HJMYJU0G_0.png',
                       './demo/waifu2x_sanity_check.jpg',
                       ART_STYLE, 1, 2)