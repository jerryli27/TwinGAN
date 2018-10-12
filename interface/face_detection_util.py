"""Contains functions for face detection. Adapted from tensorflow-face-detection repo.

https://github.com/yeephycho/tensorflow-face-detection
"""
import time
import numpy as np
import tensorflow as tf

from interface import label_map_util
from interface.object_detection_lib import visualization_utils
import util_io
import util_misc

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = '../model/frozen_inference_graph_face.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = './protos/face_label_map.pbtxt'

NUM_CLASSES = 2
MIN_SCORE_THRESH = 0.7


def crop_by_category(image,
                     boxes,
                     classes,
                     scores,
                     category_index,
                     category_to_crop,
                     max_boxes_to_draw=20,
                     min_score_thresh=MIN_SCORE_THRESH,
                     left_w_ratio=0.5, right_w_ratio=0.5, top_h_ratio=1.0, bottom_h_ratio=0.3):
  """Crop the given image with boxes and return cropped images.

  Args:
    image: uint8 numpy array with shape (img_height, img_width, 3)
    boxes: a numpy array of shape [N, 4]
    classes: a numpy array of shape [N]
    scores: a numpy array of shape [N] or None.  If scores=None, then
      this function assumes that the boxes to be plotted are groundtruth
      boxes and plot all boxes as black with no classes or scores.
    category_index: a dict containing category dictionaries (each holding
      category index `id` and category name `name`) keyed by category indices.
    instance_masks: a numpy array of shape [N, image_height, image_width], can
      be None
    keypoints: a numpy array of shape [N, num_keypoints, 2], can
      be None
    use_normalized_coordinates: whether boxes is to be interpreted as
      normalized coordinates or not.
    max_boxes_to_draw: maximum number of boxes to visualize.  If None, draw
      all boxes.
    min_score_thresh: minimum score threshold for a box to be visualized
    left_w_ratio: See unevenly_expand_xywh(). Defaults are set empirically.
    right_w_ratio: See unevenly_expand_xywh(). Defaults are set empirically.
    top_h_ratio: See unevenly_expand_xywh(). Defaults are set empirically.
    bottom_h_ratio: See unevenly_expand_xywh(). Defaults are set empirically.
  Returns:
    a list of numpy arrays
  """
  assert category_to_crop in category_index
  if not max_boxes_to_draw:
    max_boxes_to_draw = boxes.shape[0]
  num_boxes = 0
  img_height, img_width = image.shape[0], image.shape[1]
  i = 0
  ret = []
  while num_boxes < min(max_boxes_to_draw, boxes.shape[0]) and i < min(max_boxes_to_draw, boxes.shape[0]):
    if (scores is None or scores[i] > min_score_thresh) and classes[i] == category_to_crop:
      box = tuple(boxes[i].tolist())
      ymin, xmin, ymax, xmax = box
      ymin_absolute, xmin_absolute, ymax_absolute, xmax_absolute = int(ymin * img_height), int(xmin * img_width), int(
        ymax * img_height), int(xmax * img_width)
      expanded_xywh = util_misc.unevenly_expand_xywh(
        xmin_absolute, ymin_absolute, xmax_absolute - xmin_absolute, ymax_absolute - ymin_absolute, img_width,
        img_height,
        left_w_ratio=left_w_ratio, right_w_ratio=right_w_ratio, top_h_ratio=top_h_ratio, bottom_h_ratio=bottom_h_ratio)
      xmin_expanded, ymin_expanded = expanded_xywh[0], expanded_xywh[1]
      xmax_expanded, ymax_expanded = expanded_xywh[0] + expanded_xywh[2], expanded_xywh[1] + expanded_xywh[3]

      ret.append(image[ymin_expanded:ymax_expanded, xmin_expanded:xmax_expanded])
    elif scores[i] < min_score_thresh:
      break
    i += 1
  return ret


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
    (im_height, im_width, 3)).astype(np.uint8)


class FaceDetector(object):
  def __init__(self):
    self.detection_graph = tf.Graph()
    with self.detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    self.sess = tf.Session(graph=self.detection_graph, config=config)

    self.label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    self.categories = label_map_util.convert_label_map_to_categories(self.label_map, max_num_classes=NUM_CLASSES,
                                                                     use_display_name=True)
    self.category_index = label_map_util.create_category_index(self.categories)
    self.face_category_index = None
    for item in self.categories:
      if item['name'] == 'face':
        self.face_category_index = item['id']
        break
    assert self.face_category_index
    
  def _helper(self, image_path, image_np=None):
    """

    :param image_path: Path to an image with human faces.
    :param image_np: Optional numpy array containing image in [h,w,c] format. Overrides image_path.
    :return: cropped faces as a list of numpy arrays
    """
    if image_np is None:
      image_np = util_io.imread(image_path)
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
    classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
    # Actual detection.
    start_time = time.time()
    (boxes, scores, classes, num_detections) = self.sess.run(
      [boxes, scores, classes, num_detections],
      feed_dict={image_tensor: image_np_expanded})
    elapsed_time = time.time() - start_time
    print('Face cropping inference time cost: {}'.format(elapsed_time))
    
    return (image_np, boxes, scores, classes, num_detections)

  def crop_face(self, image_path, image_np=None):
    """

    :param image_path: Path to an image with human faces.
    :param image_np: Optional numpy array containing image in [h,w,c] format. Overrides image_path.
    :return: cropped faces as a list of numpy arrays
    """
    (image_np, boxes, scores, classes, num_detections) = self._helper(image_path, image_np=image_np)

    return crop_by_category(
      image_np,
      np.squeeze(boxes),
      np.squeeze(classes).astype(np.int32),
      np.squeeze(scores),
      self.category_index,
      category_to_crop=self.face_category_index
    )
  
  def mark_face(self, image_path, image_np=None):
    """Returns the image with boxes drawn around faces.

    :param image_path: Path to an image with human faces.
    :param image_np: Optional numpy array containing image in [h,w,c] format. Overrides image_path.
    :return: cropped faces as a list of numpy arrays
    """
    (image_np, boxes, scores, classes, num_detections) = self._helper(image_path, image_np=image_np)
    ret = np.copy(image_np)
    ret = visualization_utils.visualize_boxes_and_labels_on_image_array(
      ret,
      np.squeeze(boxes),
      np.squeeze(classes).astype(np.int32),
      np.squeeze(scores),
      self.category_index,
      min_score_thresh=MIN_SCORE_THRESH,
      use_normalized_coordinates=True,
    )
    face_found = bool(scores[0][0] >= MIN_SCORE_THRESH)
    return ret, face_found

  def crop_face_and_save(self, image_path, save_image_pattern):
    cropped = self.crop_face(image_path)
    for i, cropped_img in enumerate(cropped):
      util_io.imsave(save_image_pattern % (i), cropped_img)
    return cropped


def main(_):
  # Sanity check.
  # cd TwinGAN/interface
  # python face_detection_util.py
  fd = FaceDetector()
  fd.crop_face_and_save('../demo/web_interface_input/ew.jpg', '../demo/web_interface_input/ew_cropped_%d.jpg')


if __name__ == '__main__':
  tf.app.run()
