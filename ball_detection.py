import torch
# from torch.autograd import Variable
import onnx
from onnx_tf.backend import prepare
from torchvision import transforms

from edgetpu.detection.engine import DetectionEngine
from edgetpu.utils import dataset_utils

import torch.utils.data
from PIL import Image
import torchvision
import numpy as np

def get_instance_segmentation_model():
  engine = DetectionEngine("models/ssd_mobilenet_v1_coco_quant_postprocess_edgetpu.tflite")
  return engine

def get_pred(img_frame, model):
  img = Image.fromarray(img_frame)
  prediction = model.detect_with_image(img,
                                  threshold=0.05,
                                  keep_aspect_ratio="store_true",
                                  relative_coord=False,
                                  top_k=10)

  return prediction
objectlist = [36, 52, 73, 84, 33, 54, 9]

def ball_tracking_algo(img_frame, model):
  prediction = get_pred(img_frame, model)

  for p in prediction:
    print("id: ", p.label_id, "score =", p.score)
    if p.label_id in objectlist:
        #print(p.bounding_box)
        box = p.bounding_box.flatten().tolist()
        return True, box

  return False, None
