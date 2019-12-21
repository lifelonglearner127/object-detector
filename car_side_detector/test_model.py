import argparse
import pickle

import cv2
import imutils
import numpy as np

from object_detection.core import ObjectDetector, non_max_suppression
from object_detection.descriptors import HOG
from object_detection.utils import Conf


ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True, help="path to the configuration file")
ap.add_argument("-i", "--image", required=True, help="path to the image to be classified")
args = vars(ap.parse_args())

conf = Conf(args["conf"])

model = pickle.loads(open(conf["classifier_path"], "rb").read())
hog = HOG(orientations=conf["orientations"],
          pixels_per_cell=tuple(conf["pixels_per_cell"]),
          cells_per_block=tuple(conf["cells_per_block"]),
          normalize=conf["normalize"])
od = ObjectDetector(model, hog)

image = cv2.imread(args["image"])
image = imutils.resize(image, width=min(260, image.shape[1]))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
(boxes, probs) = od.detect(gray, conf["window_dim"], win_step=conf["window_step"],
                           pyramid_scale=conf["pyramid_scale"], min_prob=conf["min_probability"])
pick = non_max_suppression(np.array(boxes), probs, conf["overlap_thresh"])
orig = image.copy()

for (start_x, start_y, end_x, end_y) in boxes:
    cv2.rectangle(orig, (start_x, start_y), (end_x, end_y), (0, 0, 255), 2)

for (start_x, start_y, end_x, end_y) in pick:
    cv2.rectangle(image, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

cv2.imshow("Original", orig)
cv2.imshow("Image", image)
cv2.waitKey(0)
