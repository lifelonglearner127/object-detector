import argparse
import pickle
import random

import cv2
import numpy as np
import progressbar

from imutils import paths

from object_detection.core import ObjectDetector
from object_detection.descriptors import HOG
from object_detection.utils import Conf, dataset


ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True, help="path to the configuration file")
args = vars(ap.parse_args())
conf = Conf(args["conf"])
data = []

model = pickle.loads(open(conf["classifier_path"], "rb").read())
hog = HOG(orientations=conf["orientations"],
          pixels_per_cell=tuple(conf["pixels_per_cell"]),
          cells_per_block=tuple(conf["cells_per_block"]),
          normalize=conf["normalize"])
od = ObjectDetector(model, hog)

distraction_paths = list(paths.list_images(conf["image_distractions"]))
distraction_paths = random.sample(distraction_paths, conf["hn_num_distraction_images"])

widgets = ["Mining: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(distraction_paths), widgets=widgets).start()

for (i, image_path) in enumerate(distraction_paths):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    (boxes, probs) = od.detect(gray, conf["window_dim"], win_step=conf["hn_window_step"],
                               pyramid_scale=conf["hn_pyramid_scale"], min_prob=conf["hn_min_probability"])

    for (prob, (start_x, start_y, end_x, end_y)) in zip(probs, boxes):
        roi = cv2.resize(gray[start_y:end_y, start_x:end_x], tuple(conf["window_dim"]), interpolation=cv2.INTER_AREA)
        features = hog.describe(roi)
        data.append(np.hstack([[prob], features]))
    pbar.update(i)

pbar.finish()
print("[INFO] sorting by probability...")
data = np.array(data)
data = data[data[:, 0].argsort()[::-1]]

print("[INFO] dumping hard negatives to file...")
dataset.dump_dataset(data[:, 1:], [-1] * len(data), conf["features_path"], "hard_negatives", method="a")
