import argparse
import random

import cv2
import numpy as np
import progressbar

from imutils import paths
from scipy import io
from sklearn.feature_extraction.image import extract_patches_2d

from object_detection.utils import Conf, dataset
from object_detection.descriptors import HOG
from object_detection.core import helpers


ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True,
                help="path to the configuration file")
args = vars(ap.parse_args())

conf = Conf(args["conf"])
hog = HOG(orientations=conf["orientations"],
          pixels_per_cell=tuple(conf["pixels_per_cell"]),
          cells_per_block=tuple(conf["cells_per_block"]),
          normalize=conf["normalize"])
data = []
labels = []

# grab the positive images
positive_sample_paths = list(paths.list_images(conf["image_positives"]))
positive_sample_paths = random.sample(
    positive_sample_paths,
    int(len(positive_sample_paths) * conf["percent_gt_images"])
)
print("[INFO] describing Positive samples ROIs")

widgets = ["Extracting: ", progressbar.Percentage(), " ", progressbar.Bar(),
           " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(positive_sample_paths),
                               widgets=widgets).start()

for (i, positive_sample_path) in enumerate(positive_sample_paths):
    image = cv2.imread(positive_sample_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_id = positive_sample_path[positive_sample_path.rfind("_") + 1:].replace(".jpg", "")
    p = f"{conf['image_annotations']}/annotation_{image_id}.mat"
    bb = io.loadmat(p)["box_coord"][0]
    roi = helpers.crop_ct101_bb(image, bb, padding=conf["offset"],
                                dstSize=tuple(conf["window_dim"]))
    rois = (roi, cv2.flip(roi, 1)) if conf["use_flip"] else (roi, )

    for roi in rois:
        features = hog.describe(roi)
        data.append(features)
        labels.append(1)

    pbar.update(i)
pbar.finish()

# grab the negative images
negative_sample_paths = list(paths.list_images(conf["image_distractions"]))
pbar = progressbar.ProgressBar(maxval=conf["num_distraction_images"], widgets=widgets).start()
print("[INFO] describing distraction ROIs...")

for i in np.arange(0, conf["num_distraction_images"]):
    image = cv2.imread(random.choice(negative_sample_paths))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    patches = extract_patches_2d(image, tuple(conf["window_dim"]), max_patches=conf["num_distractions_per_image"])

    for patch in patches:
        features = hog.describe(patch)
        data.append(features)
        labels.append(-1)

    pbar.update(i)
pbar.finish()
print("[INFO] dumping features and labels to file...")
dataset.dump_dataset(data, labels, conf["features_path"], "features")
