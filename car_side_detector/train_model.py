import argparse
import pickle

import numpy as np

from sklearn.svm import SVC

from object_detection.utils import dataset, Conf


ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True,
                help="path to the configuration file")
ap.add_argument("-n", "--hard-negatives", type=int, default=-1,
                help="flag indicating whether or not hard negatives should be used")
args = vars(ap.parse_args())

print("[INFO] loading dataset...")
conf = Conf(args["conf"])
(data, labels) = dataset.load_dataset(conf["features_path"], "features")

if args["hard_negatives"] > 0:
    (hard_data, hard_labels) = dataset.load_dataset(conf["features_path"], "hard_features")
    data = np.vstack([data, hard_data])
    labels = np.hstack([labels, hard_labels])

print("[INFO] training classifier")
model = SVC(kernel="linear", C=conf["C"], probability=True, random_state=42)
model.fit(data, labels)

print("[INFO] dumping classifier...")
with open(conf["classifier_path"], "wb") as f:
    f.write(pickle.dumps(model))
