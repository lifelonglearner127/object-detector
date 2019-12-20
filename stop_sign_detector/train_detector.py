import argparse

import dlib
from imutils import paths
from scipy.io import loadmat
from skimage import io

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--class", required=True,
                help="Path to the CALTECH-101 class images")
ap.add_argument("-a", "--annotations", required=True,
                help="Path to the CALTECH-101 class annotations")
ap.add_argument("-o", "--output", required=True,
                help="Path to the output detector")
args = vars(ap.parse_args())

print("[Info] gathering images and bounding boxes")
options = dlib.simple_object_detector_training_options()
images = []
boxes = []

for image_path in paths.list_images(args["class"]):
    image_id = image_path[image_path.rfind("/") + 1:].split("_")[1]
    image_id = image_id.replace(".jpg", "")
    p = f"{args['annotations']}/annotation_{image_id}.mat"
    annotations = loadmat(p)["box_coord"]

    bb = [dlib.rectangle(left=int(x), top=int(y), right=int(w), bottom=int(h))
          for (y, h, x, w) in annotations]
    boxes.append(bb)
    images.append(io.imread(image_path))

print("[INFO] training detector...")
detector = dlib.train_simple_object_detector(images, boxes, options)

print("[INFO] dumping classifier to file...")
detector.save(args["output"])

win = dlib.image_window()
win.set_image(detector)
dlib.hit_enter_to_continue()
