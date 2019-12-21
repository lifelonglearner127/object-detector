import argparse
import cv2
import dlib
from imutils import paths


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--detector", required=True, help="Path to trained object detector")
ap.add_argument("-t", "--testing", required=True, help="Path to directory of testing images")
args = vars(ap.parse_args())

detector = dlib.simple_object_detector(args["detector"])

for testing_path in paths.list_images(args["testing"]):
    image = cv2.imread(testing_path)
    boxes = detector(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    for b in boxes:
        (x, y, w, h) = (b.left(), b.top(), b.right(), b.bottom())
        cv2.rectangle(image, (x, y), (w, h), (0, 255, 0), 2)

    cv2.imshow("Image", image)
    cv2.waitKey(0)
