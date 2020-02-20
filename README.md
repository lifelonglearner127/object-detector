# Object Detector
Object detection framework using Computer Vision and Machine Learning. I used HOG(Histogram of Oriented Gradient) as a image descriptor and Linear SVM(Support Vector Machine) as a classifier.

- [Concepts](#concepts)
- [Tips on training object detector](#tips-on-training-object-detector)
- [Examples](#examples)
    - [Stop sign detector](stop_sign_detector/README.md)
    - [Face detector](face_detector/README.md)
    - [Car side detector](car_side_detector/README.md)
- [FAQ](#faq)

## Concepts
1. Introduction to Object Detection
    - Object detection is the problem to not only identify the object in the image, but also to find its location.
    - Object detiection is hard because objects in real-world can exhibit substantial variations in viewpoint, scale, deformation, occlusion, backgournd clutter, and intra-class variation.
    - A good object detector should be robust to changes in these properties and still be able to detect the presence of the object, even under less-than ideal circumstances.

2. 6 Steps Framework
    - Sample p positive samples from your training data of the objects you want to detect, and extract HOG descriptors from these samples
    - Sample n negative samples from a negative training set that does not contain any of the objects you want to detect, and extract HOG descriptors from these samples as well. In practice n >> p.
    - Train a Linear Support Vector Machine on your positive and negative samples.
    - Apply hard-negative mining
    - Take the false-positive samples found during the hard-negative mining stage, sort them by their confidence and re-train your classifier using these hard-negative samples.
    - Your classifier is now trained and can be applied to your test dataset

## Tips on training object detector
- Take special care labeling your data
- Leverage parallel processing
- Keep in mind the image pyramid and sliding window tradeoff
- Tune descriptor hyperparameters
- Run experiments and log your results

## Examples
1. Stop sign detector using `dlib` train_simple_object_detector
2. Face detector using `dlib` train_simple_object_detector
3. Side car detector

    You can easily extend this project for your own custom object by modifying configuration file.

### Libraries
- OpenCV
- Scikit-learn
- Scikit-image
- dlib

### Project Structure

## FAQ
1. Doesn't OpenCV already come with Haar feature-based cascade classifiers? Why do we need our own detector?

    Yes, Haar classifiers shipped with OpenCV has much limitations as followings.
    - Haar cascades are extremly slow to train, taking days to work on even small datasets
    - Haar cascades tend to have an alarmingly high false-positive rate.
    - It tends not to detect an object that actually do exist due to sub-optimal parameter choices.
    - It can be especially challenging to tune, tweak, and dial in the optimal detection parameters; furthermore, the optimal parameters can vary on an image-to-image basis.

2. What is hard-negative mining?

    The trained classifier may incorrectly detect an object from negative example that actually doesn't contain the object. This means our classifier is underfitting over the training dataset. So in order for classifier to work properly (in order to get low false-positive rate), the classifier needs to be trained the training dataset plus hard-negative samples.

    For each image and each scale of each image in your negative training set, apply sliding windows technique and slide your window across the image. At each window, compute HOG descriptor and apply your classifier. If your classifier classifies a given window as an object, record the feature vector associated with the false-positive patch along with the probability of the classification.

3. What is Non-maxima suppression?

    When building object detection system, there is an inescapable issue you must handle - overlapping boxes. To handle the removal of bouding boxes(that refer to the same object), we can use either non-maxima suppression and Mean-shift algorithm.