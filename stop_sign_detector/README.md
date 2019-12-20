## Stop sign detector
dlib object detector use HOG descriptor and SVM under the hood.

- Training the detector
```
python train_detector.py --class stop_sign_images --annotations stop_sign_annotations \
	--output output/stop_sign_detector.svm
```

- Test the detector
```
python test_detector.py --detector output/stop_sign_detector.svm --testing stop_sign_testing
```