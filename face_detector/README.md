## face detector

- Train detector
    ```
    python train_detector.py --xml faces_annotations.xml --detector detector.svm
    ```

- Test detector
    ```
    python test_detector.py --detector detector.svm --testing testing
    ```