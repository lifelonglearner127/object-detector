## Car side detector
This object detector is trained on CALTECH101 dataset.

- Feature Extraction
    ```
    python extract_features.py --conf conf/cars.json
    ```
- Train classifier without negative mining
    ```
    python train_model.py --conf conf/cars.json
    python test_model_no_nms.py --conf conf/cars.json --image datasets/caltech101/101_ObjectCategories/car_side/image_0016.jpg
    ```
- Non-maxima suppression
    ```
    python test_model_no_nms.py --conf conf/cars.json --image datasets/caltech101/101_ObjectCategories/car_side/image_0016.jpg
    ```
- Negative minig
    ```
    python hard_negative_mine.py --conf conf/cars.json
    ```
- Train classifier with negative mining
    ```
    python train_model.py --conf conf/cars.json --hard-negatives 1
    python test_model.py --conf conf/cars.json --image datasets/caltech101/101_ObjectCategories/car_side/image_0016.jpg
    ```
