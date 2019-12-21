import cv2


def crop_ct101_bb(image, bb, padding=10, dstSize=(32, 32)):
    (y, h, x, w) = bb
    (x, y) = (max(x - padding, 0), max(y - padding, 0))
    roi = image[y:h+padding, x:w+padding]
    roi = cv2.resize(roi, dstSize, interpolation=cv2.INTER_AREA)
    return roi
