import cv2
import imutils


def crop_ct101_bb(image, bb, padding=10, dst_size=(32, 32)):
    (y, h, x, w) = bb
    (x, y) = (max(x - padding, 0), max(y - padding, 0))
    roi = image[y:h+padding, x:w+padding]
    roi = cv2.resize(roi, dst_size, interpolation=cv2.INTER_AREA)
    return roi


def pyramid(image, scale=1.5, min_size=(30, 30)):
    yield image

    while True:
        w = int(image.shape[1] / scale)
        image = imutils.resize(image, width=w)

        if image.shape[0] < min_size[1] or image.shape[0] < min_size[0]:
            break

        yield image


def sliding_window(image, step_size, window_size):
    for y in range(0, image.shape[0], step_size):
        for x in range(0, image.shape[1], step_size):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])
