from . import helpers


class ObjectDetector:
    def __init__(self, model, desc):
        self.model = model
        self.desc = desc

    def detect(self, image, win_dim, win_step=4, pyramid_scale=1.5, min_prob=0.7):
        boxes = []
        probs = []
        for layer in helpers.pyramid(image, scale=pyramid_scale, min_size=win_dim):
            scale = image.shape[0] / float(layer.shape[0])
            for (x, y, window) in helpers.sliding_window(layer, win_step, win_dim):
                (win_height, win_width) = window.shape[:2]

                if win_height == win_dim[1] and win_width == win_dim[0]:
                    features = self.desc.describe(window).reshape(1, -1)
                    prob = self.model.predict_proba(features)[0][1]

                    if prob > min_prob:
                        (start_x, start_y) = (int(scale * x), int(scale * y))
                        end_x = int(start_x + (scale * win_width))
                        end_y = int(start_y + (scale * win_height))

                        boxes.append((start_x, start_y, end_x, end_y))
                        probs.append(prob)

        return (boxes, probs)
