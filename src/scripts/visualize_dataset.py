

import cv2
import tensorflow as tf

from yolov3.dataset import load_tfrecord_dataset
from yolov3.utils import draw_outputs

from absl import app, flags
flags.DEFINE_integer('yolo_max_boxes', 421,
                     'maximum number of boxes per image')

input_filename = "helmet_train.tfrecord"


def main(_argv):
    class_names = [c.strip() for c in open("datasets/classes.txt").readlines()]

    dataset = load_tfrecord_dataset(input_filename, 416, 416, image_type='jpg')
    idx = 0
    for image, labels in dataset:
        idx += 1

        # boxes = []
        # scores = []
        # classes = []
        # for x1, y1, x2, y2, label in labels:
        #     if x1 == 0 and x2 == 0:
        #         continue
        #     boxes.append((x1, y1, x2, y2))
        #     scores.append(1)
        #     classes.append(label)

        # nums = [len(boxes)]
        # boxes = [boxes]
        # scores = [scores]
        # classes = [classes]
        # img = cv2.cvtColor(image.numpy(), cv2.COLOR_RGB2BGR)
        if idx % 100 == 0:
            print(idx)

        # img = cv2.resize(img, (1088, 612), interpolation=cv2.INTER_LANCZOS4)
        # img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
        # cv2.imwrite(f"buggy/{idx}.jpg", img)

    print(idx)


if __name__ == "__main__":
    app.run(main)
