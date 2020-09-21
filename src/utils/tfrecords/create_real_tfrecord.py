import tensorflow as tf
import cv2
import os
import numpy as np

from yolov3.utils import BoundBox


def parse_real_annotations(image_filename, annotations_filename):
    image = cv2.imread(image_filename)
    img_height, img_width, _ = image.shape
    bbox = BoundBox(img_width, img_height)

    with open(annotations_filename) as f:
        annotations = [item.strip() for item in f.readlines()]

    boxes = []
    for annotation in annotations:
        annotation = annotation.split(" ")
        label = int(annotation[0])
        x_center = float(annotation[1])
        y_center = float(annotation[2])
        width = float(annotation[3])
        height = float(annotation[4])
        xmin, ymin, xmax, ymax = bbox.centroid_to_box(
            x_center, y_center, width, height)

        boxes.append([xmin, ymin, xmax, ymax, label])

    return boxes


def parse_real_dataset():
    with open("datasets/real_dataset/train.real.txt") as f:
        data = [item.strip() for item in f.readlines()]

    with open("datasets/real_dataset/valid.real.txt") as f:
        data.extend([item.strip() for item in f.readlines()])

    prefix = "datasets/real_dataset/"
    result = []
    for line in data:
        name = line[:-4]
        if name[-1] == ".":
            name = name[:-1]

        image_filename = prefix+line
        annotations_filename = prefix + name + '.txt'

        boxes = parse_real_annotations(image_filename, annotations_filename)
        result.append([image_filename, boxes])

    return result



def flip(index, image_filename, boxes):
    image = cv2.imread(image_filename)
    height, width, _ = image.shape
    flipped_image = cv2.flip(image, 1)
    flipped_boxes = []
    for box in boxes:
        flipped_box = [width-box[2], box[1], width-box[0], box[3], box[4]]
        flipped_boxes.append(flipped_box)

    flipped_image_filename = f"poze/{index}.jpg"
    cv2.imwrite(flipped_image_filename, flipped_image)

    return flipped_image_filename, flipped_boxes


def main():

    items = []

    items.extend(parse_real_dataset())

    flipped_items = []
    for index, item in enumerate(items):
        image_filename = item[0]
        boxes = item[1]
        flipped_image_filename, flipped_boxes = flip(
            index, image_filename, boxes)

        flipped_items.append([flipped_image_filename, flipped_boxes])

    items.extend(flipped_items)

    np.random.shuffle(items)
    items.reverse()
    np.random.shuffle(items)
    items.reverse()
    np.random.shuffle(items)

    with tf.io.TFRecordWriter("real_train.tfrecord") as writer:
        for i in range(320):
            item = items[i]
            sample = build_sample(item[0], item[1])
            writer.write(sample.SerializeToString())

    with tf.io.TFRecordWriter("real_valid.tfrecord") as writer:
        for i in range(320, len(items)):
            item = items[i]
            sample = build_sample(item[0], item[1])
            writer.write(sample.SerializeToString())


def build_sample(image_filename, boxes):
    raw_image = open(image_filename, 'rb').read()

    image = cv2.imread(image_filename)
    img_height, img_width, _ = image.shape
    xmins = []
    ymins = []
    xmaxs = []
    ymaxs = []
    classes = []
    for box in boxes:
        xmins.append(box[0] / img_width)
        ymins.append(box[1] / img_height)
        xmaxs.append(box[2] / img_width)
        ymaxs.append(box[3] / img_height)
        classes.append(box[4])

    example = tf.train.Example(features=tf.train.Features(feature={
        "xmins": tf.train.Feature(float_list=tf.train.FloatList(value=xmins)),
        "ymins": tf.train.Feature(float_list=tf.train.FloatList(value=ymins)),
        "xmaxs": tf.train.Feature(float_list=tf.train.FloatList(value=xmaxs)),
        "ymaxs": tf.train.Feature(float_list=tf.train.FloatList(value=ymaxs)),
        "classes": tf.train.Feature(int64_list=tf.train.Int64List(value=classes)),
        "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[raw_image]))
    }))
    return example


if __name__ == "__main__":
    main()
