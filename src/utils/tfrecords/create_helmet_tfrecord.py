import os

import lxml.etree

import tensorflow as tf
import numpy as np
import cv2


def parse_xml(xml):
    if not len(xml):
        return {xml.tag: xml.text}
    result = {}
    for child in xml:
        child_result = parse_xml(child)
        if child.tag != 'object':
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}


def extract_boxes_from_voc(voc_obj):

    boxes = []

    if 'object' in voc_obj['annotation']:
        for obj in voc_obj['annotation']['object']:

            xmin = int(obj['bndbox']['xmin'])
            ymin = int(obj['bndbox']['ymin'])
            xmax = int(obj['bndbox']['xmax'])
            ymax = int(obj['bndbox']['ymax'])
            if obj['name'] in ['person', 'head', 'none']:
                obj_class = 0
            elif obj['name'] in ['hat', 'helmet', 'red', 'yellow', 'white', 'blue']:
                obj_class = 1
            # elif obj['name'] == 'red':
            #     obj_class = 1
            # elif obj['name'] == 'yellow':
            #     obj_class = 2
            # elif obj['name'] == 'blue':
            #     obj_class = 3
            # elif obj['name'] == 'white':
            #     obj_class = 4
            boxes.append([xmin, ymin, xmax, ymax, obj_class])

    return boxes


def flip_image(index, image_filename, boxes):
    image = cv2.imread(image_filename)
    height, width, _ = image.shape
    flipped_image = cv2.flip(image, 1)
    flipped_boxes = []
    for box in boxes:
        flipped_box = [width-box[2], box[1], width-box[0], box[3], box[4]]
        flipped_boxes.append(flipped_box)

    flipped_image_filename = f"flipped_images/{index}.jpg"
    cv2.imwrite(flipped_image_filename, flipped_image)
    return (flipped_image_filename, flipped_boxes)


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


def parse_voc_dataset(images_dir, annotations_dir):
    files = os.listdir(annotations_dir)
    result = []
    for filename in files:
        xml_string = lxml.etree.fromstring(
            open(annotations_dir + filename, encoding='utf-8').read())

        voc_obj = parse_xml(xml_string)
        boxes = extract_boxes_from_voc(voc_obj)
        image_filename = images_dir + filename[:-4] + '.jpg'
        result.append([image_filename, boxes])

    return result


def check_sample(image_filename, boxes):
    if not os.path.isfile(image_filename):
        return False

    image = cv2.imread(image_filename)
    height, width, channels = image.shape
    if channels != 3:
        return False

    for box in boxes:
        if box[4] not in [0, 1]:
            return False
        if (box[0] < box[2]) and (0 <= box[0]) and (box[2] <= width):
            pass
        else:
            return False
        if (box[1] < box[3]) and (0 <= box[1]) and (box[3] <= height):
            pass
        else:
            return False

    return True


def main():
    images_dir1 = "datasets/colored_helmet_dataset/JPEGImages/"
    images_dir2 = "datasets/helmet_dataset_1/JPEGImages/"
    images_dir3_1 = "datasets/helmet_dataset_2/Train/JPEGImage/"
    images_dir3_2 = "datasets/helmet_dataset_2/Test/JPEGImage/"

    annotations_dir1 = "datasets/colored_helmet_dataset/Annotations/"
    annotations_dir2 = "datasets/helmet_dataset_1/Annotations/"
    annotations_dir3_1 = "datasets/helmet_dataset_2/Train/Annotation/"
    annotations_dir3_2 = "datasets/helmet_dataset_2/Test/Annotation/"

    items = []
    items.extend(parse_voc_dataset(images_dir1, annotations_dir1))
    items.extend(parse_voc_dataset(images_dir2, annotations_dir2))
    items.extend(parse_voc_dataset(images_dir3_1, annotations_dir3_1))
    items.extend(parse_voc_dataset(images_dir3_2, annotations_dir3_2))

    flipped_items = []
    for index, item in enumerate(items):
        flipped_image_filename, flipped_boxes = flip_image(
            index+1, item[0], item[1])
        flipped_items.append([flipped_image_filename, flipped_boxes])

    items.extend(flipped_items)

    np.random.shuffle(items)
    items.reverse()
    np.random.shuffle(items)
    items.reverse()
    np.random.shuffle(items)

    with tf.io.TFRecordWriter("helmet_train.tfrecord") as writer:
        for i in range(len(items) - 800):
            item = items[i]
            sample = build_sample(item[0], item[1])
            writer.write(sample.SerializeToString())

    with tf.io.TFRecordWriter("helmet_valid.tfrecord") as writer:
        for i in range(len(items) - 800, len(items)):
            item = items[i]
            sample = build_sample(item[0], item[1])
            writer.write(sample.SerializeToString())


if __name__ == "__main__":
    main()
