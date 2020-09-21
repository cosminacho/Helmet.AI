import cv2
import numpy as np
import tensorflow as tf

prefix = "datasets/virtual_dataset/"

# input_filename = "train.annotations.txt"
# input_filename = "tune.tune.annotations.txt"
# input_filename = "valid.valid.annotations.txt"

# output_filename = "virtual_tune.tfrecord"
# output_filename = "virtual_valid.tfrecord"


old_width = 1088
old_height = 612

new_width = 640
new_height = 416


def build_sample(image_filename, annotations):
    image = cv2.imread(image_filename)
    image = cv2.resize(image, (new_width, new_height), cv2.INTER_CUBIC)
    cv2.imwrite("temp.png", image)
    raw_image = open("temp.png", 'rb').read()
    xmins = []
    ymins = []
    xmaxs = []
    ymaxs = []
    classes = []
    for annotation in annotations:
        items = annotation.split(",")
        items = [int(item) for item in items]
        xmins.append(items[0] / old_width)
        ymins.append(items[1] / old_height)
        xmaxs.append(items[2] / old_width)
        ymaxs.append(items[3] / old_height)
        classes.append(items[4])

    example = tf.train.Example(features=tf.train.Features(feature={
        "xmins": tf.train.Feature(float_list=tf.train.FloatList(value=xmins)),
        "ymins": tf.train.Feature(float_list=tf.train.FloatList(value=ymins)),
        "xmaxs": tf.train.Feature(float_list=tf.train.FloatList(value=xmaxs)),
        "ymaxs": tf.train.Feature(float_list=tf.train.FloatList(value=ymaxs)),
        "classes": tf.train.Feature(int64_list=tf.train.Int64List(value=classes)),
        "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[raw_image]))
    }))
    return example


def main():
    with open(prefix + input_filename) as f:
        data = f.readlines()
        data = [line.strip() for line in data]
    np.random.shuffle(data)
    data.reverse()
    np.random.shuffle(data)
    data.reverse()
    np.random.shuffle(data)

    with tf.io.TFRecordWriter(prefix + output_filename) as writer:
        for line in data:
            items = line.split(" ")
            image_filename = items[0]
            annotations = items[1:]
            tf_sample = build_sample(image_filename, annotations)
            writer.write(tf_sample.SerializeToString())

    # dict_writers = {}
    # for i in range(10):
    #     dict_writers[i] = tf.io.TFRecordWriter(
    #         prefix + f"virtual_train_part_{i}.tfrecord")

    # for index, line in enumerate(data):
    #     items = line.split(" ")
    #     image_filename = items[0]
    #     annotations = items[1:]
    #     tf_sample = build_sample(image_filename, annotations)
    #     dict_writers[index % 10].write(tf_sample.SerializeToString())

    # for i in range(10):
    #     dict_writers[i].close()


if __name__ == "__main__":
    main()
