import tensorflow as tf
import numpy as np

IMAGE_FEATURE_MAP = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "xmins": tf.io.VarLenFeature(tf.float32),
    "ymins": tf.io.VarLenFeature(tf.float32),
    "xmaxs": tf.io.VarLenFeature(tf.float32),
    "ymaxs": tf.io.VarLenFeature(tf.float32),
    "classes": tf.io.VarLenFeature(tf.int64)
}


def count_records():
    files = tf.data.Dataset.list_files(
        f"datasets/virtual_dataset/virtual_train_part_*.tfrecord")
    dataset = files.flat_map(tf.data.TFRecordDataset)
    cnt_images = 0
    cnt_objects = 0
    for record in dataset:
        x = tf.io.parse_single_example(record, IMAGE_FEATURE_MAP)
        boxes = tf.sparse.to_dense(x['xmaxs'])
        cnt_objects += len(boxes)
        cnt_images += 1

    print(cnt_images)
    print(cnt_objects)


def main():
    count_records()

    # raw_dataset = tf.data.TFRecordDataset(
    #     "./datasets/virtual_dataset/virtual_train.tfrecord")

    # shards = 10

    # for i in range(shards):
    #     writer = tf.data.experimental.TFRecordWriter(
    #         f"./datasets/virtual_dataset/virtual_train-part-{i}.tfrecord")
    #     writer.write(raw_dataset.shard(shards, i))


if __name__ == "__main__":
    main()
