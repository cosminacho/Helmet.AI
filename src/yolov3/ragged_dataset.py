import tensorflow as tf
import numpy as np
from absl.flags import FLAGS


@tf.function
def transform_targets_for_output(y_true, grid_size, anchor_idxs):
    # y_true: (batch_size, (nbboxes), (x1, y1, x2, y2, class, best_anchor))
    N = y_true.nrows()

    # y_true_out: (N, grid, grid, anchors, [x, y, w, h, obj, class])
    y_true_out = tf.zeros(
        (N, grid_size, grid_size, tf.shape(anchor_idxs)[0], 6))

    anchor_idxs = tf.cast(anchor_idxs, tf.int32)

    indexes = tf.TensorArray(tf.int32, 1, dynamic_size=True)
    updates = tf.TensorArray(tf.float32, 1, dynamic_size=True)
    idx = 0
    for i in tf.range(N):
        for j in tf.range(y_true[i].nrows()):

            anchor_eq = tf.equal(
                anchor_idxs, tf.cast(y_true[i][j][5], tf.int32))

            if tf.reduce_any(anchor_eq):
                box = y_true[i][j][0:4]
                box_xy = (y_true[i][j][0:2] + y_true[i][j][2:4]) / 2

                anchor_idx = tf.cast(tf.where(anchor_eq), tf.int32)
                grid_xy = tf.cast(box_xy // (1/grid_size), tf.int32)

                # grid[y][x][anchor] = (tx, ty, bw, bh, obj, class)
                indexes = indexes.write(
                    idx, [i, grid_xy[1], grid_xy[0], anchor_idx[0][0]])
                updates = updates.write(
                    idx, [box[0], box[1], box[2], box[3], 1, y_true[i][j][4]])
                idx += 1

    # tf.print(indexes.stack())
    # tf.print(updates.stack())

    return tf.tensor_scatter_nd_update(
        y_true_out, indexes.stack(), updates.stack())


def transform_targets(y_train, anchors, anchor_masks, size):
    y_outs = []
    grid_size = size // 32

    y_train = y_train.merge_dims(1, 2)
    # calculate anchor index for true boxes
    anchors = tf.cast(anchors, tf.float32)
    anchor_area = anchors[..., 0] * anchors[..., 1]
    box_wh = y_train[..., 2:4] - y_train[..., 0:2]
    box_wh = tf.tile(tf.expand_dims(box_wh, -2),
                     (1, 1, tf.shape(anchors)[0], 1))
    box_area = box_wh[..., 0] * box_wh[..., 1]
    intersection = tf.minimum(box_wh[..., 0], anchors[..., 0]) * \
        tf.minimum(box_wh[..., 1], anchors[..., 1])
    iou = intersection / (box_area + anchor_area - intersection)

    # tf.ragged.argmax is not ready, some dirty code
    anchor_idx_list = []
    for row_index in range(FLAGS.batch_size):
        iou_single = iou[row_index]
        anchor_idx = tf.cast(tf.argmax(iou_single, axis=-1), tf.float32)
        anchor_idx = tf.expand_dims(anchor_idx, axis=-1)
        anchor_idx_list.append(anchor_idx)
    anchor_idx = tf.ragged.stack(anchor_idx_list)

    y_train = tf.concat([y_train, anchor_idx], axis=-1)

    for anchor_idxs in anchor_masks:
        y_outs.append(transform_targets_for_output(
            y_train, grid_size, anchor_idxs))
        grid_size *= 2

    return tuple(y_outs)


IMAGE_FEATURE_MAP = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "xmins": tf.io.VarLenFeature(tf.float32),
    "ymins": tf.io.VarLenFeature(tf.float32),
    "xmaxs": tf.io.VarLenFeature(tf.float32),
    "ymaxs": tf.io.VarLenFeature(tf.float32),
    "classes": tf.io.VarLenFeature(tf.int64)
}


def transform_images(x_train, size, pad=False, augment=False):
    if pad:
        x_train = tf.image.resize_with_pad(
            x_train, size, size, method='bicubic', antialias=True)
    else:
        x_train = tf.image.resize(x_train, (size, size),
                                  method='bicubic', antialias=True)
    if augment:
        x_train = augment_image(x_train)
    x_train = x_train / 255
    return x_train


def augment_image(image):
    choice = np.random.randint(4)
    if choice == 0:
        image = tf.image.random_brightness(image, 0.05)
    elif choice == 1:
        image = tf.image.random_contrast(image, 0.75, 1.25)
    elif choice == 2:
        image = tf.image.random_hue(image, 0.01)
    else:
        image = tf.image.random_saturation(image, 0.75, 1.5)
    return image


def parse_tfrecord(tfrecord, size, image_type):
    x = tf.io.parse_single_example(tfrecord, IMAGE_FEATURE_MAP)

    if image_type == 'png':
        x_train = tf.image.decode_png(x['image'], channels=3)
    elif image_type == 'jpg':
        x_train = tf.image.decode_jpeg(x['image'], channels=3)

    x_train = tf.image.resize(x_train, (size, size),
                              method='bicubic', antialias=True)

    y_train = tf.stack([tf.sparse.to_dense(x['xmins']),
                        tf.sparse.to_dense(x['ymins']),
                        tf.sparse.to_dense(x['xmaxs']),
                        tf.sparse.to_dense(x['ymaxs']),
                        tf.cast(tf.sparse.to_dense(x['classes']),
                                tf.float32)], axis=1)

    y_train = tf.RaggedTensor.from_row_splits(
        y_train, [0, tf.shape(y_train)[0]])

    return x_train, y_train


def load_tfrecord_dataset(file_pattern, size, image_type):
    files = tf.data.Dataset.list_files(file_pattern)
    dataset = files.flat_map(tf.data.TFRecordDataset)

    return dataset.map(lambda x: parse_tfrecord(x, size, image_type))
