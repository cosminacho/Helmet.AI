from absl import app, flags, logging
from absl.flags import FLAGS

import tensorflow as tf

from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard,
    TerminateOnNaN
)

from yolov3.models import (
    YoloV3, YoloV3Tiny, YoloLoss,
    yolo_anchors, yolo_anchor_masks,
    yolo_tiny_anchors, yolo_tiny_anchor_masks
)

from yolov3.utils import freeze_all

flags.DEFINE_string('prefix', '', 'prefix')
flags.DEFINE_string(
    'dataset', 'datasets/virtual_dataset/virtual_train_part_*.tfrecord', 'path to training dataset')
flags.DEFINE_string(
    'val_dataset', 'datasets/virtual_dataset/virtual_valid.tfrecord', 'path to validation dataset')
flags.DEFINE_string(
    'weights', 'weights/yolov3.tf', 'path to weights file')


flags.DEFINE_enum('mode', 'fit', ['fit', 'eager_fit', 'eager_tf'],
                  'fit: model.fit, '
                  'eager_fit: model.fit(run_eagerly=True), '
                  'eager_tf: custom GradientTape')
flags.DEFINE_enum('transfer', 'none',
                  ['none', 'yolo_darknet', 'yolo_conv', 'yolo_output_conv', 'all'],
                  'none: Training from scratch (no weights transfer), '
                  'yolo_darknet: Transfer darknet sub-model weights, '
                  'yolo_conv: Transfer darknet and conv sub-model weights, '
                  'yolo_output_conv: Transfer darknet and conv sub-model weights and first output conv layer weights, '
                  'all: Transfer all weights (pretrained weights need to have the same number of classes)')
flags.DEFINE_enum('freeze', 'none',
                  ['none', 'yolo_darknet', 'yolo_conv', 'yolo_output_conv', 'all'],
                  'none: Tune all weights, '
                  'yolo_darknet: Tune all but darknet sub-model weights, '
                  'yolo_conv: Tune output sub-model weights, '
                  'yolo_output_conv: Tune only output sub-model without the first conv layer, '
                  'all: Do not allow tuning of weights')


flags.DEFINE_bool('letter_box', False, 'resize with pad')
flags.DEFINE_boolean('ragged_dataset', False, 'use tf.ragged api or not')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')

flags.DEFINE_integer('size', 416, 'size of the input image')
flags.DEFINE_enum('image_type', 'png', [
                  'png', 'jpg'], 'image_type of the image')

flags.DEFINE_integer(
    'log_freq', 100, "frequency to save to logs (num of batches)")
flags.DEFINE_integer(
    'save_freq', 1000, 'checkpoint save frequency (num of batches)')
flags.DEFINE_integer('batch_size', 32, 'batch size')


flags.DEFINE_integer('initial_epoch', 0, 'initial_epoch')
flags.DEFINE_integer('epochs', 3, 'number of epochs')

flags.DEFINE_integer('num_classes', 7, 'number of classes in the model')
flags.DEFINE_integer('weights_num_classes', None, 'specify num class for `weights` file if different, '
                     'useful in transfer learning with different number of classes')

flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')


def main(_argv):
    print_flags()

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)

    if FLAGS.ragged_dataset:
        import yolov3.ragged_dataset as dataset
    else:
        import yolov3.dataset as dataset

    if FLAGS.tiny:
        anchors = yolo_tiny_anchors
        anchor_masks = yolo_tiny_anchor_masks
    else:
        anchors = yolo_anchors
        anchor_masks = yolo_anchor_masks

    ################################################### DATASET SETUP ###########################################
    if FLAGS.dataset:
        train_dataset = dataset.load_tfrecord_dataset(
            FLAGS.dataset, FLAGS.size, FLAGS.image_type)

    train_dataset = train_dataset.shuffle(buffer_size=512)
    train_dataset = train_dataset.batch(
        FLAGS.batch_size, drop_remainder=FLAGS.ragged_dataset)
    train_dataset = train_dataset.map(lambda x, y: (
        dataset.transform_images(x, FLAGS.size, augment=True),
        dataset.transform_targets(y, anchors, anchor_masks, FLAGS.size)))
    train_dataset = train_dataset.prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)

    if FLAGS.val_dataset:
        val_dataset = dataset.load_tfrecord_dataset(
            FLAGS.val_dataset, FLAGS.size, FLAGS.image_type)
    val_dataset = val_dataset.batch(
        FLAGS.batch_size, drop_remainder=FLAGS.use_ragged_dataset)
    val_dataset = val_dataset.map(lambda x, y: (
        dataset.transform_images(x, FLAGS.size, augment=False),
        dataset.transform_targets(y, anchors, anchor_masks, FLAGS.size)))

    ########################################## MODEL SETUP ##########################################################
    optimizer = tf.keras.optimizers.Adam(lr=FLAGS.learning_rate)
    loss = [YoloLoss(anchors[mask], classes=FLAGS.num_classes)
            for mask in anchor_masks]

    model = create_model()
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'],
                  run_eagerly=(FLAGS.mode == 'eager_fit'))

    save_freq = FLAGS.save_freq if FLAGS.save_freq > 0 else 'epoch'
    callbacks = [
        ReduceLROnPlateau(monitor='loss', verbose=1,
                          patience=1, factor=0.5, min_lr=1e-5),
        ReduceLROnPlateau(monitor='val_loss', verbose=1,
                          patience=3, factor=0.5, min_lr=1e-5),
        EarlyStopping(monitor='loss', patience=3, verbose=1),
        EarlyStopping(monitor='val_loss', patience=10, verbose=1),
        # ModelCheckpoint(FLAGS.prefix + 'checks/yolov3_train_{epoch}_loss_{loss:.2f}.tf',
        #                 monitor='loss', verbose=1, save_weights_only=True, save_freq=save_freq),
        ModelCheckpoint(FLAGS.prefix + 'checks/yolov3_train_{epoch}_val_loss_{val_loss:.2f}.tf',
                        monitor='val_loss', verbose=1, save_weights_only=True, save_freq='epoch'),
        TensorBoard(log_dir=FLAGS.prefix + 'logs',
                    write_graph=False, update_freq=FLAGS.log_freq),
        TerminateOnNaN()
    ]

    history = model.fit(train_dataset,
                        initial_epoch=FLAGS.initial_epoch,
                        epochs=FLAGS.epochs,
                        callbacks=callbacks,
                        validation_data=val_dataset)


def create_model():
    if FLAGS.tiny:
        model = YoloV3Tiny(FLAGS.size, training=True,
                           classes=FLAGS.num_classes)
    else:
        model = YoloV3(FLAGS.size, training=True,
                       classes=FLAGS.num_classes)

    # Configure the model for transfer learning
    if FLAGS.transfer != 'none':
        # if we need all weights, no need to create another model
        if FLAGS.transfer == 'all':
            model.load_weights(FLAGS.prefix + FLAGS.weights)

        # else, we need only some of the weights
        # create appropriate model_pretrained, load all weights and copy the ones we need
        else:
            if FLAGS.tiny:
                model_pretrained = YoloV3Tiny(
                    FLAGS.size, training=True, classes=FLAGS.weights_num_classes or FLAGS.num_classes)
            else:
                model_pretrained = YoloV3(FLAGS.size,
                                          training=True, classes=FLAGS.weights_num_classes or FLAGS.num_classes)
            # load pretrained weights
            model_pretrained.load_weights(FLAGS.prefix + FLAGS.weights)
            # transfer darknet
            model.get_layer('yolo_darknet').set_weights(model_pretrained.get_layer(
                'yolo_darknet').get_weights())
            # transfer 'yolo_conv_i' layer weights
            if FLAGS.transfer in ['yolo_conv', 'yolo_output_conv']:
                for l in model.layers:
                    if l.name.startswith('yolo_conv'):
                        model.get_layer(l.name).set_weights(
                            model_pretrained.get_layer(l.name).get_weights())
            # transfer 'yolo_output_i' first conv2d layer
            if FLAGS.transfer == 'yolo_output_conv':
                # transfer tiny output conv2d
                if FLAGS.tiny:
                    # get and set the weights of the appropriate layers
                    model.layers[4].layers[1].set_weights(
                        model_pretrained.layers[4].layers[1].get_weights())
                    model.layers[5].layers[1].set_weights(
                        model_pretrained.layers[5].layers[1].get_weights())
                    # should I freeze batch_norm as well?
                else:
                    # get and set the weights of the appropriate layers
                    model.layers[5].layers[1].set_weights(
                        model_pretrained.layers[5].layers[1].get_weights())
                    model.layers[6].layers[1].set_weights(
                        model_pretrained.layers[6].layers[1].get_weights())
                    model.layers[7].layers[1].set_weights(
                        model_pretrained.layers[7].layers[1].get_weights())
                    # should I freeze batch_norm as well?
    # no transfer learning
    else:
        pass

    # freeze layers, if requested
    if FLAGS.freeze != 'none':
        if FLAGS.freeze == 'all':
            freeze_all(model)
        if FLAGS.freeze in ['yolo_darknet' 'yolo_conv', 'yolo_output_conv']:
            freeze_all(model.get_layer('yolo_darknet'))
        if FLAGS.freeze in ['yolo_conv', 'yolo_output_conv']:
            for l in model.layers:
                if l.name.startswith('yolo_conv'):
                    freeze_all(l)
        if FLAGS.freeze == 'yolo_output_conv':
            if FLAGS.tiny:
                # freeze the appropriate layers
                freeze_all(model.layers[4].layers[1])
                freeze_all(model.layers[5].layers[1])
            else:
                # freeze the appropriate layers
                freeze_all(model.layers[5].layers[1])
                freeze_all(model.layers[6].layers[1])
                freeze_all(model.layers[7].layers[1])
    # freeze nothing
    else:
        pass

    return model


def print_flags():
    print("##################################################")

    print(f"prefix: {FLAGS.prefix}")
    print(f"dataset: {FLAGS.dataset}")
    print(f"val_dataset: {FLAGS.val_dataset}")
    print(f"weights: {FLAGS.weights}")

    print(f"image_type: {FLAGS.image_type}")
    print(f"size: {FLAGS.size}")

    print(f"initial_epoch: {FLAGS.initial_epoch}")
    print(f"epochs: {FLAGS.epochs}")

    print(f"learning_rate: {FLAGS.learning_rate}")

    print(f"tiny: {FLAGS.tiny}")
    print(f"letter_box: {FLAGS.letter_box}")
    print(f"ragged_dataset: {FLAGS.ragged_dataset}")

    print(f"batch_size: {FLAGS.batch_size}")

    print(f"num_classes: {FLAGS.num_classes}")
    print(f"weights_num_classes: {FLAGS.weights_num_classes}")

    print(f"log_freq: {FLAGS.log_freq}")
    print(f"save_freq: {FLAGS.save_freq}")

    print(f"mode: {FLAGS.mode}")
    print(f"transfer: {FLAGS.transfer}")
    print(f"freeze: {FLAGS.freeze}")

    print("##################################################")


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
