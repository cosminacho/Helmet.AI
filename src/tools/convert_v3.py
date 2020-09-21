from absl import app, flags, logging
from absl.flags import FLAGS

import tensorflow as tf
import numpy as np

from yolov3.models import YoloV3, YoloV3Tiny
from yolov3.utils import load_darknet_weights

flags.DEFINE_string('weights', './weights/tiny/yolov3-tiny_helmet.weights',
                    'path to weights file')
flags.DEFINE_string(
    'output', './weights/tiny/yolov3-tiny_helmet.tf', 'path to output')
flags.DEFINE_boolean('tiny', True, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('num_classes', 5, 'number of classes in the model')


def main(_argv):

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)
    yolo.summary()
    logging.info('model created')

    load_darknet_weights(yolo, FLAGS.weights, FLAGS.tiny)
    logging.info('weights loaded')

    img = np.random.random((1, 320, 320, 3)).astype(np.float32)
    output = yolo.predict(img)
    logging.info('sanity check passed')

    yolo.save_weights(FLAGS.output)
    logging.info('weights saved')


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
