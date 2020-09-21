import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import numpy as np
import tensorflow as tf
from yolov3.models import (
    YoloV3, YoloV3Tiny
)
from yolov3.dataset import transform_images


flags.DEFINE_string('weights', 'weights/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_string('output', 'weights/yolov3.tflite',
                    'path to saved_model')
flags.DEFINE_string('classes', 'datasets/classes.txt', 'path to classes file')
flags.DEFINE_string('image', 'datasets/test/test1.jpg', 'path to input image')
flags.DEFINE_integer('num_classes', 7, 'number of classes in the model')
flags.DEFINE_integer('size', 416, 'image size')


def main(_argv):
    if FLAGS.tiny:
        yolo = YoloV3Tiny(size=FLAGS.size, classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(size=FLAGS.size, classes=FLAGS.num_classes)

    yolo.load_weights(FLAGS.weights)
    logging.info('weights loaded')

    converter = tf.lite.TFLiteConverter.from_keras_model(yolo)
    tflite_model = converter.convert()
    open(FLAGS.output, 'wb').write(tflite_model)
    logging.info("model saved to: {}".format(FLAGS.output))

    interpreter = tf.lite.Interpreter(model_path=FLAGS.output)
    interpreter.allocate_tensors()
    logging.info('tflite model loaded')

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    img = tf.image.decode_image(open(FLAGS.image, 'rb').read(), channels=3)
    img = tf.expand_dims(img, 0)
    img = transform_images(img, 416)

    t1 = time.time()
    outputs = interpreter.set_tensor(input_details[0]['index'], img)

    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])

    print(output_data)


if __name__ == '__main__':
    app.run(main)
