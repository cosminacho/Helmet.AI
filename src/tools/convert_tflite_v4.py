import tensorflow as tf
from absl import app, flags, logging
from absl.flags import FLAGS
import numpy as np
import cv2
from core.yolov4 import YOLOv4, YOLOv3, YOLOv4_tiny, YOLOv3_tiny, decode
import core.utils as utils
import os
from core.config import cfg

flags.DEFINE_string(
    'weights', './weights/yolov4/yolov4_model_3.weights', 'path to weights file')
flags.DEFINE_string('output', None, 'path to output')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_integer('input_size', 512, 'path to output')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('quantize_mode', "int8",
                    'quantize mode (int8, float16, full_int8)')
# flags.DEFINE_string(
#     'dataset', "/media/user/Source/Data/coco_dataset/coco/5k.txt", 'path to dataset')


def representative_data_gen():
    fimage = open(FLAGS.dataset).read().split()
    for input_value in range(100):
        if os.path.exists(fimage[input_value]):
            original_image = cv2.imread(fimage[input_value])
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            image_data = utils.image_preprocess(
                np.copy(original_image), [FLAGS.input_size, FLAGS.input_size])
            img_in = image_data[np.newaxis, ...].astype(np.float32)
            print(input_value)
            yield [img_in]
        else:
            continue


def save_tflite():
    NUM_CLASS = len(utils.read_class_names(cfg.YOLO.CLASSES))
    input_layer = tf.keras.layers.Input(
        [FLAGS.input_size, FLAGS.input_size, 3])
    if FLAGS.tiny:
        if FLAGS.model == 'yolov3':
            feature_maps = YOLOv3_tiny(input_layer, NUM_CLASS)
        else:
            feature_maps = YOLOv4_tiny(input_layer, NUM_CLASS)
        bbox_tensors = []
        for i, fm in enumerate(feature_maps):
            bbox_tensor = decode(fm, NUM_CLASS, i)
            bbox_tensors.append(bbox_tensor)
        model = tf.keras.Model(input_layer, bbox_tensors)
        utils.load_weights_tiny(model, FLAGS.weights)
    else:
        if FLAGS.model == 'yolov3':
            feature_maps = YOLOv3(input_layer, NUM_CLASS)
            bbox_tensors = []
            for i, fm in enumerate(feature_maps):
                bbox_tensor = decode(fm, NUM_CLASS, i)
                bbox_tensors.append(bbox_tensor)
            model = tf.keras.Model(input_layer, bbox_tensors)
            utils.load_weights_v3(model, FLAGS.weights)
        elif FLAGS.model == 'yolov4':
            feature_maps = YOLOv4(input_layer, NUM_CLASS)
            bbox_tensors = []
            for i, fm in enumerate(feature_maps):
                bbox_tensor = decode(fm, NUM_CLASS, i)
                bbox_tensors.append(bbox_tensor)
            model = tf.keras.Model(input_layer, bbox_tensors)
            utils.load_weights(model, FLAGS.weights)
    model.summary()

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if tf.__version__ >= '2.2.0':
        converter.experimental_new_converter = False

    if FLAGS.quantize_mode == 'int8':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    elif FLAGS.quantize_mode == 'float16':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [
            tf.compat.v1.lite.constants.FLOAT16]
    elif FLAGS.quantize_mode == 'full_int8':
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        converter.allow_custom_ops = True
        converter.representative_dataset = representative_data_gen

    tflite_model = converter.convert()
    open(FLAGS.output, 'wb').write(tflite_model)
    logging.info("model saved to: {}".format(FLAGS.output))


def demo():
    interpreter = tf.lite.Interpreter(model_path=FLAGS.output)
    interpreter.allocate_tensors()
    logging.info('tflite model loaded')

    input_details = interpreter.get_input_details()
    print(input_details)
    output_details = interpreter.get_output_details()
    print(output_details)

    input_shape = input_details[0]['shape']

    input_data = np.array(np.random.random_sample(
        input_shape), dtype=np.float32)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = [interpreter.get_tensor(
        output_details[i]['index']) for i in range(len(output_details))]

    print(output_data)


def main(_argv):
    save_tflite()
    demo()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
