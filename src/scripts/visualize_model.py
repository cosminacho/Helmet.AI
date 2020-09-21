import tensorflow as tf
from absl import app
from yolov3.models import YoloV3, YoloV3Tiny

size = 416
classes = 7


def main(_argv):
    # yolo = YoloV3(size, classes=classes, training=False)
    yolo = YoloV3Tiny(size, classes=classes,  training=False)
    # tf.keras.utils.plot_model(yolo, to_file='model.png', show_shapes=True,
    #                           show_layer_names=True, rankdir='TB', expand_nested=False)
    print("\n\n\n")
    yolo.summary()
    # print("model generated")


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
