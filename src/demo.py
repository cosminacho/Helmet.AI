import time
from absl import app, flags, logging
from absl.flags import FLAGS

import cv2
import numpy as np
import tensorflow as tf

import time
from absl import app, flags, logging
from absl.flags import FLAGS

import cv2
import numpy as np
import tensorflow as tf

import core.utils as utils
from core.config import cfg
from core.yolov4 import YOLOv4, YOLOv3, YOLOv3_tiny, YOLOv4_tiny, decode

from deep_sort import generate_detections as gdet
from deep_sort import nn_matching
from deep_sort.tracker import Tracker
from deep_sort.detection import Detection

from tensorflow.python.saved_model import tag_constants

import face_recognition

flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './weights/yolov4/yolov4_model_3.weights',
                    'path to weights file')
flags.DEFINE_integer('size', 512, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/test/video1.mp4', 'path to input video')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID',
                    'codec used in VideoWriter when saving video to file')


def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)

    if FLAGS.tiny:
        STRIDES = np.array(cfg.YOLO.STRIDES_TINY)
        ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS_TINY, FLAGS.tiny)
        XYSCALE = cfg.YOLO.XYSCALE_TINY
    else:
        STRIDES = np.array(cfg.YOLO.STRIDES)
        XYSCALE = cfg.YOLO.XYSCALE
        if FLAGS.model == 'yolov4':
            ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS, FLAGS.tiny)
        else:
            ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS_V3, FLAGS.tiny)

    CLASSES = utils.read_class_names(cfg.YOLO.CLASSES)
    NUM_CLASS = len(CLASSES)
    input_size = FLAGS.size
    try:
        vid = cv2.VideoCapture(int(FLAGS.video))
    except:
        vid = cv2.VideoCapture(FLAGS.video)

    times = []
    if FLAGS.output:
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    if FLAGS.framework == 'tf':
        input_layer = tf.keras.layers.Input([input_size, input_size, 3])
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
                if FLAGS.weights.split(".")[len(FLAGS.weights.split(".")) - 1] == "weights":
                    utils.load_weights(model, FLAGS.weights)
                else:
                    model.load_weights(FLAGS.weights).expect_partial()
        model.summary()
    elif FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
    elif FLAGS.framework == 'trt':
        saved_model_loaded = tf.saved_model.load(
            FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

    max_cosine_distance = 0.5  # 0.5 / 0.7
    nn_budget = None
    model_filename = './weights/tracker/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    key_list = list(CLASSES.keys())
    val_list = list(CLASSES.values())
    Track_only = ["person"]

    nacho_image = face_recognition.load_image_file("data/faces/nacho.jpg")
    nacho_face_encoding = face_recognition.face_encodings(nacho_image)[0]
    known_face_encodings = [nacho_face_encoding]
    known_face_names = ["Nacho"]

    logging.info("Models loaded!")

    while True:
        return_value, frame = vid.read()
        if not return_value:
            logging.warning("Empty Frame")
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_size = frame.shape[:2]
        image_data = utils.image_preprocess(
            np.copy(frame), [input_size, input_size])
        image_data = image_data[np.newaxis, ...].astype(np.float32)

        t1 = time.time()
        if FLAGS.framework == 'tf':
            pred_bbox = model.predict(image_data)
        elif FLAGS.framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred_bbox = [interpreter.get_tensor(
                output_details[i]['index']) for i in range(len(output_details))]
        elif FLAGS.framework == 'trt':
            batched_input = tf.constant(image_data)
            pred_bbox = []
            result = infer(batched_input)
            for _, value in result.items():
                value = value.numpy()
                pred_bbox.append(value)

        t2 = time.time()
        times.append(t2-t1)
        times = times[-20:]
        ms = sum(times)/len(times)*1000
        fps = 1000 / ms

        if FLAGS.model == 'yolov4':
            pred_bbox = utils.postprocess_bbbox(
                pred_bbox, ANCHORS, STRIDES, XYSCALE)
        else:
            pred_bbox = utils.postprocess_bbbox(pred_bbox, ANCHORS, STRIDES)
        bboxes = utils.postprocess_boxes(
            pred_bbox, frame_size, input_size, 0.5)  # 0.25
        bboxes = utils.nms(bboxes, 0.213, method='nms')  # 0.213

        bboxes = utils.calculate_safety(bboxes)

        # FACE RECOGNITION PART
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(
            frame, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(
                known_face_encodings, face_encoding)
            name = "Unknown"
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
            # face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            # best_match_index = np.argmin(face_distances)
            # if matches[best_match_index]:
            #     name = known_face_names[best_match_index]
            face_names.append(name)

        for bbox in bboxes:
            person_coor = np.array(bbox[:4], dtype=np.int32)
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                face_coor = np.array(
                    [left, top, right, bottom], dtype=np.int32)
                iou_score = utils.calculate_iou(person_coor, face_coor)
                if iou_score > 0.75:
                    bbox.append(name)
                    break
            if len(bbox) < 8:
                bbox.append("Unknown")

        boxes, scores, names, safety_scores, face_ids = [], [], [], [], []
        for bbox in bboxes:
            if len(Track_only) != 0 and CLASSES[int(bbox[5])] in Track_only or len(Track_only) == 0:
                boxes.append([bbox[0].astype(int), bbox[1].astype(int), bbox[2].astype(
                    int)-bbox[0].astype(int), bbox[3].astype(int)-bbox[1].astype(int)])
                scores.append(bbox[4])
                names.append(CLASSES[int(bbox[5])])
                safety_scores.append(int(bbox[6]))
                face_ids.append(bbox[7])

        boxes = np.array(boxes)
        names = np.array(names)
        scores = np.array(scores)
        features = np.array(encoder(frame, boxes))
        safety_scores = np.array(safety_scores)
        face_ids = np.array(face_ids)
        detections = [Detection(bbox, score, class_name, feature, face_name_id, safety_score) for bbox,
                      score, class_name, feature, face_name_id, safety_score in zip(boxes, scores, names, features, face_ids, safety_scores)]

        tracker.predict()
        tracker.update(detections)

        tracked_bboxes = []
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:  # 1 / 5
                continue
            bbox = track.to_tlbr()
            class_name = track.get_class()
            # tracking_id = track.track_id
            # index = key_list[val_list.index(class_name)]
            face_name_id = track.get_face_name()
            safety_score = track.get_safety_score()

            tracked_bboxes.append(bbox.tolist() + [face_name_id, safety_score])

        image = utils.draw_demo(frame, tracked_bboxes)
        # image = utils.draw_bbox(frame, tracked_bboxes,
        #                         classes=CLASSES, tracking=True)
        # image = cv2.putText(image, "Time: {:.2f} FPS".format(
        #     fps), (0, 24), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.namedWindow("Detections", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("Detections", image)
        if FLAGS.output:
            out.write(image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    if FLAGS.output:
        out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
