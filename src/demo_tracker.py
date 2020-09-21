import time
from absl import app, flags, logging
from absl.flags import FLAGS

import cv2
import numpy as np
import tensorflow as tf

import core.utils as utils
from core.config import cfg
from core.yolov4 import YOLOv4, YOLOv3, YOLOv3_tiny, YOLOv4_tiny, decode

from yolov3.dataset import transform_images
from yolov3.models import YoloV3Tiny

from tensorflow.python.saved_model import tag_constants

import face_recognition

from deep_sort import generate_detections as gdet
from deep_sort import nn_matching
from deep_sort.tracker import Tracker
from deep_sort.detection import Detection

flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_integer('ppe_size', 512, 'resize images to')
flags.DEFINE_integer('helmet_size', 512, 'resize images to')

flags.DEFINE_string('video', './data/test/video1.mp4', 'path to input video')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID',
                    'codec used in VideoWriter when saving video to file')

flags.DEFINE_string('ppe_model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_boolean('ppe_tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string(
    'ppe_weights', './weights/yolov4/yolov4_model_3.weights', 'path to weights file')


flags.DEFINE_string('helmet_model', 'yolov3', 'yolov3 or yolov4')
flags.DEFINE_boolean('helmet_tiny', True, 'yolo or yolo-tiny')
flags.DEFINE_string(
    'helmet_weights', './weights/tiny/yolov3-tiny_helmet.tf', 'path to weights file')

STRIDES_PPE = np.array(cfg.YOLO.STRIDES)
XYSCALE_PPE = cfg.YOLO.XYSCALE
ANCHORS_PPE = utils.get_anchors(cfg.YOLO.ANCHORS, False)
CLASSES_PPE = utils.read_class_names(cfg.YOLO.CLASSES)
NUM_CLASSES_PPE = len(CLASSES_PPE)

CLASSES_HELMET = utils.read_class_names(cfg.YOLO.CLASSES_TINY)
NUM_CLASSES_HELMET = len(CLASSES_HELMET)


def create_ppe_detector(input_size):
    if FLAGS.framework == 'tf':
        input_layer = tf.keras.layers.Input([input_size, input_size, 3])
        feature_maps = YOLOv4(input_layer, NUM_CLASSES_PPE)
        bbox_tensors = []
        for i, fm in enumerate(feature_maps):
            bbox_tensor = decode(fm, NUM_CLASSES_PPE, i)
            bbox_tensors.append(bbox_tensor)
        model = tf.keras.Model(input_layer, bbox_tensors)
        utils.load_weights(model, FLAGS.ppe_weights)
        model.summary()
        return model
    elif FLAGS.framework == 'trt':
        saved_model_loaded = tf.saved_model.load(
            FLAGS.ppe_weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']
        return infer


def create_helmet_detector(input_size):
    yolo = YoloV3Tiny(input_size, classes=NUM_CLASSES_HELMET)
    yolo.load_weights(FLAGS.helmet_weights)
    yolo.summary()
    return yolo


def post_process_boxes(pred_bbox, model_type, frame_size, input_size):
    if model_type == 'yolov4':
        pred_bbox = utils.postprocess_bbbox(
            pred_bbox, ANCHORS_PPE, STRIDES_PPE, XYSCALE_PPE)
        bboxes = utils.postprocess_boxes(
            pred_bbox, frame_size, input_size, 0.5)  # 0.25
        bboxes = utils.nms(bboxes, 0.213, method='nms')  # 0.213
        return bboxes
    else:
        bboxes = []
        boxes, objectness, classes, nums = pred_bbox
        boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]
        wh = np.array([frame_size[1], frame_size[0]])
        for i in range(nums):
            x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
            x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
            bboxes.append([x1y1[0], x1y1[1], x2y2[0], x2y2[1],
                           objectness[i], int(classes[i])])
        return bboxes


def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)

    ppe_input_size = FLAGS.ppe_size
    helmet_input_size = FLAGS.helmet_size
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

    ppe_detector = create_ppe_detector(ppe_input_size)
    helmet_detector = create_helmet_detector(helmet_input_size)

    nacho_image1 = face_recognition.load_image_file("./data/faces/nacho1.jpg")
    nacho_image2 = face_recognition.load_image_file("./data/faces/nacho2.jpg")
    nacho_image3 = face_recognition.load_image_file("./data/faces/nacho3.jpg")

    nacho_face_encoding1 = face_recognition.face_encodings(nacho_image1)[0]
    nacho_face_encoding2 = face_recognition.face_encodings(nacho_image2)[0]
    nacho_face_encoding3 = face_recognition.face_encodings(nacho_image3)[0]

    known_face_encodings = [nacho_face_encoding1,
                            nacho_face_encoding2, nacho_face_encoding3]
    known_face_names = ["Nacho", "Nacho", "Nacho"]
    face_locations = []
    face_encodings = []
    face_names = []

    max_cosine_distance = 0.7  # 0.5 / 0.7
    nn_budget = None
    model_filename = './weights/tracker/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    Track_only = []

    logging.info("Models loaded!")
    while True:
        return_value, frame = vid.read()
        if not return_value:
            logging.warning("Empty Frame")
            break

        frame_size = frame.shape[: 2]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        img_in = tf.expand_dims(frame, 0)
        img_in = transform_images(img_in, helmet_input_size)

        image_data = utils.image_preprocess(
            np.copy(frame), [ppe_input_size, ppe_input_size])
        image_data = image_data[np.newaxis, ...].astype(np.float32)

        t1 = time.time()
        if FLAGS.framework == 'tf':
            ppe_pred_bbox = ppe_detector.predict(image_data)
        elif FLAGS.framework == 'trt':
            batched_input = tf.constant(image_data)
            ppe_pred_bbox = []
            result = ppe_detector(batched_input)
            for _, value in result.items():
                value = value.numpy()
                ppe_pred_bbox.append(value)

        helmet_pred_bbox = helmet_detector.predict(img_in)

        # face_locations = face_recognition.face_locations(small_frame)
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(
            frame, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(
                known_face_encodings, face_encoding)
            name = "Unknown"

            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]
            face_distances = face_recognition.face_distance(
                known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            face_names.append(name)

        t2 = time.time()
        times.append(t2 - t1)
        times = times[-20:]
        ms = sum(times)/len(times)*1000
        fps = 1000 / ms

        ppe_bboxes = post_process_boxes(
            ppe_pred_bbox, 'yolov4', frame_size, ppe_input_size)
        helmet_bboxes = post_process_boxes(
            helmet_pred_bbox, 'yolov3', frame_size, helmet_input_size)

        face_bboxes = []
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # top *= 4
            # left *= 4
            # right *= 4
            # bottom *= 4
            face_bboxes.append([left, top, right, bottom, name])
        bboxes = utils.calculate_status(ppe_bboxes, helmet_bboxes, [])

        boxes, safety_scores, site_roles, face_names = [], [], [], []
        for bbox in bboxes:
            boxes.append([bbox[0].astype(int), bbox[1].astype(int), bbox[2].astype(
                int)-bbox[0].astype(int), bbox[3].astype(int)-bbox[1].astype(int)])
            safety_scores.append(bbox[4])
            site_roles.append(bbox[5])
            face_names.append("None")
        for bbox in face_bboxes:
            boxes.append([bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]])
            safety_scores.append(0)
            site_roles.append(-1)
            face_names.append(bbox[4])

        boxes = np.array(boxes)
        safety_scores = np.array(safety_scores)
        site_roles = np.array(site_roles)
        face_names = np.array(face_names)
        features = np.array(encoder(frame, boxes))
        detections = [Detection(bbox, 0.9, 0, feature, safety_score, site_role, face_name) for bbox, feature,
                      safety_score, site_role, face_name in zip(boxes, features, safety_scores, site_roles, face_names)]

        tracker.predict()
        tracker.update(detections)

        tracked_bboxes = []
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:  # 1 / 5
                continue
            bbox = track.to_tlbr()

            tracking_id = track.track_id
            safety_score = track.get_safety_score()
            site_role = track.get_site_role()
            face_name = track.get_face_name()
            if site_role == -1:
                to_add = [face_name, site_role, tracking_id]
            else:
                to_add = [safety_score, site_role, tracking_id]

            tracked_bboxes.append(
                bbox.tolist() + to_add)
        image = utils.draw_demo(frame, tracked_bboxes)
        image = cv2.putText(image, "Time: {:.2f} FPS".format(
            fps), (0, 24), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

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
