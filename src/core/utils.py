import cv2
import numpy as np
import tensorflow as tf


def load_weights_tiny(model, weights_file, model_name):
    if model_name == 'yolov3':
        layer_size = 13
        output_pos = [9, 12]
    else:
        layer_size = 21
        output_pos = [17, 20]
    wf = open(weights_file, 'rb')
    major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)

    j = 0
    for i in range(layer_size):
        conv_layer_name = 'conv2d_%d' % i if i > 0 else 'conv2d'
        print(conv_layer_name)
        bn_layer_name = 'batch_normalization_%d' % j if j > 0 else 'batch_normalization'

        conv_layer = model.get_layer(conv_layer_name)
        filters = conv_layer.filters
        k_size = conv_layer.kernel_size[0]
        in_dim = conv_layer.input_shape[-1]

        if i not in output_pos:
            # darknet weights: [beta, gamma, mean, variance]
            bn_weights = np.fromfile(wf, dtype=np.float32, count=4 * filters)
            # tf weights: [gamma, beta, mean, variance]
            bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]
            bn_layer = model.get_layer(bn_layer_name)
            j += 1
        else:
            conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)

        # darknet shape (out_dim, in_dim, height, width)
        conv_shape = (filters, in_dim, k_size, k_size)
        conv_weights = np.fromfile(
            wf, dtype=np.float32, count=np.product(conv_shape))
        # tf shape (height, width, in_dim, out_dim)
        conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])

        if i not in output_pos:
            conv_layer.set_weights([conv_weights])
            bn_layer.set_weights(bn_weights)
        else:
            conv_layer.set_weights([conv_weights, conv_bias])

    assert len(wf.read()) == 0, 'failed to read all data'
    wf.close()


def load_weights_v3(model, weights_file):
    wf = open(weights_file, 'rb')
    major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)

    j = 0
    for i in range(75):
        conv_layer_name = 'conv2d_%d' % i if i > 0 else 'conv2d'
        bn_layer_name = 'batch_normalization_%d' % j if j > 0 else 'batch_normalization'

        conv_layer = model.get_layer(conv_layer_name)
        filters = conv_layer.filters
        k_size = conv_layer.kernel_size[0]
        in_dim = conv_layer.input_shape[-1]

        if i not in [58, 66, 74]:
            # darknet weights: [beta, gamma, mean, variance]
            bn_weights = np.fromfile(wf, dtype=np.float32, count=4 * filters)
            # tf weights: [gamma, beta, mean, variance]
            bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]
            bn_layer = model.get_layer(bn_layer_name)
            j += 1
        else:
            conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)

        # darknet shape (out_dim, in_dim, height, width)
        conv_shape = (filters, in_dim, k_size, k_size)
        conv_weights = np.fromfile(
            wf, dtype=np.float32, count=np.product(conv_shape))
        # tf shape (height, width, in_dim, out_dim)
        conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])

        if i not in [58, 66, 74]:
            conv_layer.set_weights([conv_weights])
            bn_layer.set_weights(bn_weights)
        else:
            conv_layer.set_weights([conv_weights, conv_bias])

    assert len(wf.read()) == 0, 'failed to read all data'
    wf.close()


def load_weights(model, weights_file):
    wf = open(weights_file, 'rb')
    major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)

    j = 0
    for i in range(110):
        conv_layer_name = 'conv2d_%d' % i if i > 0 else 'conv2d'
        bn_layer_name = 'batch_normalization_%d' % j if j > 0 else 'batch_normalization'

        conv_layer = model.get_layer(conv_layer_name)
        filters = conv_layer.filters
        k_size = conv_layer.kernel_size[0]
        in_dim = conv_layer.input_shape[-1]

        if i not in [93, 101, 109]:
            # darknet weights: [beta, gamma, mean, variance]
            bn_weights = np.fromfile(wf, dtype=np.float32, count=4 * filters)
            # tf weights: [gamma, beta, mean, variance]
            bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]
            bn_layer = model.get_layer(bn_layer_name)
            j += 1
        else:
            conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)

        # darknet shape (out_dim, in_dim, height, width)
        conv_shape = (filters, in_dim, k_size, k_size)
        conv_weights = np.fromfile(
            wf, dtype=np.float32, count=np.product(conv_shape))
        # tf shape (height, width, in_dim, out_dim)
        conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])

        if i not in [93, 101, 109]:
            conv_layer.set_weights([conv_weights])
            bn_layer.set_weights(bn_weights)
        else:
            conv_layer.set_weights([conv_weights, conv_bias])

    assert len(wf.read()) == 0, 'failed to read all data'
    wf.close()


def read_class_names(class_file_name):
    '''loads class name from a file'''
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names


def get_anchors(anchors_path, tiny=False):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = np.array(anchors.split(','), dtype=np.float32)
    if tiny:
        return anchors.reshape(2, 3, 2)
    else:
        return anchors.reshape(3, 3, 2)


def image_preprocess(image, target_size, gt_boxes=None):

    ih, iw = target_size
    h,  w, _ = image.shape

    scale = min(iw/w, ih/h)
    nw, nh = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih-nh) // 2
    image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized
    image_paded = image_paded / 255.

    if gt_boxes is None:
        return image_paded

    else:
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        return image_paded, gt_boxes


def draw_bbox(image, bboxes, classes=None, show_label=True, tracking=False):
    """
    bboxes: [x_min, y_min, x_max, y_max, probability, cls_id] format coordinates.
    """
    num_classes = len(classes)
    image_h, image_w, _ = image.shape

    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)
        fontScale = 0.9  # 0.9
        score = bbox[4]
        class_ind = int(bbox[5])
        if class_ind in [0, 4]:
            bbox_color = (255, 0, 0)
        else:
            bbox_color = (0, 255, 0)

        bbox_thick = int(1.5 * (image_h + image_w) / 600)  # 1.2
        c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

        if show_label:
            if tracking:
                bbox_mess = "%s" % (classes[class_ind])
            else:
                bbox_mess = '%s: %.2f' % (classes[class_ind], score)
            t_size = cv2.getTextSize(
                bbox_mess, 0, fontScale, thickness=bbox_thick//2)[0]  # // 2
            cv2.rectangle(
                image, (c1[0], c2[1]), (c1[0] + t_size[0], c2[1] - t_size[1] - 3), bbox_color, -1)  # filled
            cv2.putText(image, bbox_mess, (c1[0], c2[1]-2), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, (0, 0, 0), bbox_thick//2, lineType=cv2.LINE_AA)  # // 2

    return image


def bboxes_iou(boxes1, boxes2):

    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * \
        (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * \
        (boxes2[..., 3] - boxes2[..., 1])

    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:4], boxes2[..., 2:4])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    ious = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

    return ious


def bboxes_ciou(boxes1, boxes2):

    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    left = np.maximum(boxes1[..., 0], boxes2[..., 0])
    up = np.maximum(boxes1[..., 1], boxes2[..., 1])
    right = np.maximum(boxes1[..., 2], boxes2[..., 2])
    down = np.maximum(boxes1[..., 3], boxes2[..., 3])

    c = (right - left) * (right - left) + (up - down) * (up - down)
    iou = bboxes_iou(boxes1, boxes2)

    ax = (boxes1[..., 0] + boxes1[..., 2]) / 2
    ay = (boxes1[..., 1] + boxes1[..., 3]) / 2
    bx = (boxes2[..., 0] + boxes2[..., 2]) / 2
    by = (boxes2[..., 1] + boxes2[..., 3]) / 2

    u = (ax - bx) * (ax - bx) + (ay - by) * (ay - by)
    d = u/c

    aw = boxes1[..., 2] - boxes1[..., 0]
    ah = boxes1[..., 3] - boxes1[..., 1]
    bw = boxes2[..., 2] - boxes2[..., 0]
    bh = boxes2[..., 3] - boxes2[..., 1]

    ar_gt = bw/bh
    ar_pred = aw/ah

    ar_loss = 4 / (np.pi * np.pi) * (np.arctan(ar_gt) -
                                     np.arctan(ar_pred)) * (np.arctan(ar_gt) - np.arctan(ar_pred))
    alpha = ar_loss / (1 - iou + ar_loss + 0.000001)
    ciou_term = d + alpha * ar_loss

    return iou - ciou_term


def nms(bboxes, iou_threshold, sigma=0.3, method='nms'):
    """
    :param bboxes: (xmin, ymin, xmax, ymax, score, class)

    Note: soft-nms, https://arxiv.org/pdf/1704.04503.pdf
          https://github.com/bharatsingh430/soft-nms
    """
    classes_in_img = list(set(bboxes[:, 5]))
    best_bboxes = []

    for cls in classes_in_img:
        cls_mask = (bboxes[:, 5] == cls)
        cls_bboxes = bboxes[cls_mask]

        while len(cls_bboxes) > 0:
            max_ind = np.argmax(cls_bboxes[:, 4])
            best_bbox = cls_bboxes[max_ind]
            best_bboxes.append(best_bbox)
            cls_bboxes = np.concatenate(
                [cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
            iou = bboxes_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
            weight = np.ones((len(iou),), dtype=np.float32)

            assert method in ['nms', 'soft-nms']

            if method == 'nms':
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0

            if method == 'soft-nms':
                weight = np.exp(-(1.0 * iou ** 2 / sigma))

            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
            score_mask = cls_bboxes[:, 4] > 0.
            cls_bboxes = cls_bboxes[score_mask]

    return best_bboxes


def diounms_sort(bboxes, iou_threshold, sigma=0.3, method='nms', beta_nms=0.6):
    best_bboxes = []
    return best_bboxes


def postprocess_bbbox(pred_bbox, ANCHORS, STRIDES, XYSCALE=[1, 1, 1]):
    for i, pred in enumerate(pred_bbox):
        conv_shape = pred.shape
        output_size = conv_shape[1]
        conv_raw_dxdy = pred[:, :, :, :, 0:2]
        conv_raw_dwdh = pred[:, :, :, :, 2:4]
        xy_grid = np.meshgrid(np.arange(output_size), np.arange(output_size))
        xy_grid = np.expand_dims(
            np.stack(xy_grid, axis=-1), axis=2)  # [gx, gy, 1, 2]

        xy_grid = np.tile(tf.expand_dims(xy_grid, axis=0), [1, 1, 1, 3, 1])
        xy_grid = xy_grid.astype(np.float)

        # pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * STRIDES[i]
        pred_xy = ((tf.sigmoid(conv_raw_dxdy) *
                    XYSCALE[i]) - 0.5 * (XYSCALE[i] - 1) + xy_grid) * STRIDES[i]
        # pred_wh = (tf.exp(conv_raw_dwdh) * ANCHORS[i]) * STRIDES[i]
        pred_wh = (tf.exp(conv_raw_dwdh) * ANCHORS[i])
        pred[:, :, :, :, 0:4] = tf.concat([pred_xy, pred_wh], axis=-1)

    pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
    pred_bbox = tf.concat(pred_bbox, axis=0)
    return pred_bbox


def postprocess_boxes(pred_bbox, org_img_shape, input_size, score_threshold):

    valid_scale = [0, np.inf]
    pred_bbox = np.array(pred_bbox)

    pred_xywh = pred_bbox[:, 0:4]
    pred_conf = pred_bbox[:, 4]
    pred_prob = pred_bbox[:, 5:]

    # # (1) (x, y, w, h) --> (xmin, ymin, xmax, ymax)
    pred_coor = np.concatenate([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
                                pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)
    # # (2) (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
    org_h, org_w = org_img_shape
    resize_ratio = min(input_size / org_w, input_size / org_h)

    dw = (input_size - resize_ratio * org_w) / 2
    dh = (input_size - resize_ratio * org_h) / 2

    pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
    pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

    # # (3) clip some boxes those are out of range
    pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
    invalid_mask = np.logical_or(
        (pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
    pred_coor[invalid_mask] = 0

    # # (4) discard some invalid boxes
    bboxes_scale = np.sqrt(np.multiply.reduce(
        pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
    scale_mask = np.logical_and(
        (valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))

    # # (5) discard some boxes with low scores
    classes = np.argmax(pred_prob, axis=-1)
    scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
    # scores = pred_prob[np.arange(len(pred_coor)), classes]
    score_mask = scores > score_threshold
    mask = np.logical_and(scale_mask, score_mask)
    coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]

    return np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)


def freeze_all(model, frozen=True):
    model.trainable = not frozen
    if isinstance(model, tf.keras.Model):
        for l in model.layers:
            freeze_all(l, frozen)


def unfreeze_all(model, frozen=False):
    model.trainable = not frozen
    if isinstance(model, tf.keras.Model):
        for l in model.layers:
            unfreeze_all(l, frozen)


def calculate_isp(box1, box2):
    box1 = np.array(box1)
    box2 = np.array(box2)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    # box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    left_up = np.maximum(box1[:2], box2[:2])
    right_down = np.minimum(box1[2:4], box2[2:4])

    inter_section = np.maximum(right_down - left_up, 0)
    inter_area = inter_section[0] * inter_section[1]
    # union_area = box1_area + box2_area - inter_area

    isp = 1.0 * inter_area / box1_area
    return isp


def calculate_status(ppe_bboxes, helmet_boxes, face_boxes):
    helmet_boxes = np.array([h_box for h_box in helmet_boxes if h_box[5] != 4])

    result = []
    dictio = {"person": [], "head": [], "chest": []}
    for item in ppe_bboxes:
        class_ind = int(item[5])
        if class_ind == 6:
            dictio["person"].append(item)
        elif class_ind in [4, 5]:
            dictio["chest"].append(item)
        else:
            dictio["head"].append(item)

    for person_box in dictio["person"]:
        person_coor = np.array(person_box[:4], dtype=np.int32)
        safety_score = 0
        site_role = 4
        for head_box in dictio["head"]:
            head_coor = np.array(head_box[:4], dtype=np.int32)
            isp_score = calculate_isp(head_coor, person_coor)
            if isp_score >= 0.8:
                head_id = int(head_box[5])
                if head_id in [1, 2, 3]:
                    safety_score += 1
                    if head_id == 1 and len(helmet_boxes) != 0:
                        head_coors = np.expand_dims(head_coor, axis=0)
                        head_coors = np.repeat(
                            head_coors, len(helmet_boxes), axis=0)
                        ious = bboxes_iou(
                            head_coors, helmet_boxes[:, :4].astype(np.int32))

                        argmax_idx = np.argmax(ious)
                        site_role = int(helmet_boxes[argmax_idx][5])
                break

        for chest_box in dictio["chest"]:
            chest_coor = np.array(chest_box[:4], dtype=np.int32)
            isp_score = calculate_isp(chest_coor, person_coor)
            if isp_score >= 0.8:
                if int(chest_box[5]) == 5:
                    safety_score += 2
                break

        # face_name = "Unknown"
        # for face_box in face_boxes:
        #     face_coor = np.array(face_box[:4], dtype=np.int32)
        #     isp_score = calculate_isp(face_coor, head_coor)
        #     if isp_score >= 0.8:
        #         face_name = face_box[4]
        #         break

        target = list(person_coor)
        target.extend([safety_score, site_role])
        result.append(target)
    return result


def draw_demo(image, bboxes):
    image_h, image_w, _ = image.shape

    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)
        c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])

        fontScale = 1.5  # 0.9
        bbox_thick = int(1.5 * (image_h + image_w) / 600)

        if bbox[5] == -1:
            bbox_mess = bbox[4]
            bbox_color = (0, 255, 0)
        else:
            score = bbox[4]
            if score == 0:
                msg = "Unsafe"
                bbox_color = (255, 0, 0)
            elif score == 1:
                msg = "Just Head"
                bbox_color = (255, 165, 0)
            elif score == 2:
                msg = "Just Chest"
                bbox_color = (255, 255, 0)
            elif score == 3:
                msg = "Safe"
                bbox_color = (0, 255, 0)

            adj = "no role"
            if bbox[5] == 0:
                adj = "blue"
            if bbox[5] == 1:
                adj = "yellow"
            if bbox[5] == 2:
                adj = "white"
            if bbox[5] == 3:
                adj = "red"

            bbox_mess = f"{adj}: {msg}"
        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

        t_size = cv2.getTextSize(
            bbox_mess, 0, fontScale, thickness=bbox_thick//2)[0]  # // 2
        cv2.rectangle(
            image, (c1[0], c2[1]), (c1[0] + t_size[0], c2[1] - t_size[1] - 3), bbox_color, -1)  # filled
        cv2.putText(image, bbox_mess, (c1[0], c2[1]-2), cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale, (0, 0, 0), bbox_thick//2, lineType=cv2.LINE_AA)  # // 2

    return image
