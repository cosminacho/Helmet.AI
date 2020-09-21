import numpy as np
import cv2

input_filename = "data/datasets/virtual_dataset/annotations/valid.annotations.txt"
# input_filename = "datasets/virtual_dataset/valid.valid.annotations.txt"
# input_filename = "datasets/virtual_dataset/tune.tune.annotations.txt"

# input_filename = "datasets/virtual_dataset/train.annotations.txt"
# input_filename = "datasets/virtual_dataset/valid.annotations.txt"
# input_filename = "datasets/virtual_dataset/tune.annotations.txt"

# output_filename = "datasets/virtual_dataset/new.train.annotations.txt"
# output_filename = "datasets/virtual_dataset/new.valid.annotations.txt"
# output_filename = "datasets/virtual_dataset/new.tune.annotations.txt"

# class_filename = "datasets/virtual_dataset/classes.txt"

# output_filename_valid = "datasets/virtual_dataset/valid.valid.annotations.txt"
# output_filename_tune = "datasets/virtual_dataset/tune.tune.annotations.txt"


def calculate_box_area(box):
    width = box[2] - box[0]
    height = box[3] - box[1]
    return width*height


def calculate_overlap_index(box1, box2):
    xmin_inter = None
    ymin_inter = None
    xmax_inter = None
    ymax_inter = None
    # checked
    if box1[0] <= box2[0] and box1[2] >= box2[0]:
        xmin_inter = box2[0]
    elif box1[0] >= box2[0] and box1[0] <= box2[2]:
        xmin_inter = box1[0]

    # checked
    if box1[1] <= box2[1] and box1[3] >= box2[1]:
        ymin_inter = box2[1]
    elif box1[1] >= box2[1] and box1[1] <= box2[3]:
        ymin_inter = box1[1]

    # checked
    if box1[2] >= box2[0] and box1[2] <= box2[2]:
        xmax_inter = box1[2]
    elif box1[0] <= box2[2] and box1[2] >= box2[2]:
        xmax_inter = box2[2]

    # checked
    if box1[3] >= box2[1] and box1[3] <= box2[3]:
        ymax_inter = box1[3]
    elif box1[1] <= box2[3] and box1[3] >= box2[3]:
        ymax_inter = box2[3]

    # print(xmin_inter, ymin_inter, xmax_inter, ymax_inter)
    if xmin_inter == None or xmax_inter == None or ymin_inter == None or ymax_inter == None:
        return 0.0

    box_area = calculate_box_area(box1)
    inter_area = calculate_box_area(
        [xmin_inter, ymin_inter, xmax_inter, ymax_inter])
    return inter_area / box_area


def remove_overlapping_boxes(boxes):
    dictio = {}
    for box in boxes:
        if box[4] in dictio:
            dictio[box[4]].append(box)
        else:
            dictio[box[4]] = [box]
    if 6 not in dictio:
        return boxes
    human_boxes_array = dictio[6]
    target_boxes = []
    for i in range(len(human_boxes_array)):
        for j in range(len(human_boxes_array)):
            if i == j:
                continue
            iou_score = calculate_overlap_index(
                human_boxes_array[i], human_boxes_array[j])
            if iou_score >= 0.75:
                target_boxes.append([iou_score, human_boxes_array[i]])

    if len(target_boxes) == 0:
        return boxes

    target_boxes.sort(key=lambda tup: tup[0])
    target_human_box = target_boxes.pop(-1)[1]
    head_boxes = []
    body_boxes = []

    for label in dictio:
        if label == 6:
            continue
        for box in dictio[label]:
            iou_score = calculate_overlap_index(box, target_human_box)
            if iou_score >= 0.75:
                if label == 0 or label == 1 or label == 2 or label == 3:
                    head_boxes.append([calculate_box_area(box), box])
                elif label == 4 or label == 5:
                    body_boxes.append([calculate_box_area(box), box])

    head_to_remove = -1
    body_to_remove = -1
    head_boxes.sort(key=lambda tup: tup[0])
    head_boxes.reverse()
    body_boxes.sort(key=lambda tup: tup[0])
    body_boxes.reverse()

    if len(head_boxes) != 0:
        head_to_remove = head_boxes.pop(-1)[1]
    if len(body_boxes) != 0:
        body_to_remove = body_boxes.pop(-1)[1]

    result = []

    for box in boxes:
        if box in [target_human_box, head_to_remove, body_to_remove]:
            pass
        else:
            result.append(box)

    return remove_overlapping_boxes(result)


def draw_labels(img, outputs, class_names):

    for output in outputs:
        x1y1 = (output[0], output[1])
        x2y2 = (output[2], output[3])
        img = cv2.rectangle(img, x1y1, x2y2, (0, 255, 0), 2)
        img = cv2.putText(img, class_names[output[4]],
                          x1y1, cv2.FONT_HERSHEY_SIMPLEX,
                          1, (0, 0, 255), 2)
    return img


def main():
    # class_names = [c.strip()
    #                for c in open(class_filename).readlines()]

    with open(input_filename) as f:
        data = [item.strip() for item in f.readlines()]

    # np.random.shuffle(data)
    # data.reverse()
    # np.random.shuffle(data)
    # data.reverse()
    # np.random.shuffle(data)

    # result_valid = []
    # result_tune = []
    # cnt_valid = 0
    # cnt_tune = 0
    cnt = 0
    for index, line in enumerate(data):
        items = line.split(" ")
        image_filename = items[0]
        boxes = items[1:]
        boxes = [[int(item) for item in box.split(",")] for box in boxes]

        new_boxes = remove_overlapping_boxes(boxes)
        if len(new_boxes) != len(boxes):
            cnt += 1
            print(image_filename)

    print(cnt)

    # image = cv2.imread(image_filename)
    # labeled_image = draw_labels(image, boxes, class_names)
    # cv2.imwrite(f"tuned/image{index}.png", labeled_image)

    #     if len(boxes) == len(new_boxes) and cnt_valid < 400:
    #         cnt_valid += 1
    #         result_valid.append(line + '\n')
    #     else:
    #         cnt_tune += 1
    #         result_tune.append(line + '\n')

    # with open("testare.txt", 'w') as f:
    #     f.writelines(result_valid)

    # with open(output_filename_tune, 'w') as f:
    #     f.writelines(result_tune)

    # print(cnt_valid)
    # print(cnt_tune)


if __name__ == "__main__":
    main()
