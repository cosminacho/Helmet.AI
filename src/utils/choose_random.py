import os
import numpy as np


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


def main():
    input_filename = "datasets/virtual_dataset/annotations/all.tune.annotations.txt"

    with open(input_filename, 'r') as f:
        data = [item.strip() for item in f.readlines()]

    np.random.shuffle(data)
    data.reverse()
    np.random.shuffle(data)
    data.reverse()
    np.random.shuffle(data)

    count = 0
    non_count = 0
    for line in data:
        items = line.split(" ")
        image_filename = items[0][:-4] + '.jpg'
        if not os.path.isfile(image_filename):
            print('problemuta')
            break

        boxes = [[int(item) for item in box.split(",")] for box in items[1:]]
        new_boxes = remove_overlapping_boxes(boxes)

        if len(boxes) != len(new_boxes):
            non_count += 1
            continue

        count += 1
        image_name = image_filename.split("/")[-1]
        text_name = image_name[:-4] + '.txt'
        os.rename(image_filename,
                  "datasets/virtual_dataset/valid/images/" + image_name)
        os.rename("datasets/virtual_dataset/tune/annotations/" + text_name,
                  "datasets/virtual_dataset/valid/annotations/" + text_name)
        if count == 3500:
            print('dadada')
            print("non count" + str(non_count))
            break


if __name__ == "__main__":
    main()
