import json
import os
import cv2

labels_dict = {
    "head": 0,
    "helmet": 1,
    "mask": 2,
    "headset": 3,
    "chest": 4,
    "vest": 5,
    "person": 6
}

prefix = "datasets/efden_dataset/"


def parse_entry(entry):
    image_name = entry['External ID']
    name = image_name[:-4]
    annotations_name = name + '.txt'

    if 'objects' not in entry['Label']:
        print("nunu")
        return None

    lines = []
    if not os.path.isfile("pictures/" + image_name):
        print('ahh')
        return None

    image = cv2.imread("pictures/" + image_name)
    img_height, img_width, _ = image.shape
    for obj in entry['Label']['objects']:
        obj_type = obj['value']
        obj_type = labels_dict[obj_type]

        x_center = (obj['bbox']['left'] +
                    (obj['bbox']['width'] / 2)) / img_width
        y_center = (obj['bbox']['top'] +
                    (obj['bbox']['height'] / 2)) / img_height

        width = obj['bbox']['width'] / img_width
        height = obj['bbox']['height'] / img_height

        lines.append(f"{obj_type} {x_center} {y_center} {width} {height}\n")

    with open("pictures/" + annotations_name, 'w') as f:
        f.writelines(lines)
    return image_name


def main():
    with open("efden_labels.json", "r") as f:
        data = f.read()

    data = json.loads(data)
    # with open("train.txt", 'w') as f:
    #     for entry in data:
    #         image_name = entry['External ID']
    #         f.write("data/obj/" + image_name + '\n')

    # with open("train.txt", 'w') as f:
    for entry in data:
        image_name = parse_entry(entry)
        if image_name == None:
            continue
        # f.write("data/obj/" + image_name)
    # garbage_files = os.listdir("garbage")
    # for garbage_file in garbage_files:
    #     name = garbage_file[:-4]
    #     garbage_txt = name + '.txt'
    #     fx = open("garbage/" + garbage_txt, 'w')
    #     fx.close()
    #     f.write("data/obj/" + garbage_file)


if __name__ == "__main__":
    main()
