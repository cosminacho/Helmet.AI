import numpy as np
import os
import cv2

import zipfile


def main():
    train_images = os.listdir("datasets/virtual_dataset/train/images")
    valid_images = os.listdir("datasets/virtual_dataset/valid/images")

    np.random.shuffle(train_images)
    train_images.reverse()
    np.random.shuffle(train_images)
    train_images.reverse()
    np.random.shuffle(train_images)

    np.random.shuffle(valid_images)
    valid_images.reverse()
    np.random.shuffle(valid_images)
    valid_images.reverse()
    np.random.shuffle(valid_images)

    with open("datasets/virtual_dataset/train.txt", 'w') as f:
        for line in train_images:
            f.write("data/obj/" + line + '\n')

    with open("datasets/virtual_dataset/valid.txt", 'w') as f:
        for line in valid_images:
            f.write("data/obj/" + line + '\n')

    train_images = os.listdir("datasets/virtual_dataset/train/images")
    valid_images = os.listdir("datasets/virtual_dataset/valid/images")
    images = []
    images.extend(train_images)
    images.extend(valid_images)
    print(len(images))

    train_annotations = os.listdir(
        "datasets/virtual_dataset/train/annotations")
    valid_annotations = os.listdir(
        "datasets/virtual_dataset/valid/annotations")
    annotations = []
    annotations.extend(train_annotations)
    annotations.extend(valid_annotations)
    print(len(annotations))

    zips = []
    for index in range(4):
        zips.append(zipfile.ZipFile(f"virtual_train_part_{index+1}.zip", 'w'))

    zips.append(zipfile.ZipFile(f"virtual_train_part_5_1.zip", 'w'))
    zips.append(zipfile.ZipFile(f"virtual_train_part_5_2.zip", 'w'))

    #######################################################################################

    even = True
    for i, image in enumerate(images):
        index = i % 5

        if index == 4:
            if even:
                index = 4
            else:
                index = 5
            even = not even

        if i < 126900:
            zips[index].write(
                "datasets/virtual_dataset/train/images/" + image, "obj/" + image)
        else:
            zips[index].write(
                "datasets/virtual_dataset/valid/images/" + image, "obj/" + image)

    even = True
    for i, annotation in enumerate(annotations):
        index = i % 5

        if index == 4:
            if even:
                index = 4
            else:
                index = 5
            even = not even

        if i < 126900:
            zips[index].write(
                "datasets/virtual_dataset/train/annotations/" + annotation, "obj/" + annotation)
        else:
            zips[index].write(
                "datasets/virtual_dataset/valid/annotations/" + annotation, "obj/" + annotation)

    #######################################################################################

    for i in range(6):
        zips[i].close()


if __name__ == "__main__":
    main()
