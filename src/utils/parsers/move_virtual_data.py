import os


def move_training_dataset():
    with open("datasets/virtual_dataset/train.virtual.txt") as f:
        images_files = ["datasets/virtual_dataset/" +
                        "/".join(line.rstrip().split("/")[2:]) for line in f]
        annotations_files = [line[:-4] + '.txt' for line in images_files]

    for image_file in images_files:
        new_name = "./datasets/virtual_dataset/train/images/" + \
            "_".join(image_file.split("/")[-3:])
        os.rename(image_file, new_name)

    for annotation_file in annotations_files:
        new_name = "./datasets/virtual_dataset/train/annotations/" + \
            "_".join(annotation_file.split("/")[-3:])
        os.rename(annotation_file, new_name)


def move_validation_dataset():
    with open("datasets/virtual_dataset/valid.virtual.txt") as f:
        images_files = ["datasets/virtual_dataset/" +
                        "/".join(line.rstrip().split("/")[2:]) for line in f]
        annotations_files = [line[:-4] + '.txt' for line in images_files]

    for image_file in images_files:
        new_name = "./datasets/virtual_dataset/valid/images/" + \
            "_".join(image_file.split("/")[-3:])
        os.rename(image_file, new_name)

    for annotation_file in annotations_files:
        new_name = "./datasets/virtual_dataset/valid/annotations/" + \
            "_".join(annotation_file.split("/")[-3:])
        os.rename(annotation_file, new_name)


def move_tuning_dataset():
    for current_path, directories, file_names in os.walk("datasets/virtual_dataset", topdown=True):
        if current_path == 'datasets/virtual_dataset':
            directories.remove("train")
            directories.remove("valid")
            directories.remove("tune")

        for file_name in file_names:
            old_name = os.path.join(current_path, file_name)
            new_name = "_".join(old_name.split("\\")[1:])
            prefix = "datasets/virtual_dataset/tune/"
            if old_name[-3:] == "png":
                new_name = prefix + "images/" + new_name
            elif old_name[-3:] == "txt":
                new_name = prefix + "annotations/" + new_name
            if os.path.isfile(old_name):
                os.rename(old_name, new_name)


if __name__ == "__main__":
    print("Starting preprocessing ...\n")
    # move_training_dataset()
    # move_validation_dataset()
    # move_tuning_dataset()
    print("Done!")
