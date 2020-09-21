import os

from yolov3.utils import BoundBox

prefix = "datasets/virtual_dataset/"

train_annotations_path = 'train/annotations/'
# valid_annotations_path = 'valid/annotations/'
# tune_annotations_path = 'tune/annotations/'

train_images_path = 'train/images/'
# valid_images_path = 'valid/images/'
# tune_images_path = 'tune/images/'

train_annotations_output_file = "train.annotations.txt"
# valid_annotations_output_file = "valid.annotations.txt"
# tune_annotations_output_file = "tune.annotations.txt"


def main():
    bbox = BoundBox(1088, 612)

    output_file_results = []
    files = os.listdir(prefix + train_annotations_path)
    for filename in files:
        with open(prefix + train_annotations_path + filename) as f:
            data = f.readlines()
            data = [item.strip() for item in data]
        file_result = [prefix + train_images_path + filename[:-4] + ".png"]

        for line in data:
            current_box = line.split(" ")
            obj_class = int(current_box[0])
            center_width = float(current_box[1])
            center_height = float(current_box[2])
            width = float(current_box[3])
            height = float(current_box[4])
            xmin, ymin, xmax, ymax = bbox.centroid_to_box(
                center_width, center_height, width, height)
            line_result = ",".join(
                [str(xmin), str(ymin), str(xmax), str(ymax), str(obj_class)])
            file_result.append(line_result)
        file_result = " ".join(file_result) + "\n"
        output_file_results.append(file_result)

    with open(train_annotations_output_file, 'w') as f:
        f.writelines(output_file_results)


if __name__ == "__main__":
    main()
