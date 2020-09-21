import numpy as np
import tensorflow as tf
import cv2

# import matplotlib.pyplot as plt


num_clusters = 9

prefix = "datasets/virtual_dataset/"
input_filename = "train.annotations.txt"
output_filename = "anchors.txt"


def iou(boxes, clusters):
    n = boxes.shape[0]
    k = num_clusters

    box_area = boxes[:, 0] * boxes[:, 1]
    box_area = box_area.repeat(k)
    box_area = np.reshape(box_area, (n, k))

    cluster_area = clusters[:, 0] * clusters[:, 1]
    cluster_area = np.tile(cluster_area, [1, n])
    cluster_area = np.reshape(cluster_area, (n, k))

    box_w_matrix = np.reshape(boxes[:, 0].repeat(k), (n, k))
    cluster_w_matrix = np.reshape(np.tile(clusters[:, 0], (1, n)), (n, k))
    min_w_matrix = np.minimum(cluster_w_matrix, box_w_matrix)

    box_h_matrix = np.reshape(boxes[:, 1].repeat(k), (n, k))
    cluster_h_matrix = np.reshape(np.tile(clusters[:, 1], (1, n)), (n, k))
    min_h_matrix = np.minimum(cluster_h_matrix, box_h_matrix)

    inter_area = np.multiply(min_w_matrix, min_h_matrix)

    result = inter_area / (box_area + cluster_area - inter_area)
    return result


def avg_iou(boxes, clusters):
    accuracy = np.mean([np.max(iou(boxes, clusters), axis=1)])
    return accuracy


def kmeans_clustering(boxes, k, dist=np.median):
    box_number = boxes.shape[0]
    distances = np.empty((box_number, k))
    last_nearest = np.zeros((box_number,))
    np.random.seed()

    clusters = boxes[np.random.choice(
        box_number, k, replace=False)]  # init k clusters

    while True:
        distances = 1 - iou(boxes, clusters)

        current_nearest = np.argmin(distances, axis=1)
        if (last_nearest == current_nearest).all():
            break  # clusters won't change
        for cluster in range(k):
            clusters[cluster] = dist(  # update clusters
                boxes[current_nearest == cluster], axis=0)

        last_nearest = current_nearest

    return clusters


def result_2_text(data):
    row = np.shape(data)[0]
    with open(output_filename, "a") as f:
        for i in range(row):
            if i == 0:
                x_y = "%d,%d" % (data[i][0], data[i][1])
            else:
                x_y = ", %d,%d" % (data[i][0], data[i][1])
            f.write(x_y)


def text_2_boxes():
    dataset = []
    with open(prefix + input_filename, 'r') as f:
        for line in f:
            line = line.strip()
            infos = line.split(" ")
            for i in range(1, len(infos)):
                current_box_coordinates = infos[i].split(",")
                width = int(
                    current_box_coordinates[2]) - int(current_box_coordinates[0])
                height = int(
                    current_box_coordinates[3]) - int(current_box_coordinates[1])
                dataset.append([width, height])

    result = np.array(dataset)
    return result


IMAGE_FEATURE_MAP = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "xmins": tf.io.VarLenFeature(tf.float32),
    "ymins": tf.io.VarLenFeature(tf.float32),
    "xmaxs": tf.io.VarLenFeature(tf.float32),
    "ymaxs": tf.io.VarLenFeature(tf.float32),
    "classes": tf.io.VarLenFeature(tf.int64)
}


def main():

    files = tf.data.Dataset.list_files(
        "datasets/real_dataset/*.tfrecord")
    dataset = files.flat_map(tf.data.TFRecordDataset)
    global_w = np.array([])
    global_h = np.array([])

    total_width = 0
    total_height = 0
    cnt = 0
    for record in dataset:
        cnt += 1
        x = tf.io.parse_single_example(record, IMAGE_FEATURE_MAP)
        image = tf.image.decode_jpeg(x['image'], channels=3)
        image = image.numpy()
        total_height += image.shape[0]
        total_width += image.shape[1]
        # print(image.numpy().shape)
        # image = tf.image.resize(image, (416, 416), method='bicubic')
        # image = image.numpy()
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # cv2.imwrite("test.jpg", image)

        xmins = np.round(tf.sparse.to_dense(x['xmins']).numpy() * 608)
        ymins = np.round(tf.sparse.to_dense(x['ymins']).numpy() * 608)
        xmaxs = np.round(tf.sparse.to_dense(x['xmaxs']).numpy() * 608)
        ymaxs = np.round(tf.sparse.to_dense(x['ymaxs']).numpy() * 608)

        widths = xmaxs - xmins
        heights = ymaxs - ymins

        global_w = np.concatenate((global_w, widths))
        global_h = np.concatenate((global_h, heights))

    print(total_width/cnt)
    print(total_height/cnt)
    print(cnt)
    all_boxes = np.stack((global_w, global_h)).T

    for i in range(20):
        np.random.shuffle(all_boxes)
        np.random.shuffle(all_boxes)
        np.random.shuffle(all_boxes)

        result = kmeans_clustering(all_boxes[:], num_clusters)
        result = result[np.lexsort(result.T[0, None])]
        result_2_text(result)

        print("K anchors:\n {}".format(result))
        print("Accuracy: {:.2f}%".format(
            avg_iou(all_boxes, result) * 100))


# def main():
#     all_boxes = text_2_boxes()
#     np.random.shuffle(all_boxes)
#     # all_boxes.reverse()
#     np.random.shuffle(all_boxes)
#     # all_boxes.reverse()
#     np.random.shuffle(all_boxes)

#     print(all_boxes.shape)

#     # heights = all_boxes[:, 1]
#     # widths = all_boxes[:, 0]
#     # plt.hist(heights, bins=10)
#     # plt.hist(widths, bins=10)
#     # plt.show()

#     # result = kmeans_clustering(all_boxes, num_clusters)
#     # result = result[np.lexsort(result.T[0, None])]
#     # result_2_text(result)

#     # 9 ANCHORS GOOD
#     result = [[16,  19], [24,  27], [34,  38], [39,  87], [49,  51],
#               [60, 122], [72,  75], [100, 153], [160, 293]]
#     result = np.array(result)

#     # 6 ANCHORS GOOD
#     # result = [[16,  19],
#     #           [27,  30],
#     #           [40, 45],
#     #           [56, 83],
#     #           [86, 139],
#     #           [146, 260]]
#     # result = np.array(result)

#     print("K anchors:\n {}".format(result))
#     print("Accuracy: {:.2f}%".format(
#         avg_iou(all_boxes, result) * 100))


if __name__ == "__main__":
    main()
