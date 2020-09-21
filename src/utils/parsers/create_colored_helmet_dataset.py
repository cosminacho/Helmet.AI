import os
import numpy as np
import cv2


def main():
    with open("all.txt") as f:
        data = [item.strip() for item in f.readlines()]

    np.random.shuffle(data)
    data.reverse()
    np.random.shuffle(data)
    data.reverse
    np.random.shuffle(data)
    with open("train.txt", 'w') as f:
        for i in range(len(data) - 640):
            line = data[i]
            item = "data/obj/" + line + '.jpg' + '\n'
            f.write(item)

    with open("valid.txt", 'w') as f:
        for i in range(len(data)-640, len(data)):
            line = data[i]
            item = "data/obj/" + line + '.jpg' + '\n'
            f.write(item)


if __name__ == "__main__":
    main()
