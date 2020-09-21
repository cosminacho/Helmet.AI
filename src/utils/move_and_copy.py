import os
from shutil import copyfile


def main():
    # with open("data/virtual_dataset/annotations/original/valid.virtual.txt") as f:
    #     lines = [item.strip() for item in f.readlines()]

    # names = []
    # for line in lines:
    #     item = line.split("/")
    #     name = '_'.join(item[-3:])
    #     name = name[:-4] + '.jpg'
    #     names.append(name)

    # dictio = {}
    # for name in names:
    #     dictio[name] = True

    # files = []
    # files.extend(os.listdir("data/virtual_dataset/valid/images"))
    # files.extend(os.listdir("data/virtual_dataset/tune/images"))

    # cnt = 0
    # for filename in files:
    #     if filename in dictio:
    #         txt_file = filename[:-4] + '.txt'
    #         if cnt >= 3500:
    #             dir_path = "data/virtual_dataset/tune/"
    #         else:
    #             dir_path = "data/virtual_dataset/valid/"

    #         copyfile(dir_path + "images/" + filename,
    #                  "data/virtual_dataset/poze/" + filename)
    #         copyfile(dir_path + "annotations/" + txt_file,
    #                  "data/virtual_dataset/poze/" + txt_file)

    #     cnt += 1
    files = os.listdir("data/virtual_dataset/obj")
    with open("valid.txt", 'w') as f:
        for filename in files:
            name = filename[:-4]
            ann = filename[-4:]

            if ann == '.txt':
                continue

            f.write("data/obj/" + filename + '\n')


if __name__ == "__main__":
    main()
