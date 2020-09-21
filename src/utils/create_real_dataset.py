import os


def main():
    prefix1 = "datasets/real_dataset/train/"
    prefix2 = "datasets/real_dataset/valid/"

    # files = os.listdir(prefix1)
    # for filename in files:
    #     os.rename(prefix1 + filename, prefix1 + 'train_' + filename)

    files = os.listdir(prefix2)
    for filename in files:
        os.rename(prefix2 + filename, prefix2 + 'valid_' + filename)

    return

    with open("datasets/real_dataset/valid.real.txt", 'r') as f:
        lines = [line.strip() for line in f.readlines()]

    data = []
    for index, line in enumerate(lines):
        filename = "datasets/real_dataset/"
        img_filename = filename + line
        if line[-4] == '.':
            filename += line[:-4]
        else:
            filename += line[:-5]

        txt_filename = filename + '.txt'
        data.append(f"valid/item_{index+1}.jpg\n")
        os.rename(img_filename,
                  f"datasets/real_dataset/valid/item_{index+1}.jpg")
        os.rename(txt_filename,
                  f"datasets/real_dataset/valid/item_{index+1}.txt")

    with open("datasets/real_dataset/valid.valid.real.txt", 'w') as f:
        f.writelines(data)


if __name__ == "__main__":
    main()
