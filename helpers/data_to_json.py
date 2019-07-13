import json
import os
import random

import cv2

from config import *

random.seed(43)


def get_data(path):
    data = {}
    for folder in sorted(os.listdir(path)):
        data[folder] = []
        for file in sorted(os.listdir(os.path.join(path, folder))):
            f = os.path.join(path, folder, file)
            # if cv2.imread(f) is not None:
            data[folder].append(f)

    class_keys = sorted(list(data.keys()))
    classes = dict([(class_keys[x], x) for x in range(len(class_keys))])

    train = []
    val = []

    for class_idx in class_keys:
        train += [(x, classes[class_idx]) for x in data[class_idx][:-splitter:-1]]
        val += [(x, classes[class_idx]) for x in data[class_idx][-splitter:]]

    random.shuffle(train)
    random.shuffle(val)

    return train, val


if __name__ == '__main__':
    train, val = get_data(path)

    print('Size of training triplets:', len(train), '\nSize of validation triplets:', len(val))

    # with open('./data/train_small.json', 'w') as f:
    #     json.dump(train, f)
    #
    # with open('./data/val_small.json', 'w') as f:
    #     json.dump(val, f)

    print('Data saved to json files in data folder.')
