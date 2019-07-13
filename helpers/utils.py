import cv2
import numpy as np
from PIL import Image


def preprocessing(img_path, input_dim):
    image = Image.open(img_path)
    image = np.array(image.resize(input_dim, resample=Image.BILINEAR))

    if image.shape[2] >= 3:
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    else:
        image = np.reshape(image, image.shape + (1,))
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # image = ((image - (0.485, 0.456, 0.406)) / (0.229, 0.224, 0.225))
    # image = ((image.astype(np.float32) - (141.41668995647873, 130.8247845828307, 121.2195841923002)) / (62.559792127264, 59.92365299858556, 58.08908465585324))
    image = image / 255.
    image = np.moveaxis(image, -1, 0)
    image = image.astype(np.float32)

    return image


def average(l):
    return sum(l) / len(l)


def make_trainable_false(model, n):
    for x in list(model.named_parameters())[:n]:
        x[1].requires_grad = False


def get_grads_status(model):
    grads_status = [(x[0], x[1].requires_grad) for x in list(model.named_parameters())]
    print('Trainable parameters:')
    print(*grads_status, sep='\n')
