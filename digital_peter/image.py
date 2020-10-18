import cv2
import numpy as np


def process_image(img):
    w, h, _ = img.shape

    new_w = 128
    new_h = int(h * (new_w / w))
    img = cv2.resize(img, (new_h, new_w))
    w, h, _ = img.shape

    img = img.astype('float32')

    if w < 128:
        add_zeros = np.full((128 - w, h, 3), 255)
        img = np.concatenate((img, add_zeros))
        w, h, _ = img.shape

    if h < 1024:
        add_zeros = np.full((w, 1024 - h, 3), 255)
        img = np.concatenate((img, add_zeros), axis=1)
        w, h, _ = img.shape

    if h > 1024 or w > 128:
        dim = (1024, 128)
        img = cv2.resize(img, dim)

    img = cv2.subtract(255, img)
    img = img / 255
    return img
