import cv2
import numpy as np


def process_image(img):
    height, width, _ = img.shape
    if height > width:
        img = np.rot90(img)
        height, width = width, height

    new_height = 128
    new_width = int(round(width * new_height / height))
    img = cv2.resize(img, (new_width, new_height))  # sic!
    height, width, _ = img.shape

    img = img.astype('float32')

    if height < 128:
        top_zeros_height = (128 - height) // 2
        bottom_zeros_height = 128 - height - top_zeros_height
        top_zeros = np.full((top_zeros_height, width, 3), 255)
        bottom_zeros = np.full((bottom_zeros_height, width, 3), 255)
        img = np.concatenate((top_zeros, img, bottom_zeros))
        height, width, _ = img.shape

    if height > 128:
        raise Exception(f"strange image size {height, width}")

    img /= 255

    return img
