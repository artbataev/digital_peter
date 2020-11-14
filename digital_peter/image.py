import cv2
import numpy as np


def process_image(img, target_height=128):
    height, width, _ = img.shape
    if height > width:
        img = np.rot90(img)
        height, width = width, height

    new_height = target_height
    new_width = int(round(width * new_height / height))
    img = cv2.resize(img, (new_width, new_height))  # sic!
    height, width, _ = img.shape

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # added after baseline

    if height < target_height:
        top_zeros_height = (target_height - height) // 2
        bottom_zeros_height = target_height - height - top_zeros_height
        top_zeros = np.full((top_zeros_height, width, 3), 255, dtype=img.dtype)
        bottom_zeros = np.full((bottom_zeros_height, width, 3), 255, dtype=img.dtype)
        img = np.concatenate((top_zeros, img, bottom_zeros))
        height, width, _ = img.shape

    if height > target_height:
        raise Exception(f"strange image size {height, width}")

    return img
