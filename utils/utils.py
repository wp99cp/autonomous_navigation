import cv2
import numpy as np


def get_rgb_image(_msg):
    np_arr = np.frombuffer(_msg.data, np.uint8)

    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224))
    return img
