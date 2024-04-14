import cv2
import numpy as np


def get_rgb_image(_msg):
    np_arr = np.frombuffer(_msg.data, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
