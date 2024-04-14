import os

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from bagpy import bagreader


def find(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]


def get_cache_file(bag_path, topic):
    return f'/tmp/{bag_path.split("/")[-1][0:-4]}/{topic[1:].replace("/", "-")}.pkl'


class Dataloader:

    def __init__(self, bag_path_npc, bag_path_lpc, bag_path_jetson):
        self.bag_path_npc = bag_path_npc
        self.bag_path_lpc = bag_path_lpc
        self.bag_path_jetson = bag_path_jetson

        self.bag_npc = None
        self.bag_lpc = None
        self.bag_jetson = None

        self.topics_npc = None
        self.topics_lpc = None
        self.topics_jetson = None

        # step size between each iteration
        self.step_size = 1

    def _load_bags(self):
        assert self.bag_path_npc is not None, "bag path npc not set"
        assert self.bag_path_lpc is not None, "bag path lpc not set"
        assert self.bag_path_jetson is not None, "bag path jetson not set"

        self.bag_npc = bagreader(self.bag_path_npc, tmp=True)
        self.bag_lpc = bagreader(self.bag_path_lpc, tmp=True)
        self.bag_jetson = bagreader(self.bag_path_jetson, tmp=True)

    def _bag_loaded(self):
        return self.bag_npc is not None and self.bag_lpc is not None and self.bag_jetson is not None

    def get_data_frame(self, topic: str, ignore_cache: bool = False):

        # check if the topic is already cached as a pickle file
        for bag_path in [self.bag_path_npc, self.bag_path_lpc, self.bag_path_jetson]:
            cache_file = get_cache_file(bag_path, topic)

            if not ignore_cache and os.path.exists(cache_file):
                # load from cache
                df = pd.read_pickle(cache_file)
                return df

        # the topic is not cached, so we need to load it from the bag file
        # for that we must load all the bag files
        if not self._bag_loaded():
            self._load_bags()

        # search for the topic in the bag files
        if topic in self.bag_npc.topics:
            bag = self.bag_npc
            cache_file = get_cache_file(self.bag_path_npc, topic)
        elif topic in self.bag_lpc.topics:
            bag = self.bag_lpc
            cache_file = get_cache_file(self.bag_path_lpc, topic)
        elif topic in self.bag_jetson.topics:
            bag = self.bag_jetson
            cache_file = get_cache_file(self.bag_path_jetson, topic)

        # if the topic is not found in any bag, raise an error
        else:
            raise ValueError("Topic not found in any bag")

        # retrieve the topic and save it as a pickle file
        csv_path = bag.message_by_topic(topic)
        df = pd.read_csv(csv_path)

        # convert the data column to a bytearray
        df['data'] = df['data'].apply(lambda x: x.decode('ascii'))

        # save the dataframe as a pickle file
        df.to_pickle(cache_file)

        # delete csv file
        os.remove(csv_path)
        return df


def raw_to_img(_data):
    buf = np.ndarray(
        shape=(_data.height, _data.width),
        dtype=np.uint8, buffer=_data.data.encode()
    )
    return Image.fromarray(buf)


def compressed_to_img(_data):
    """
    converts a bgr8; jpeg compressed bgr8 to an image

    """

    raw_data = _data.data

    print(f"type(raw_data): {type(raw_data)}")
    print(f"raw_data: {raw_data}")

    img = cv2.imdecode(raw_data, cv2.IMREAD_COLOR)
    print(f"img shape: {img.shape}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


if __name__ == '__main__':
    bags_base_dir = 'data/20230818_Bahnhofstrasse_Dodo/mission_data/dodo_mission_2023_08_19/2023-08-19-19-40-19/'

    dataloader = Dataloader(
        bag_path_npc=bags_base_dir + '2023-08-19-19-40-19_anymal-d020-npc_0.bag',
        bag_path_lpc=bags_base_dir + '2023-08-19-19-40-19_anymal-d020-lpc_0.bag',
        bag_path_jetson=bags_base_dir + '2023-08-19-19-40-19_anymal-d020-jetson_0.bag'
    )

    df_wide_angle_camera_front = dataloader.get_data_frame('/wide_angle_camera_front/image_color_rect/compressed')
    print(f"df shape: {df_wide_angle_camera_front.shape}")
    print(f"df head: {df_wide_angle_camera_front.head()}")

    for i in range(0, 5):
        _data = df_wide_angle_camera_front.iloc[i]
        img = compressed_to_img(_data)
        print(f"img shape: {img}")
        exit(0)

    print("\n\n\n")

    # df_depth_camera_left = dataloader.get_data_frame('/depth_camera_left/depth/image_rect_raw')

    # print(f"df shape: {df_depth_camera_left.shape}")
    # print(f"df head: {df_depth_camera_left.head()}")

    # for i in range(0, 5):
    #    _data = df_depth_camera_left.iloc[i]
    #    img = raw_to_img(_data)
    #    # img.show()
