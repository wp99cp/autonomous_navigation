import threading

import numpy as np
import rosbag
from pytictac import ClassTimer, accumulate_time


def _print_lock(func):
    def wrapper(*args, **kwargs):
        args[0].print_lock.acquire()
        func(*args, **kwargs)
        args[0].print_lock.release()

    return wrapper


class DataHandler:

    def __init__(
            self,
            jetson_bag_file: str = None,
            lpc_bag_file: str = None,
            npc_bag_file: str = None
    ):
        self.jetson_bag_file = jetson_bag_file
        self.lpc_bag_file = lpc_bag_file
        self.npc_bag_file = npc_bag_file

        print(f"Jetson bag file: {self.jetson_bag_file}")
        print(f"lpc bag file: {self.lpc_bag_file}")
        print(f"npc bag file: {self.npc_bag_file}")

        self.cct = ClassTimer(objects=[self], names=["DataHandler"])

        self.jetson_bag = None
        self.lpc_bag = None
        self.npc_bag = None

        self.topics_register = {}

        # lock for printing
        self.print_lock = threading.Lock()

    @accumulate_time
    def load_data(self):
        print(f'Loading data from {self.jetson_bag_file}')

        # we open the bags in parallel
        threads = [
            threading.Thread(target=self._open_jetson_bag),
            threading.Thread(target=self._open_npc_bag),
            threading.Thread(target=self._open_lpc_bag)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    @accumulate_time
    @_print_lock
    def _print_bag_meta_data(self, bag: rosbag.Bag, bag_name: str):
        """
        Thread safe method to print bag meta data
        """

        print("\n +++++++++++++++")
        print(f"\n{bag_name} bag info:\n")

        print(f" » start time: {bag.get_start_time()}")
        print(f" » end time: {bag.get_end_time()}")
        print(f" » messages: {bag.get_message_count()}")

        print(f"\n{bag_name} bag topics:\n")
        topics = [topic for topic in bag.get_type_and_topic_info().topics]
        for topic in sorted(topics):
            print(f"  - {topic}")
        print("\n +++++++++++++++ \n")

    @accumulate_time
    def _register_topics(self, topics: [str], bag_ref: rosbag.Bag):

        for topic in topics:
            if topic not in self.topics_register:
                self.topics_register[topic] = bag_ref
            else:
                raise ValueError(f"Topic {topic} already registered")

    @accumulate_time
    def _open_jetson_bag(self):

        if self.jetson_bag_file is None:
            return

        print(f'Opening jetson bag file {self.jetson_bag_file}')

        bag = rosbag.Bag(self.jetson_bag_file)
        self._print_bag_meta_data(bag, "Jetson")
        self._register_topics([topic for topic in bag.get_type_and_topic_info().topics], bag)

        self.jetson_bag = bag

    @accumulate_time
    def _open_lpc_bag(self):

        if self.lpc_bag_file is None:
            return

        print(f'Opening lpc bag file {self.lpc_bag_file}')

        bag = rosbag.Bag(self.lpc_bag_file)
        self._print_bag_meta_data(bag, "lpc")
        self._register_topics([topic for topic in bag.get_type_and_topic_info().topics], bag)

        self.lpc_bag = bag

    @accumulate_time
    def _open_npc_bag(self):

        if self.npc_bag_file is None:
            return

        print(f'Opening npc bag file {self.npc_bag_file}')

        bag = rosbag.Bag(self.npc_bag_file)
        self._print_bag_meta_data(bag, "npc")
        self._register_topics([topic for topic in bag.get_type_and_topic_info().topics], bag)

        self.npc_bag = bag

    @accumulate_time
    def get_next_msg(self, topic_name: str, start_time: float):

        if topic_name not in self.topics_register:
            raise ValueError(f"Topic {topic_name} not found in the registered topics")

        bag = self.topics_register[topic_name]

        return next(bag.read_messages(topics=[topic_name], start_time=start_time))

    @accumulate_time
    def get_start_time(self):

        return min([bag.get_start_time() for bag in [self.jetson_bag, self.lpc_bag, self.npc_bag] if bag is not None])

    @accumulate_time
    def get_end_time(self):
        return max([bag.get_end_time() for bag in [self.jetson_bag, self.lpc_bag, self.npc_bag] if bag is not None])

    def report_time(self):
        print(f"\nTiming report for {self.__class__.__name__}")
        print(self.cct.__str__())
        print()
