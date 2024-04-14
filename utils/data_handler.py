import threading
from datetime import datetime

import rosbag
from pytictac import ClassTimer, accumulate_time
from rospy import Time
from tqdm import tqdm

from utils.sliding_window_queue import SlidingWindowQueue


def _to_timestamp(img_stream_t):
    seconds = img_stream_t.to_sec()
    date = datetime.fromtimestamp(seconds)
    return date


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
    def report_frequencies(self):

        print("\nFrequency report:")
        for topic, bag in self.topics_register.items():
            print(f"» {topic}: {bag.get_message_count() / (bag.get_end_time() - bag.get_start_time())}")

    @accumulate_time
    def get_msg_stream(self, topic_name: str, start_time: Time = None):

        if topic_name not in self.topics_register:
            raise ValueError(f"Topic {topic_name} not found in the registered topics")

        bag = self.topics_register[topic_name]
        return bag.read_messages(topics=[topic_name]), bag.get_start_time(), bag.get_end_time()

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

    @accumulate_time
    def get_synchronized_dataset(self, limit: int = None):

        training_data = []

        # get msg streams
        img_stream, img_stream_start_t, img_stream_end_t = self.get_msg_stream(
            '/wide_angle_camera_front/image_color/compressed')
        state_stream, state_stream_start_t, state_stream_end_t = self.get_msg_stream(
            '/state_estimator/anymal_state')
        command_stream, command_stream_start_t, command_stream_end_t = self.get_msg_stream(
            '/motion_reference/command_twist')

        counter = 0

        img_stream_topic, img_stream_msg, img_stream_t = next(img_stream)
        state_topic, state_msg, state_t = next(state_stream)

        img_queue = SlidingWindowQueue(maxsize=2)
        state_queue = SlidingWindowQueue(maxsize=5)
        event_queue = SlidingWindowQueue(maxsize=20)

        training_data_tmp = None
        for (command_topic, command_msg, command_t) in tqdm(command_stream):
            counter += 1

            if limit is not None and counter > limit:
                break

            try:

                while _to_timestamp(img_stream_t) < _to_timestamp(command_t):
                    img_stream_topic, img_stream_msg, img_stream_t = next(img_stream)
                    img_queue.put((img_stream_msg, img_stream_t))

                while _to_timestamp(state_t) < _to_timestamp(command_t):
                    state_topic, state_msg, state_t = next(state_stream)
                    state_queue.put((state_msg, state_t))

                    # check if the event queue only olds future states
                    if training_data_tmp is not None:
                        assert training_data_tmp['time'] < _to_timestamp(state_t), \
                            f"Expected {training_data_tmp['time']} < {state_t}"

                    event_queue.put((state_msg, state_t))

            except StopIteration:
                print("  stopping synchronization")
                break

            # save training data from previous iteration
            if training_data_tmp is not None:

                training_data.append((
                    training_data_tmp,
                    event_queue.dump_as_array()
                ))

            else:
                print("  skipping iteration")
                event_queue.empty()

            # create training data for current iteration
            # while skipping invalid data
            try:
                training_data_tmp = {
                    'imgs': img_queue.dump_as_array(),
                    'states': state_queue.dump_as_array(),
                    "time": _to_timestamp(command_t),
                    'commands': command_msg,
                }

            except AssertionError:
                training_data_tmp = None

        print(f"Synchro done for {counter} frames")
        return training_data
