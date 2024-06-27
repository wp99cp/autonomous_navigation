import threading

import rosbag
from pytictac import ClassTimer, accumulate_time
from rospy import Time
from tqdm import tqdm

from utils.sliding_window_queue import SlidingWindowQueue

LOOKBACK = 3


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
            npc_bag_file: str = None,
            print_details: bool = False
    ):
        self.jetson_bag_file = jetson_bag_file
        self.lpc_bag_file = lpc_bag_file
        self.npc_bag_file = npc_bag_file
        self.print_details = print_details

        if self.print_details:
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

        self.training_data_arr = []

    @accumulate_time
    def load_data(self):

        if self.print_details:
            print("Loading data from bags")

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

        if self.print_details:
            self.report_frequencies()

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

        if self.print_details:
            print(f'Opening jetson bag file {self.jetson_bag_file}')

        bag = rosbag.Bag(self.jetson_bag_file)

        if self.print_details:
            self._print_bag_meta_data(bag, "Jetson")

        self._register_topics([topic for topic in bag.get_type_and_topic_info().topics], bag)

        self.jetson_bag = bag

    @accumulate_time
    def _open_lpc_bag(self):

        if self.lpc_bag_file is None:
            return

        if self.print_details:
            print(f'Opening lpc bag file {self.lpc_bag_file}')

        bag = rosbag.Bag(self.lpc_bag_file)

        if self.print_details:
            self._print_bag_meta_data(bag, "lpc")

        self._register_topics([topic for topic in bag.get_type_and_topic_info().topics], bag)

        self.lpc_bag = bag

    @accumulate_time
    def _open_npc_bag(self):

        if self.npc_bag_file is None:
            return

        if self.print_details:
            print(f'Opening npc bag file {self.npc_bag_file}')

        bag = rosbag.Bag(self.npc_bag_file)

        if self.print_details:
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

        self.training_data_arr = []

        # get msg streams
        img_stream, img_stream_start_t, _ = self.get_msg_stream(
            '/wide_angle_camera_rear/image_color_rect/compressed')
        state_stream, state_stream_start_t, _ = self.get_msg_stream(
            '/state_estimator/anymal_state')
        command_stream, command_stream_start_t, _ = self.get_msg_stream(
            '/motion_reference/command_twist')

        _, img_msg, _ = next(img_stream)
        _, state_msg, _ = next(state_stream)

        img_queue = SlidingWindowQueue(maxsize=2)
        pre_state_queue = SlidingWindowQueue(maxsize=5)
        post_state_queue = SlidingWindowQueue(maxsize=300)
        training_queue = SlidingWindowQueue(maxsize=300)

        counter = 0
        fst_cmd_t = None
        last_cmd_t = None
        for (_, command_msg, _) in tqdm(command_stream):
            counter += 1

            if fst_cmd_t is None:
                fst_cmd_t = command_msg.header.stamp
            last_cmd_t = command_msg.header.stamp

            if limit is not None and counter > limit:
                break

            try:
                while img_msg.header.stamp <= command_msg.header.stamp:
                    img_queue.put(img_msg)
                    _, img_msg, _ = next(img_stream)

                while state_msg.header.stamp <= command_msg.header.stamp:
                    pre_state_queue.put(state_msg)
                    _, state_msg, _ = next(state_stream)

                    # ignore states before the first command
                    if state_msg.header.stamp >= fst_cmd_t:
                        post_state_queue.put(state_msg)

                    if (not training_queue.empty() and post_state_queue.full() and
                            post_state_queue.queue[0].header.stamp >= training_queue.queue[0]['command'].header.stamp):
                        self.save_to_training_set(fst_cmd_t, post_state_queue, training_queue)

            except StopIteration:
                print("  stopping synchronization")
                break

            if img_queue.full() and pre_state_queue.full():

                cmd_t = command_msg.header.stamp
                imgs = img_queue.dump_as_array()
                states = pre_state_queue.dump_as_array()

                # check order of states
                for i in range(1, len(states)):
                    assert states[i].header.stamp >= states[i - 1].header.stamp, \
                        f"Expected {states[i].header.stamp} >= {states[i - 1].header.stamp}"

                # check order of images
                for i in range(1, len(imgs)):
                    assert imgs[i].header.stamp >= imgs[i - 1].header.stamp, \
                        f"Expected {imgs[i].header.stamp} >= {imgs[i - 1].header.stamp}"

                # all states and images should be before the command
                assert states[-1].header.stamp <= cmd_t, f"Expected {states[-1].header.stamp} <= {cmd_t}"
                assert imgs[-1].header.stamp <= cmd_t, f"Expected {imgs[-1].header.stamp} <= {cmd_t}"
                assert cmd_t >= fst_cmd_t, f"Expected {cmd_t} >= {fst_cmd_t}"

                training_queue.put({
                    'imgs': imgs,
                    'states': states,
                    'command': command_msg,
                })

        print("\n========================")
        print(f"Synchro done for {counter} frames, extracted {len(self.training_data_arr)} training data")
        print(f"Time between first command and last command:")
        print(f" - {(last_cmd_t - fst_cmd_t).to_sec()} seconds")
        print(f" - {(last_cmd_t - fst_cmd_t).to_sec() / 60} minutes")
        print("========================\n")
        return self.training_data_arr

    training_data_queue = SlidingWindowQueue(maxsize=300)

    def save_to_training_set(self, fst_cmd_t, post_state_queue, training_queue):

        training_data = training_queue.get()
        cmd_t = training_data['command'].header.stamp

        events = post_state_queue.copy_to_array()
        post_queue_start_t = events[0].header.stamp
        post_queue_end_t = events[-1].header.stamp

        upcoming_command_timestamps = [cmd['command'].header.stamp for cmd in training_queue.queue if
                                       cmd['command'].header.stamp <= post_queue_end_t]
        training_data['commands_timestamp'] = [cmd_t] + upcoming_command_timestamps
        assert post_state_queue.full(), "Expected post_state_queue to be full"

        assert post_queue_start_t >= fst_cmd_t, f"Expected {post_queue_start_t} <= {fst_cmd_t}"
        assert post_queue_start_t >= cmd_t, f"Expected {post_queue_start_t} >= {cmd_t}"

        assert len(events) == post_state_queue.maxsize, \
            f"Expected {post_state_queue.maxsize} events, got {len(events)}"

        # check order of events
        for i in range(1, len(events)):
            assert events[i].header.stamp >= events[i - 1].header.stamp, \
                f"Expected {events[i].header.stamp} >= {events[i - 1].header.stamp}"

        # validate time constraints
        assert cmd_t >= fst_cmd_t, \
            f"Expected {cmd_t} <= {fst_cmd_t}"

        command = training_data['command']
        training_data['commands'] = [command]
        del training_data['command']

        ARTIFICAL_SPACING = 3
        for i in range(LOOKBACK):
            idx = -i * ARTIFICAL_SPACING - 1
            if idx < -len(self.training_data_queue.queue):
                break

            self.training_data_queue.queue[idx][0]['commands'].append(command)

        # if we have collected enough commands (A=LOOKBACK), we save the data
        if bool(self.training_data_queue.queue) and len(self.training_data_queue.queue[0][0]['commands']) >= LOOKBACK:
            training_data, events = self.training_data_queue.get()

            training_data['commands'] = training_data['commands'][:LOOKBACK]

            # print where the events are placed
            command_0 = training_data['commands'][0].header.stamp
            command_1 = training_data['commands'][1].header.stamp
            command_2 = training_data['commands'][2].header.stamp

            state_ts = list(map(lambda x: x.header.stamp, events))

            states_before_0 = list(filter(lambda x: x < command_0, state_ts))
            states_before_1 = list(filter(lambda x: x < command_1, state_ts))
            states_before_2 = list(filter(lambda x: x < command_2, state_ts))

            # print(f"Commands after {len(states_before_0)}, {len(states_before_1)}, {len(states_before_2)} states")

            self.training_data_arr.append((
                training_data,
                events
            ))

        self.training_data_queue.put((
                training_data,
                events
            ))
