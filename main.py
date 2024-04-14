import cv2
import numpy as np
from matplotlib import pyplot as plt
from rospy import rostime
from tqdm import tqdm

from utils.data_handler import DataHandler


def get_rgb_image(_msg):
    np_arr = np.frombuffer(_msg.data, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


def main():
    bags_base_dir = 'data/20230302_Hoengg_Forst_Dodo/mission_data/2023-03-02-10-03-50/'

    data_handler = DataHandler(
        jetson_bag_file=bags_base_dir + '2023-03-02-10-03-50_anymal-d020-jetson_mission_0.bag',
        lpc_bag_file='2023-03-02-10-03-50_anymal-d020-lpc_mission_0.bag',
        npc_bag_file='2023-03-02-10-03-50_anymal-d020-npc_mission_0.bag'
    )

    data_handler.load_data()

    positions = []
    twist_linear = []
    command_twist = []

    imgs = []
    steps = 1_000
    print(f"Processing {steps} steps")
    for i in tqdm(range(steps)):
        mission_percent = i / float(steps * 6.)

        time = data_handler.get_start_time() + (
                data_handler.get_end_time() - data_handler.get_start_time()) * mission_percent
        time = rostime.Time.from_sec(time)

        try:  # sometimes we are missing the last frame

            topic, msg, t = data_handler.get_next_msg(
                '/motion_reference/command_twist',
                time
            )

            command_twist.append([i, msg.twist.linear.x, -msg.twist.angular.z])

            # Robot info, e.g., base position [m] and orientation [rad], joint position [rad], velocity [rad/s], acceleration [rad/s2], and torque [Nm].
            topic, msg, t = data_handler.get_next_msg(
                '/state_estimator/anymal_state',
                time
            )

            twist_linear.append(
                [i, msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z])

            # additional data
            append_position(data_handler, positions, time)

            # add frame for image
            img = append_frame(data_handler, imgs, time)

            # break  # TODO: remove this line to get all images

        except StopIteration:
            break

    # print the positions as a 2d plot
    positions = np.array(positions)
    twist_linear = np.array(twist_linear)
    command_twist = np.array(command_twist)

    fig, ax = plt.subplots(1, 5, figsize=(10, 8),
                           gridspec_kw={'width_ratios': [6, 1, 1, 1, 1]})

    ax[0].plot(positions[:, 0], positions[:, 1])
    ax[1].plot(twist_linear[:, 1], twist_linear[:, 0])
    ax[2].plot(twist_linear[:, 2], twist_linear[:, 0])
    ax[3].plot(command_twist[:, 1], command_twist[:, 0])
    ax[4].plot(command_twist[:, 2], command_twist[:, 0])

    # add legend
    ax[0].set_title('Position')
    ax[1].set_title('Twst Lin (x)')
    ax[2].set_title('Twst Lin (y)')
    ax[3].set_title('Com Twst (x)')
    ax[4].set_title('Com Twst  (z)')

    # save images as a video
    height, width, layers = img.shape
    size = (width, height)

    out = cv2.VideoWriter('project.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 24, size)

    print("\n\nCreating video")
    for i in tqdm(range(len(imgs))):
        img = imgs[i]

        ax[0].plot(positions[i, 0], positions[i, 1], 'ro')
        ax[1].plot(twist_linear[i, 1], twist_linear[i, 0], 'ro')
        ax[2].plot(twist_linear[i, 2], twist_linear[i, 0], 'ro')
        ax[3].plot(command_twist[i, 1], command_twist[i, 0], 'ro')
        ax[4].plot(command_twist[i, 2], command_twist[i, 0], 'ro')

        plt.savefig('temp.png', transparent=True)
        overlay = cv2.imread('temp.png')
        overlay = cv2.resize(overlay, (width, height))
        img = cv2.addWeighted(img, 0.5, overlay, 0.5, 0)
        out.write(img)

    out.release()

    # report the time taken for each method
    data_handler.report_time()


def append_position(data_handler, positions, time):
    topic, msg, t = data_handler.get_next_msg(
        '/state_estimator/pose_in_odom',
        time
    )
    msg = msg.pose.pose.position
    positions.append([msg.x, msg.y])
    # print(f"Position: {msg.x, msg.y, msg.z}")


def append_frame(data_handler, imgs, time):
    topic, msg, t = data_handler.get_next_msg(
        '/wide_angle_camera_front/image_color/compressed',
        time
    )
    img = get_rgb_image(msg)
    imgs.append(img)
    return img


if __name__ == "__main__":
    main()
