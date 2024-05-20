import multiprocessing
import os
import pickle

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from model import Model
from utils.data_handler import DataHandler
from utils.utils import get_rgb_image

DATA_PICKLE_FILE = '/tmp/data_FrRL.pkl'
CREATE_PLOTS = False


def _feature_extraction(xs):
    # feature extraction
    numerical_features = []

    for state in xs['states']:
        numerical_features.append(state.twist.twist.linear.x)
        numerical_features.append(state.twist.twist.linear.y)
        numerical_features.append(state.twist.twist.linear.z)

    # get images and convert them to RGB
    imgs = []
    times = []
    for img in xs['imgs']:
        imgs.append(get_rgb_image(img))
        times.append(img.header.stamp)

    return np.array(numerical_features), np.array(imgs), times


def _label_extraction(ys, xs):
    # plot the contact forces /state_estimator/contact_force_lf_foot
    contact_force_LF_z = [y.contacts[0].wrench.force.z for y in ys]
    contact_force_RF_z = [y.contacts[1].wrench.force.z for y in ys]
    contact_force_LH_z = [y.contacts[2].wrench.force.z for y in ys]
    contact_force_RH_z = [y.contacts[3].wrench.force.z for y in ys]

    position_LF_z = [y.contacts[0].position.z for y in ys]
    position_RF_z = [y.contacts[1].position.z for y in ys]
    position_LH_z = [y.contacts[2].position.z for y in ys]
    position_RH_z = [y.contacts[3].position.z for y in ys]

    twist_x = [y.twist.twist.linear.x for y in ys]
    twist_y = [y.twist.twist.linear.y for y in ys]
    twist_z = [y.twist.twist.linear.z for y in ys]

    # map index to time stamps
    idxs = [y.header.stamp.to_sec() for y in ys]

    # set title
    time_stamp = ys[0].header.stamp.to_sec()

    # convert time to humain readable format
    import datetime
    time_stamp = datetime.datetime.fromtimestamp(time_stamp).strftime('%Y-%m-%d %H:%M:%S %f')

    # plot forces using matplotlib
    if CREATE_PLOTS:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(2, 2, figsize=(15, 10))
        plt1 = ax[0][0]
        plt2 = ax[0][1]
        plt3 = ax[1][0]
        plt4 = ax[1][1]

        # plot vertical lines at command timestamps
        command = xs['command']
        cmd_timestamps = [x.to_sec() for x in xs['commands_timestamp']]
        cmd_timestamps = np.array(cmd_timestamps)
        for p in [plt1, plt2, plt4]:
            for i, cmd_timestamp in enumerate(cmd_timestamps):
                if i == 0:
                    p.axvline(x=cmd_timestamp, color='b', linestyle='-.', label='upcoming commands', alpha=0.1)
                else:
                    p.axvline(x=cmd_timestamp, color='b', linestyle='-.', alpha=0.1)

            # plot horizontal line at command.twist.linear.x
            p.axvline(x=command.header.stamp.to_sec(), color='b', linestyle='--', alpha=0.5)

        plt1.plot(idxs, contact_force_LF_z, label='LF')
        plt1.plot(idxs, contact_force_RF_z, label='RF')
        plt1.plot(idxs, contact_force_LH_z, label='LH')
        plt1.plot(idxs, contact_force_RH_z, label='RH')
        plt1.set_title(f'Contact forces (z) [starting at {time_stamp}]')
        plt1.set_ylim([-100, 500])

        plt1.legend(loc='upper right')
        plt1.set_xlabel('Time [s]')
        plt1.set_ylabel('Force [N]')

        plt2.plot(idxs, position_LF_z, label='LF')
        plt2.plot(idxs, position_RF_z, label='RF')
        plt2.plot(idxs, position_LH_z, label='LH')
        plt2.plot(idxs, position_RH_z, label='RH')
        plt2.set_title(f'Contact positions (z) [starting at {time_stamp}]')
        plt2.set_xlabel('Time [s]')
        plt2.set_ylabel('Position [m]')
        plt2.legend(loc='upper right')

        img = xs['imgs'][0]
        t = img.header.stamp
        img = get_rgb_image(img)

        plt3.imshow(img)
        time_stamp = datetime.datetime.fromtimestamp(t.to_sec()).strftime('%Y-%m-%d %H:%M:%S %f')
        plt3.set_title(f'RGB Image [Captured at {time_stamp}]')

        # add description
        plt3.text(7, 190, f'cmd.linear.x={command.twist.linear.x:.2f} m/s', color='white', fontsize=10,
                  fontweight='bold')
        plt3.text(7, 200, f'cmd.angular.z={command.twist.angular.z:.2f} rad/s', color='white', fontsize=10,
                  fontweight='bold')

        # plot horizontal line at command.twist.linear.x
        plt4.axhline(y=command.twist.linear.x, color='b', linestyle='--', label='cmd.linear.x', alpha=0.5)

        plt4.plot(idxs, twist_x, label='x')
        plt4.plot(idxs, twist_y, label='y')
        plt4.plot(idxs, twist_z, label='z')
        plt4.set_title(f'twist.linear [starting at {time_stamp}]')
        plt4.set_xlabel('Time [s]')
        plt4.set_ylabel('Twist [m/s]')

        plt4.legend(loc='upper left')
        plt4.set_ylim([-1.6, 1.6])

        plt.tight_layout()

        # save plot in folder /tmp
        time_stamp = ys[150].header.stamp
        time_stamp_nano = time_stamp.to_nsec()

        # fill up the string with zeros
        time_stamp_str = str(time_stamp_nano) + '0' * (19 - len(str(time_stamp_nano)))
        fig.savefig(f'/tmp/contact_forces_{time_stamp_str}.png')

        # close the plot
        plt.close(fig)

    return [0, 1, 2, 3]


def _prepare_data(item):
    xs, y = item
    return _feature_extraction(xs), _label_extraction(y, xs)


def train_model(data_set):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Start training model on device:", device)

    # split data into training and validation
    (inputs, targets) = zip(*data_set)
    print(f"Data set: {len(inputs)} samples")
    print(f"targets len: {targets[0]}")
    model = Model(num_actions=len(targets[0])).to(device)

    # move input and target to device
    inputs_scalar = [torch.tensor(t[0]) for t in inputs]
    inputs_images = [t[1] for t in inputs]
    targets = [torch.tensor(t) for t in targets]

    # prepare data for resnet
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    pre_inputs_scalar = []
    pre_inputs_images = []
    pre_targets = []

    # pre-precess data for training
    print("\nPre-processing data for training...")
    for (_input_scalars, _input_imgs, _target) in tqdm(zip(inputs_scalar, inputs_images, targets),
                                                       total=len(inputs_scalar)):
        # extract images
        img1 = Image.fromarray(_input_imgs[0])
        img2 = Image.fromarray(_input_imgs[1])

        img1 = preprocess(img1)
        img2 = preprocess(img2)

        # covert everything to float tensor
        _input_scalars = _input_scalars.float()
        _target = _target.float()
        img1 = img1.float()
        img2 = img2.float()

        # move everything to device
        _input_scalars = _input_scalars.to(device)
        _target = _target.to(device)
        img1 = img1.to(device)
        img2 = img2.to(device)

        pre_inputs_scalar.append(_input_scalars)
        pre_inputs_images.append(img1)  # TODO: add img2
        pre_targets.append(_target)

    print("\nFinished pre-processing data for training.")

    # convert data to batches of tensors
    b_size = 32

    # to tensor batches
    pre_inputs_scalar = torch.stack(pre_inputs_scalar)
    pre_inputs_images = torch.stack(pre_inputs_images)
    pre_targets = torch.stack(pre_targets)

    pre_inputs_scalar_batches = pre_inputs_scalar.split(b_size)
    pre_inputs_images_batches = pre_inputs_images.split(b_size)
    pre_targets_batches = pre_targets.split(b_size)

    # train the model
    epochs = 10

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, amsgrad=True, betas=(0.9, 0.999), eps=1e-6)
    loss_fn = torch.nn.CrossEntropyLoss()
    model.train()

    print("\nTraining the model...")
    for epoch in range(epochs):

        pbar = tqdm(zip(pre_inputs_scalar_batches, pre_inputs_images_batches, pre_targets_batches),
                    total=len(pre_inputs_scalar_batches))

        for (_input_scalars_batch, _input_imgs_batch, _target_batch) in pbar:
            logits, res = model.forward(_input_scalars_batch, _input_imgs_batch)
            loss = loss_fn(logits, _target_batch)

            pbar.set_description(f"Epoch: {epoch}/{epochs} | Loss: {loss.item()}")

            # run backpropagation
            loss.backward()
            optimizer.step()

    # test the model performance
    mean_mse = 0
    count = 0

    print("\nTesting the model...")
    for (_input_scalars_batch, _input_imgs_batch, _target_batch) in \
            tqdm(zip(pre_inputs_scalar_batches, pre_inputs_images_batches, pre_targets_batches),
                 total=len(pre_inputs_scalar_batches)):

        count += 1

        logits, res = model.forward(_input_scalars_batch, _input_imgs_batch)

        if count == 1:
            print("\n")
            print("Example from the first batch:")
            print(f" » target: {_target_batch[0].to('cpu').detach().numpy()}")
            print(f" » logits: {logits[0].to('cpu').detach().numpy()}")
            print(f" » result: {res[0].to('cpu').detach().numpy()}")
            print("\n")

        mean_mse += torch.nn.functional.mse_loss(res, _target_batch).to('cpu').detach().numpy()

    mean_mse /= len(inputs_scalar)
    print(f"Mean MSE: {mean_mse}")


def main():
    force_data_extraction = True

    # check if pre-processed data exists in tmp folder
    # or flag to force re-processing
    if not os.path.exists(DATA_PICKLE_FILE) or force_data_extraction:
        extract_data_from_bags()

    # load pre-processed data
    data_set = pickle.load(open(DATA_PICKLE_FILE, 'rb'))
    assert data_set is not None, "Data set is None"
    print(f"Data set loaded: {len(data_set)} samples")

    train_model(data_set)


def extract_data_from_bags():
    bags_base_dir = 'data/20230302_Hoengg_Forst_Dodo/mission_data/'

    # list all folders in the base directory
    dirs = os.listdir(bags_base_dir)

    for d in dirs:
        # find jetson, lpc and npc bag files
        jetson_bag_file = None
        lpc_bag_file = None
        npc_bag_file = None

        for f in os.listdir(bags_base_dir + d):
            if 'jetson' in f:
                jetson_bag_file = bags_base_dir + d + '/' + f
            if 'lpc' in f:
                lpc_bag_file = bags_base_dir + d + '/' + f
            if 'npc' in f:
                npc_bag_file = bags_base_dir + d + '/' + f

        assert jetson_bag_file is not None, "Jetson bag file not found"
        assert lpc_bag_file is not None, "LPC bag file not found"
        assert npc_bag_file is not None, "NPC bag file not found"

        print("\nProcessing bags:")
        print(f" » {jetson_bag_file.split('/')[-1]}, {lpc_bag_file.split('/')[-1]}, {npc_bag_file.split('/')[-1]}")

        try:

            data_handler = DataHandler(
                jetson_bag_file=jetson_bag_file,
                lpc_bag_file=lpc_bag_file,
                npc_bag_file=npc_bag_file,
                print_details=False
            )

            data_handler.load_data()

            # get synchronized data
            print("\nGetting synchronized data")
            synchronized_training_data = data_handler.get_synchronized_dataset(limit=None)

            if synchronized_training_data is None or len(synchronized_training_data) == 0:
                print("No synchronized data found")
                continue

            # run the prepare data function for the first example as a test
            print("\nRunning prepare data function for the first example as a test")
            _prepare_data(synchronized_training_data[0])
            continue

            # prepare data
            # Use the maximum number of available CPU cores for parallel processing
            num_cores = min(multiprocessing.cpu_count() * 2, 32)
            print("\nPreparing data with multiprocessing n_cors =", num_cores)
            with multiprocessing.Pool(num_cores) as p:
                training_data = list(
                    tqdm(p.imap(_prepare_data, synchronized_training_data), total=len(synchronized_training_data)))
            print(f"\nTraining data: {len(training_data)} samples")

            # TODO: remove the following line
            # training_data = training_data[:100]

            # save data as a pickle file
            print("\nSaving data to", DATA_PICKLE_FILE)
            pickle.dump(training_data, open(DATA_PICKLE_FILE, 'wb'))
            print("\nData saved to", DATA_PICKLE_FILE)

            # report the time taken for each method
            data_handler.report_time()

        except Exception as e:
            print(f"Error processing bags: {e}")


if __name__ == "__main__":
    main()
