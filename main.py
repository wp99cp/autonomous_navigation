import multiprocessing
import os
import pickle

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from model import Model
from utils.data_handler import DataHandler
from utils.dataset_with_meta import DatasetWithMeta
from utils.utils import get_rgb_image

DATA_PICKLE_FILE = '/tmp/data_FrRL.pkl'
CREATE_PLOTS = False

LOOKBACK = 3


class ResultTracker:

    def __init__(self):
        self.metrics = {}

    def add_metric(self, name, value, description=None):
        if name not in self.metrics:
            self.metrics[name] = {
                'values': [],
                'description': ''
            }

        self.metrics[name]['values'].append(value)

        # update the description
        if description is not None:
            self.metrics[name]['description'] = description

    def get_metric(self, name):
        return self.metrics[name]

    def print_summary(self):

        for name, data in sorted(self.metrics.items(), key=lambda x: x[0], reverse=True):
            name = name.ljust(20)
            print(f" {name} {np.mean(data['values']):.3f} (std {np.std(data['values']):.3f})  » {data['description']}")


def _feature_extraction(xs, ys):
    # feature extraction
    numerical_features = []

    for state in xs['states']:
        numerical_features.append(state.twist.twist.linear.x)
        numerical_features.append(state.twist.twist.linear.y)
        numerical_features.append(state.twist.twist.linear.z)

        numerical_features.append(state.twist.twist.angular.x)
        numerical_features.append(state.twist.twist.angular.y)
        numerical_features.append(state.twist.twist.angular.z)

    # normalize numerical features using [-2, 2]
    numerical_features = np.array(numerical_features)
    numerical_features = (numerical_features - 2.) / 4.

    # clip the values to [-1, 1]
    numerical_features = np.clip(numerical_features, -1., 1.)

    command = xs['command']

    com_x = command.twist.linear.x
    com_z = command.twist.angular.z

    # TODO: extract actions
    actions = [[com_x, com_z] for _ in range(LOOKBACK)]

    # get images and convert them to RGB
    imgs = []
    times = []
    for img in xs['imgs']:
        imgs.append(get_rgb_image(img))
        times.append(img.header.stamp)

    return np.array(numerical_features), np.array(imgs), np.array(actions), times


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

    command = xs['command']

    ##############################################################
    ##############################################################
    ##############################################################
    ##############################################################

    # map index to time stamps
    idxs = [y.header.stamp.to_sec() for y in ys]

    # set title
    time_stamp = ys[0].header.stamp.to_sec()

    # convert time to humain readable format
    import datetime
    time_stamp = datetime.datetime.fromtimestamp(time_stamp).strftime('%Y-%m-%d %H:%M:%S %f')

    # plot forces using matplotlib
    if CREATE_PLOTS:
        fig, ax = plt.subplots(2, 2, figsize=(15, 10))
        plt1 = ax[0][0]
        plt2 = ax[0][1]
        plt3 = ax[1][0]
        plt4 = ax[1][1]

        # plot vertical lines at command timestamps
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

    ##############################################################
    ##############################################################
    ##############################################################
    ##############################################################

    ##############################################################
    # stumbling / slippering
    # is calculated based on the feet velocities
    ##############################################################

    # calculate the velocity of the feet
    velocity_LF = np.diff(position_LF_z, n=1)
    velocity_RF = np.diff(position_RF_z, n=1)
    velocity_LH = np.diff(position_LH_z, n=1)
    velocity_RH = np.diff(position_RH_z, n=1)

    # calculate the mean velocity
    max_velocity_LF = np.max(np.abs(velocity_LF))
    max_velocity_RF = np.max(np.abs(velocity_RF))
    max_velocity_LH = np.max(np.abs(velocity_LH))
    max_velocity_RH = np.max(np.abs(velocity_RH))

    # check if the mean velocity is above a threshold
    velocity_threshold = 11e-3
    has_stumbling = (max_velocity_LF > velocity_threshold or
                     max_velocity_RF > velocity_threshold or
                     max_velocity_LH > velocity_threshold or
                     max_velocity_RH > velocity_threshold)

    ##############################################################
    # misplaced feets
    # is calculated based on the contact forces
    ##############################################################

    contact_force_threshold = 600
    has_high_contact_force = (max(contact_force_LF_z) > contact_force_threshold or
                              max(contact_force_RF_z) > contact_force_threshold or
                              max(contact_force_LH_z) > contact_force_threshold or
                              max(contact_force_RH_z) > contact_force_threshold)
    has_misplaced_feets = has_high_contact_force

    ##############################################################
    # unable to follow commands
    # is calculated by the difference between the command and the actual twist.linear.x
    ##############################################################
    command_twist_x = command.twist.linear.x
    mean_error_command_twist_x = np.mean(np.abs(np.array(twist_x) - command_twist_x))
    is_unable_to_follow_commands = mean_error_command_twist_x >= 0.18

    ##############################################################
    # report the results
    ##############################################################

    # TODO: split the labels...
    labels = [is_unable_to_follow_commands, has_misplaced_feets, has_stumbling]
    return [labels, labels, labels]


def _prepare_data(item):
    xs, y = item
    return _feature_extraction(xs, y), _label_extraction(y, xs)


def train_model(
        train_loader: DataLoader[DatasetWithMeta],
        test_loader: DataLoader[DatasetWithMeta],
        result_tracker: ResultTracker
):
    print("""

##############################################################
# Training the model
##############################################################


""")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Start training model on device:", device)

    # initialize the model
    first_element: DatasetWithMeta = train_loader.dataset
    num_actions = first_element.target_len()
    model = Model(num_events=num_actions).to(device)
    del first_element

    # train the model
    epochs = 15

    # calc training imbalance
    compensate_imbalance = True
    if compensate_imbalance:
        targets = train_loader.dataset.targets
        class_weights = torch.tensor([
            1. / (targets[:, :, i].sum() / (len(targets) * LOOKBACK)) for i in range(targets.shape[2])
        ]).to(device)

        # apply root transform
        class_weights = torch.sqrt(class_weights)

        # max to 10
        class_weights = torch.clamp(class_weights, 0, 10)
    else:
        class_weights = None

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00001, amsgrad=True, betas=(0.9, 0.999), eps=1e-6)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    model.train()

    print("\nTraining the model...")
    for epoch in range(epochs):

        rolling_mean_loss = None

        model.train()

        pbar = tqdm(train_loader, total=len(train_loader))
        for i, (_input_imgs, _input_scalars, _actions, _target) in enumerate(pbar):
            _input_imgs, _input_scalars, _actions, _target = (
                prepare_on(_input_imgs, _input_scalars, _actions, _target, device))

            logits, res = model.forward(_input_scalars, _input_imgs, actions=_actions)
            loss = loss_fn(logits, _target)

            if rolling_mean_loss is None:
                rolling_mean_loss = loss.item()
            else:
                rolling_mean_loss = (i * rolling_mean_loss + loss.item()) / (i + 1)

            train_mse = torch.nn.functional.mse_loss(res, _target).to('cpu').detach().numpy()
            pbar.set_description(f"{epoch}/{epochs} | LOSS: {loss.item():.6f} | MSE: {train_mse:.3f} | "
                                 f"AVG Loss: {rolling_mean_loss:.3f}")

            # run backpropagation
            loss.backward()
            optimizer.step()

        # calc test loss
        mean_mse = 0
        count = 0
        model.eval()

        avg_loss = 0.

        for _input_imgs, _input_scalars, _actions, _target in test_loader:
            _input_imgs, _input_scalars, _actions, _target = (
                prepare_on(_input_imgs, _input_scalars, _actions, _target, device))

            count += 1

            logits, res = model.forward(_input_scalars, _input_imgs, actions=_actions)
            loss = loss_fn(logits, _target)
            avg_loss += loss.item()
            mean_mse += torch.nn.functional.mse_loss(res, _target).to('cpu').detach().numpy()

        mean_mse /= test_loader.__len__()
        avg_loss /= test_loader.__len__()
        print(f"Epoch: {epoch}/{epochs} | AVG Test Loss: {avg_loss:.6f} | Test MSE: {mean_mse:.3f}")

        old_lr = optimizer.param_groups[0]['lr']
        lr_scheduler.step()
        print(f"Learning rate: {old_lr} -> {optimizer.param_groups[0]['lr']}")

    # save model parameters
    torch.save(model.state_dict(), 'model_params.pth')

    ##############################################################
    # Run Inference
    ##############################################################

    targets = []
    logits = []
    predictions = []

    for _input_imgs, _input_scalars, _actions, _target in test_loader:
        _input_imgs, _input_scalars, _actions, _target = prepare_on(_input_imgs, _input_scalars, _actions, _target,
                                                                    device)

        batch_logits, batch_pred = model.forward(_input_scalars, _input_imgs, _actions)

        logits.append(batch_logits.to('cpu').detach())
        predictions.append(batch_pred.to('cpu').detach())
        targets.append(_target.to('cpu').detach())

    logits = torch.cat(logits, dim=0)
    predictions = torch.cat(predictions, dim=0)
    targets = torch.cat(targets, dim=0)

    mean_prediction = targets.mean(dim=0)
    # repeat the mean prediction for all samples
    mean_prediction = mean_prediction.repeat(predictions.shape[0], 1, 1)

    # print some samples
    for i in range(5):
        print(f"Target: {targets[i]}, Prediction: {predictions[i]}")

    ##############################################################
    # Evaluate the model's performance
    ##############################################################

    # calculate the loss
    loss_fn = torch.nn.BCELoss()
    loss = loss_fn(predictions, targets)
    print(f"Final test loss: {loss.item()}")

    loss = loss_fn(mean_prediction, targets)
    print(f"Final test loss (mean prediction): {loss.item()}")

    calc_matrices(predictions, result_tracker, targets, prefix='Test_')
    calc_matrices(mean_prediction, result_tracker, targets, prefix='MEAN_')

    print("Model trained and evaluated")


def calc_matrices(predictions, result_tracker, targets, prefix=''):
    # calculate the mse
    mse = torch.nn.functional.mse_loss(predictions, targets).to('cpu').detach().numpy()
    result_tracker.add_metric(f'{prefix}MSE', mse, 'Mean Squared Error')
    for threshold in [0.4, 0.6, 0.75, 0.85, 0.9, 0.99]:

        tp = ((predictions > threshold) & (targets > threshold)).float().sum()
        tn = ((predictions <= threshold) & (targets <= threshold)).float().sum()

        accuracy = (tp + tn) / targets.numel()
        result_tracker.add_metric(f'{prefix}ACC@{threshold}', accuracy, f'Accuracy at threshold {threshold}')

        for evnt in range(predictions.shape[2]):
            tp = ((predictions[:, :, evnt] > threshold) & (targets[:, :, evnt] > threshold)).float().sum()
            tn = ((predictions[:, :, evnt] <= threshold) & (targets[:, :, evnt] <= threshold)).float().sum()
            fp = ((predictions[:, :, evnt] > threshold) & (targets[:, :, evnt] <= threshold)).float().sum()
            fn = ((predictions[:, :, evnt] <= threshold) & (targets[:, :, evnt] > threshold)).float().sum()

            precision = tp / (tp + fp + 1e-6)
            recall = tp / (tp + fn + 1e-6)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
            accuracy = (tp + tn) / (tp + tn + fp + fn)

            result_tracker.add_metric(f'{prefix}PRE@{threshold}_{evnt}', precision,
                                      f'Precision at threshold {threshold} for event {evnt}')
            result_tracker.add_metric(f'{prefix}REC@{threshold}_{evnt}', recall,
                                      f'Recall at threshold {threshold} for event {evnt}')
            result_tracker.add_metric(f'{prefix}F1@{threshold}_{evnt}', f1,
                                      f'F1 at threshold {threshold} for event {evnt}')
            result_tracker.add_metric(f'{prefix}ACC@{threshold}_{evnt}', accuracy,
                                      f'Accuracy at threshold {threshold} for event {evnt}')


def prepare_on(_input_imgs, _input_scalars, _actions, _target, device):
    # we only use the first image
    _input_imgs = _input_imgs[0]

    # move everything to device
    _actions = _actions.to(device)
    _input_scalars = _input_scalars.to(device)
    _target = _target.to(device)
    _input_imgs = _input_imgs.to(device)
    return _input_imgs, _input_scalars, _actions, _target


def main():
    # TODO: set limit to -1 to process all data
    # force data extraction and limit the number of samples
    force_data_extraction = False
    limit = 5

    # check if pre-processed data exists in tmp folder
    # or flag to force re-processing
    if not os.path.exists(DATA_PICKLE_FILE) or force_data_extraction:
        extract_data_from_bags(limit=limit)

    # load pre-processed data
    data_set = pickle.load(open(DATA_PICKLE_FILE, 'rb'))
    assert data_set is not None, "Data set is None"

    ##############################################################
    # train the model using the data set
    ##############################################################

    # we train the model 5 times to get a better average of the results
    result_tracker = ResultTracker()
    REPEATS = 1
    for i in range(REPEATS):
        print(f"Data set loaded: {len(data_set)} samples")
        train_set, test_set = train_test_split(data_set, test_size=0.2, shuffle=True)

        # unzip the train_set and convert it to a tensor dataset
        dataset_train = DatasetWithMeta(train_set)
        dataset_loader_train = torch.utils.data.DataLoader(
            dataset_train, batch_size=32, shuffle=True, num_workers=8, pin_memory=True
        )

        dataset_test = DatasetWithMeta(test_set)
        dataset_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=32, shuffle=False)

        report_imbalance(dataset_test, dataset_train)

        train_model(dataset_loader_train, dataset_loader_test, result_tracker)

        ##############################################################
        # print the summary
        ##############################################################
        print("""

##############################################################
# print the summary
##############################################################


    """)
    result_tracker.print_summary()


def report_imbalance(dataset_test, dataset_train):
    # calc dataset imbalance for each target
    print("\n")
    print("Dataset imbalance [train]:")
    targets = dataset_train.targets
    target_names = ['unable_to_follow_commands', 'has_misplaced_feets', 'has_stumbling']
    for i in range(targets.shape[1]):
        print(f" » {target_names[i]} (target  {i}): {targets[:, :, i].sum() / (len(targets) * LOOKBACK) * 100:.2f}%")

    # calc dataset imbalance for each target
    print("\n")
    print("Dataset imbalance [test]:")
    targets = dataset_test.targets
    for i in range(targets.shape[1]):
        print(f" » {target_names[i]} (target  {i}): {targets[:, :, i].sum() / (len(targets) * LOOKBACK) * 100:.2f}%")
    print("\n")


def extract_data_from_bags(limit=-1):
    bags_base_dir = 'data/RosBags/raw/'

    # list all folders in the base directory
    dirs = os.listdir(bags_base_dir)

    dataset = []

    count = 0
    for d in dirs:
        # find jetson, lpc and npc bag files
        jetson_bag_file = None
        lpc_bag_file = None
        npc_bag_file = None

        for i in range(0, 10):

            # check if the limit is reached
            if count == limit:
                break
            count += 1

            for f in os.listdir(bags_base_dir + d):
                if 'jetson' in f and '_' + str(i) + '.bag' in f:
                    jetson_bag_file = bags_base_dir + d + '/' + f
                if 'lpc' in f and '_' + str(i) + '.bag' in f:
                    lpc_bag_file = bags_base_dir + d + '/' + f
                if 'npc' in f and '_' + str(i) + '.bag' in f:
                    npc_bag_file = bags_base_dir + d + '/' + f

            if jetson_bag_file is None or lpc_bag_file is None or npc_bag_file is None:
                continue

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

                # prepare data
                # Use the maximum number of available CPU cores for parallel processing
                num_cores = min(multiprocessing.cpu_count() * 2, 32)
                print("\nPreparing data with multiprocessing n_cors =", num_cores)
                with multiprocessing.Pool(num_cores) as p:
                    training_data = list(
                        tqdm(p.imap(_prepare_data, synchronized_training_data), total=len(synchronized_training_data)))
                print(f"\nTraining data: {len(training_data)} samples")

                dataset.extend(training_data)
                data_handler.report_time()

            except Exception as e:
                print(f"Error processing bags: {e}")

    # total number of samples
    print(f"\nTotal number of samples: {len(dataset)}")

    # save data as a pickle file
    print("\nSaving data to", DATA_PICKLE_FILE)

    # check if the file exists
    if os.path.exists(DATA_PICKLE_FILE):
        os.remove(DATA_PICKLE_FILE)

    pickle.dump(dataset, open(DATA_PICKLE_FILE, 'wb'))
    print("\nData saved to", DATA_PICKLE_FILE)


if __name__ == "__main__":
    main()
