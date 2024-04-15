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


def _feature_extraction(X):
    # feature extraction
    numerical_features = []

    for state, t in X['states']:
        numerical_features.append(state.twist.twist.linear.x)
        numerical_features.append(state.twist.twist.linear.y)
        numerical_features.append(state.twist.twist.linear.z)

    # get images and convert them to RGB
    imgs = []
    for img, t in X['imgs']:
        imgs.append(get_rgb_image(img))

    return np.array(numerical_features), np.array(imgs)


def _label_extraction(y):
    # label extraction

    # TODO: implement label extraction
    return [1, 0, 0, 0, 1, 0, 1, 0, 0, 0]


def _prepare_data(item):
    X, y = item
    return (_feature_extraction(X), _label_extraction(y))


def train_model(data_set):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Start training model on device:", device)

    # split data into training and validation
    (inputs, targets) = zip(*data_set)
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
    epochs = 2

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, amsgrad=True, betas=(0.9, 0.999), eps=1e-6)
    loss_fn = torch.nn.MSELoss()
    model.train()

    print("\nTraining the model...")
    for epoch in range(epochs):

        pbar = tqdm(zip(pre_inputs_scalar_batches, pre_inputs_images_batches, pre_targets_batches),
                    total=len(pre_inputs_scalar_batches))

        for (_input_scalars_batch, _input_imgs_batch, _target_batch) in pbar:
            res = model.forward(_input_scalars_batch, _input_imgs_batch)
            loss = loss_fn(res, _target_batch)

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

        res = model.forward(_input_scalars_batch, _input_imgs_batch)

        if count == 1:
            print("\n")
            print("Example from the first batch:")
            print(f" » target: {_target_batch[0].to('cpu').detach().numpy()}")
            print(f" » result: {res[0].to('cpu').detach().numpy()}")
            print("\n")

        mean_mse += torch.nn.functional.mse_loss(res, _target_batch).to('cpu').detach().numpy()

    mean_mse /= len(inputs_scalar)
    print(f"Mean MSE: {mean_mse}")


def main():
    force_data_extraction = False

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
    bags_base_dir = 'data/20230302_Hoengg_Forst_Dodo/mission_data/2023-03-02-10-03-50/'
    data_handler = DataHandler(
        jetson_bag_file=bags_base_dir + '2023-03-02-10-03-50_anymal-d020-jetson_mission_0.bag',
        lpc_bag_file=bags_base_dir + '2023-03-02-10-03-50_anymal-d020-lpc_mission_0.bag',
        npc_bag_file=bags_base_dir + '2023-03-02-10-03-50_anymal-d020-npc_mission_0.bag'
    )

    data_handler.load_data()
    data_handler.report_frequencies()

    # get synchronized data
    print("\nGetting synchronized data")
    synchronized_training_data = data_handler.get_synchronized_dataset(limit=None)

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


if __name__ == "__main__":
    main()
