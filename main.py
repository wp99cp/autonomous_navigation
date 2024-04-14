import multiprocessing

import numpy as np
from tqdm import tqdm

from utils.data_handler import DataHandler
from utils.utils import get_rgb_image


def _feature_extraction(X):
    # feature extraction
    numerical_features = []

    # get images and convert them to RGB
    imgs = []
    for img, t in X['imgs']:
        imgs.append(get_rgb_image(img))

    return np.array(numerical_features), np.array(imgs)


def _label_extraction(y):
    # label extraction

    # TODO: implement label extraction
    return [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]


def _prepare_data(item):
    X, y = item
    return (_feature_extraction(X), _label_extraction(y))


def main():
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

    # report the time taken for each method
    data_handler.report_time()


if __name__ == "__main__":
    main()
