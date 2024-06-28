import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


class DatasetWithMeta(torch.utils.data.Dataset):
    # prepare imgs for resnet
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    def __init__(self, _data):
        super().__init__()

        self.len = len(_data)

        self.imgs = []
        self.metas = []
        self.actions = []

        input_combined, self.targets = zip(*_data)

        # split input data into three lists
        # this is some nasty code, but it is necessary save memory
        input_combined_0 = input_combined[:len(input_combined) // 3]
        input_combined_1 = input_combined[len(input_combined) // 3:2 * len(input_combined) // 3]
        input_combined_2 = input_combined[2 * len(input_combined) // 3:]
        del input_combined

        for (input_scalars, input_img, actions, _) in tqdm(input_combined_0):

            # prepare actions
            self.actions.append(actions)

            # prepare images
            prepared_imgs = []
            for img in input_img:
                prepared_imgs.append(self.preprocess(Image.fromarray(img)).float())
            self.imgs.append(prepared_imgs)
            del prepared_imgs
            self.metas.append(input_scalars)
        del input_combined_0

        for (input_scalars, input_img, actions, _) in tqdm(input_combined_1):

            # prepare actions
            self.actions.append(actions)

            # prepare images
            prepared_imgs = []
            for img in input_img:
                prepared_imgs.append(self.preprocess(Image.fromarray(img)).float())
            self.imgs.append(prepared_imgs)
            del prepared_imgs
            self.metas.append(input_scalars)
        del input_combined_1

        for (input_scalars, input_img, actions, _) in tqdm(input_combined_2):

            # prepare actions
            self.actions.append(actions)

            # prepare images
            prepared_imgs = []
            for img in input_img:
                prepared_imgs.append(self.preprocess(Image.fromarray(img)).float())
            self.imgs.append(prepared_imgs)
            del prepared_imgs
            self.metas.append(input_scalars)
        del input_combined_2

        del _data
        del _

        # convert targets from boolean to float
        self.targets = np.array(self.targets).astype(np.float32)
        self.targets = torch.tensor(self.targets)
        self.imgs = self.imgs

        self.actions = np.array(self.actions).astype(np.float32)

        # convert metas to float tensor
        self.metas = np.array(self.metas).astype(np.float32)
        self.metas = torch.tensor(self.metas)

        print(f"Dataset with {self.len} samples created")

    def __len__(self):
        return self.len

    def meta_len(self):
        return len(self.metas[0])

    def img_len(self):
        return len(self.imgs[0])

    def target_len(self):
        return len(self.targets[0])

    def __getitem__(self, index):
        return self.imgs[index], self.metas[index], self.actions[index], self.targets[index]
