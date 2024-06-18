import numpy as np
import torch
from PIL import Image
from torchvision import transforms


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

        for (input_scalars, input_img, actions, _) in input_combined:

            # prepare actions
            self.actions.append(actions.astype(np.float32))

            # prepare images
            prepared_imgs = []
            for img in input_img:
                prepared_imgs.append(self.preprocess(Image.fromarray(img)).float())
            self.imgs.append(prepared_imgs)
            del prepared_imgs
            self.metas.append(input_scalars)

        del input_combined
        del _data
        del _

        # convert targets from boolean to float
        self.targets = np.array(self.targets).astype(np.float32)
        self.targets = torch.tensor(self.targets)

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
