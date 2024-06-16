import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights


class Model(nn.Module):

    def __init__(self, num_actions: int):
        super().__init__()

        weights = ResNet18_Weights.IMAGENET1K_V1
        self.resnet18 = models.resnet18(weights=weights)

        self.n_concat = 512 + 30  # 1000 from the resnet18 and 15 from the scalar features

        # define the dense layers
        self.fc0 = nn.Linear(1000, 512)
        self.fc1 = nn.Linear(self.n_concat, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_actions)

        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)

        self.batch_norm = nn.BatchNorm1d(self.n_concat)

    def forward(self, scalar_features, image):
        image = self.resnet18(image)
        image = self.fc0(image)

        # combine the image features with the scalar features
        combined_features = torch.cat((scalar_features, image), dim=1)

        assert combined_features.shape[
                   1] == self.n_concat, f"Have you changed the number of features? Expected {self.n_concat} but got {combined_features.shape[1]}"

        # add dense layers
        combined_features = self.fc1(combined_features)
        combined_features = self.dropout1(combined_features)
        combined_features = self.fc2(combined_features)
        combined_features = self.dropout2(combined_features)

        logits = self.fc3(combined_features)

        # apply sigmoid function element-wise
        output = torch.sigmoid(logits)

        return logits, output
