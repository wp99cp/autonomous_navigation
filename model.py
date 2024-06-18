import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights


class Model(nn.Module):

    def __init__(self, num_events: int):
        super().__init__()
        self.lookback = 3

        weights = ResNet18_Weights.IMAGENET1K_V1
        self.resnet18 = models.resnet18(weights=weights)

        self.action_size = 2
        self.feature_size = 330
        self.n_concat = 512 + 96  # 512 from the resnet18 and 15 from the scalar features

        self.dfeat = nn.Linear(self.feature_size, 96)

        self.act0 = nn.Linear(self.action_size, 32)
        self.act1 = nn.Linear(32, 32)

        # activation function
        self.activation = nn.ReLU()

        # define the dense layers
        self.fc0 = nn.Linear(1000, 512)
        self.fc1 = nn.Linear(self.n_concat, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 128)

        self.lstm = nn.LSTM(input_size=128 + 32, hidden_size=128, num_layers=1, batch_first=True)
        self.fc4 = nn.Linear(128, num_events)

        self.dropout1 = nn.Dropout(0.4)
        self.dropout2 = nn.Dropout(0.4)
        self.dropout3 = nn.Dropout(0.4)

        self.batch_norm = nn.BatchNorm1d(self.n_concat)

    def forward(self, scalar_features, image, actions):
        """
        We expect the action to be a tensor of shape (batch_size, lookback, action_size)
        """

        image = self.resnet18(image)
        image = self.activation(self.fc0(image))

        scalar_features = self.activation(self.dfeat(scalar_features))

        # combine the image features with the scalar features
        combined_features = torch.cat((scalar_features, image), dim=1)
        assert combined_features.shape[
                   1] == self.n_concat, f"Have you changed the number of features? Expected {self.n_concat} but got {combined_features.shape[1]}"

        # add dense layers
        combined_features = self.activation(self.fc1(combined_features))
        combined_features = self.dropout1(combined_features)

        combined_features = self.activation(self.fc2(combined_features))
        combined_features = self.dropout2(combined_features)

        combined_features = self.activation(self.fc3(combined_features))
        before_lstm = self.dropout3(combined_features)

        # apply the action features in for every timestep
        actions = self.activation(self.act0(actions))
        actions = self.activation(self.act1(actions))

        before_lstm = before_lstm.unsqueeze(1).repeat(1, self.lookback, 1)
        before_lstm = torch.cat((before_lstm, actions), dim=2)

        # apply the lstm
        lstm_out, _ = self.lstm(before_lstm)

        # apply the final dense layer
        logits = self.fc4(lstm_out)

        # apply sigmoid function element-wise
        output = torch.sigmoid(logits)
        return logits, output
