import torch
from torch import nn


class BiLSTM(nn.Module):
    def __init__(self, in_channels=6, hidden_size=100, num_classes=5):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.classifier = nn.Linear(2 * hidden_size, num_classes)

    def forward(self, x):
        # Your loaders provide (batch, channels, time).
        x = x.transpose(1, 2)  # (batch, time, channels)

        _, (hidden, _) = self.lstm(x)

        # Final states from the forward and backward directions.
        features = torch.cat(
            (hidden[-2], hidden[-1]), dim=1
        )  # (batch, 200)

        return self.classifier(features)  # (batch, 5) logits