import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dropout=0.2):
        super().__init__()

        padding = kernel_size // 2

        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.block(x)


class FCN(nn.Module):
    def __init__(self, in_channels, hidden_size, kernel_size, output_size, dropout=0.2):
        super().__init__()

        if len(hidden_size) != len(kernel_size):
            raise ValueError("hidden_size and kernel_size must have the same length")

        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.output_size = output_size

        layers = []
        in_channels = in_channels

        for h_size, kernel_size in zip(hidden_size, kernel_size):
            layers.append(
                ConvBlock(
                    in_channels=in_channels,
                    out_channels=h_size,
                    kernel_size=kernel_size,
                    dropout=dropout,
                )
            )
            in_channels = h_size

        self.features = nn.Sequential(*layers)

        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Classification layer
        self.classifier = nn.Linear(hidden_size[-1], output_size)

    def forward(self, x):
        """
        x: (batch_size, seq_length, in_channels)
        """

        # Conv1d expects (batch, channels, seq_length)
        x = x.permute(0, 2, 1)

        x = self.features(x)

        # Global Average Pooling
        x = self.global_pool(x)      # (batch, channels, 1)
        x = x.squeeze(-1)            # (batch, channels)

        logits = self.classifier(x)

        return logits