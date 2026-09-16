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
    def __init__(self, input_size, hidden_sizes, kernel_sizes, output_size, dropout=0.2):
        super().__init__()

        if len(hidden_sizes) != len(kernel_sizes):
            raise ValueError("hidden_sizes and kernel_sizes must have the same length")

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.kernel_sizes = kernel_sizes
        self.output_size = output_size

        layers = []
        in_channels = input_size

        for hidden_size, kernel_size in zip(hidden_sizes, kernel_sizes):
            layers.append(
                ConvBlock(
                    in_channels=in_channels,
                    out_channels=hidden_size,
                    kernel_size=kernel_size,
                    dropout=dropout,
                )
            )
            in_channels = hidden_size

        self.features = nn.Sequential(*layers)

        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Classification layer
        self.classifier = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x):
        """
        x: (batch_size, seq_length, input_size)
        """

        # Conv1d expects (batch, channels, seq_length)
        x = x.permute(0, 2, 1)

        x = self.features(x)

        # Global Average Pooling
        x = self.global_pool(x)      # (batch, channels, 1)
        x = x.squeeze(-1)            # (batch, channels)

        logits = self.classifier(x)

        return logits