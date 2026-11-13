from torch import nn


class FCN(nn.Module):
    """
    Input: (batch, channels, time). Output: (batch, classes).
    
    From "Time Series Classification from Scratch with Deep Neural Networks: A Strong Baseline" 
    """

    def __init__(self, in_channels=6, num_classes=5):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 128, kernel_size=8, padding="same"),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, 256, kernel_size=5, padding="same"),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Conv1d(256, 128, kernel_size=3, padding="same"),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)       # (batch, 128, time)
        x = self.pool(x)           # (batch, 128, 1)
        x = x.squeeze(-1)          # (batch, 128)
        return self.classifier(x)  # (batch, num_classes)