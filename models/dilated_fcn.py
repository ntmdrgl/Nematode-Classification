from torch import nn


class DilatedFCN(nn.Module):
    """Postural-map classifier from Javer et al., Fig. 3 (ECCV workshops 2018).

    Input: (batch, in_channels, time). Output: (batch, num_classes) logits.
    Variables form a spatial axis, not Conv2d input channels. Symmetric padding
    allows even a single variable; strided convolutions round sizes upward.
    The paper does not specify padding or LeakyReLU slope, so these use
    same-size padding at stride 1 and slope 0.01. No unspecified residual
    shortcuts are added to the convolution sequence shown in the figure.
    """

    def __init__(self, in_channels=6, num_classes=5):
        super().__init__()
        if in_channels < 1 or num_classes < 1:
            raise ValueError("in_channels and num_classes must be positive.")

        def block(inputs, outputs, kernel_size=3, stride=(1, 1), dilation=(1, 1)):
            padding = tuple((kernel_size // 2) * value for value in dilation)
            return nn.Sequential(
                nn.Conv2d(
                    inputs, outputs, kernel_size,
                    stride=stride, padding=padding, dilation=dilation,
                ),
                nn.BatchNorm2d(outputs),
                nn.LeakyReLU(negative_slope=0.01),
            )

        self.features = nn.Sequential(
            nn.Unflatten(1, (1, in_channels)),  # (batch, 1, variables, time)
            block(1, 16, kernel_size=7),
            block(16, 32, stride=(2, 2)),
            block(32, 64, stride=(2, 2)),
            block(64, 128, stride=(1, 2)),
            block(128, 256, stride=(1, 2)),
            block(256, 512, stride=(1, 2)),
            block(512, 512, dilation=(1, 2)),
            block(512, 512, dilation=(1, 4)),
            block(512, 512, dilation=(1, 2)),
            block(512, 512, dilation=(1, 1)),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Conv2d(512, num_classes, kernel_size=1)

    def forward(self, x):
        features = self.features(x)
        return self.classifier(self.pool(features)).flatten(1)
