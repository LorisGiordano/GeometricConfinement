import torch
import torch.nn as nn
from monai.networks.blocks import Convolution

class VGG16(nn.Module):
    """
    Implementation of VGG-16 in 3D using MONAI and PyTorch.

    Args:
        in_channels (int): Number of input channels (e.g., 1 for grayscale volumes, 3 for RGB volumes).
        num_classes (int): Number of output classes.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 1000):
        super(VGG16, self).__init__()

        # Helper to create a Conv3d + ReLU block
        def conv3d_block(in_ch, out_ch):
            return Convolution(
                dimensions=3,
                in_channels=in_ch,
                out_channels=out_ch,
                strides=1,
                kernel_size=3,
                act='RELU',
                norm=None,
                dropout=0.0,
                bias=True,
            )

        # VGG-16 configuration in 3D
        self.features = nn.Sequential(
            # Block 1: 2 conv layers
            conv3d_block(in_channels, 64),
            conv3d_block(64, 64),
            nn.MaxPool3d(kernel_size=2, stride=2),

            # Block 2: 2 conv layers
            conv3d_block(64, 128),
            conv3d_block(128, 128),
            nn.MaxPool3d(kernel_size=2, stride=2),

            # Block 3: 3 conv layers
            conv3d_block(128, 256),
            conv3d_block(256, 256),
            conv3d_block(256, 256),
            nn.MaxPool3d(kernel_size=2, stride=2),

            # Block 4: 3 conv layers
            conv3d_block(256, 512),
            conv3d_block(512, 512),
            conv3d_block(512, 512),
            nn.MaxPool3d(kernel_size=2, stride=2),

            # Block 5: 3 conv layers
            conv3d_block(512, 512),
            conv3d_block(512, 512),
            conv3d_block(512, 512),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )

        # Classifier: adaptive pooling + fully connected layers
        self.classifier = nn.Sequential(
            # adapt to 7x7x7 as in original VGG
            nn.AdaptiveAvgPool3d((7, 7, 7)),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    # Example usage:
    model = VGG16(in_channels=1, num_classes=2)
    input_tensor = torch.randn((1, 1, 96, 96, 96))  # batch_size, channels, D, H, W
    output = model(input_tensor)
    print(output.shape)  # Expected output: torch.Size([1, 2])
