import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """
    Conv2d → BN → ReLU → Conv2d → BN → ReLU
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)

class PatchClassifier(nn.Module):
    def __init__(self, in_channels: int = 6):
        super().__init__()
        self.encoder = nn.Sequential(
            DoubleConv(in_channels, 64),
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(64, 128),
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(128, 256),
            nn.AdaptiveAvgPool2d(output_size=1)
        )
        self.fc = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        return logits