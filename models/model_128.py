import numpy as np
import torch
import torch.nn as nn
from base_models import fft_2d

class CNNClfWithFFT128(nn.Module):
    def __init__(self, num_classes : int = 2):
        super(CNNClfWithFFT128, self).__init__()
        input_size = 512
        layer_size = (64, 64, 128, 32)
        filter_size = (7, 5, 3, 3)
        padding_size = (3, 2, 1, )

        self.rgb_branch = nn.Sequential(
            nn.Conv2d(3, layer_size[0], kernel_size=filter_size[0], stride=1, padding=padding_size[0]),
            nn.BatchNorm2d(layer_size[0]),
            nn.ReLU(),
            
            nn.Conv2d(layer_size[0], layer_size[1], kernel_size=filter_size[1], stride=1, padding=padding_size[1]),
            nn.BatchNorm2d(layer_size[1]),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(layer_size[1], layer_size[2], kernel_size=filter_size[2], stride=2, padding=padding_size[2]),
            nn.BatchNorm2d(layer_size[2]),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(layer_size[2], layer_size[3], kernel_size=filter_size[3], stride=1, padding=padding_size[2]),
            nn.BatchNorm2d(layer_size[3]),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.magnitude_branch = nn.Sequential(
            nn.Conv2d(1, layer_size[0], kernel_size=filter_size[0], stride=1, padding=padding_size[0]),
            nn.BatchNorm2d(layer_size[0]),
            nn.ReLU(),
            
            nn.Conv2d(layer_size[0], layer_size[1], kernel_size=filter_size[1], stride=1, padding=padding_size[1]),
            nn.BatchNorm2d(layer_size[1]),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(layer_size[1], layer_size[2], kernel_size=filter_size[2], stride=2, padding=padding_size[2]),
            nn.BatchNorm2d(layer_size[2]),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(layer_size[2], layer_size[3] // 2, kernel_size=filter_size[3], stride=1, padding=padding_size[2]),
            nn.BatchNorm2d(layer_size[3] // 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.phase_branch = nn.Sequential(
            nn.Conv2d(1, layer_size[0], kernel_size=filter_size[0], stride=1, padding=padding_size[0]),
            nn.BatchNorm2d(layer_size[0]),
            nn.ReLU(),
            
            nn.Conv2d(layer_size[0], layer_size[1], kernel_size=filter_size[1], stride=1, padding=padding_size[1]),
            nn.BatchNorm2d(layer_size[1]),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(layer_size[1], layer_size[2], kernel_size=filter_size[2], stride=2, padding=padding_size[2]),
            nn.BatchNorm2d(layer_size[2]),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(layer_size[2], layer_size[3] // 2, kernel_size=filter_size[3], stride=1, padding=padding_size[2]),
            nn.BatchNorm2d(layer_size[3] // 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        fc_input_size = 2*layer_size[3]
        image_size = input_size // 16

        self.fc = nn.Sequential(
            nn.Linear(fc_input_size*image_size*image_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x_mag, x_pha = fft_2d(x)

        f_rgb = self.rgb_branch(x)
        f_mag = self.magnitude_branch(x_mag)
        f_pha = self.phase_branch(x_pha)

        fused = torch.cat([f_rgb, f_mag, f_pha], dim=1)

        output = self.fc(fused)

        return output