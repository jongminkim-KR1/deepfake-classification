import numpy as np
import torch
import torch.nn as nn

def fft_2d(img: torch.Tensor):
    fft_result = torch.fft.fft2(img)

    fft_shifted = torch.fft.fftshift(fft_result)

    magnitude = torch.abs(fft_shifted)
    phase = torch.angle(fft_shifted)
    return magnitude, phase

class CNNClfWithFFT(nn.Module):
    def __init__(self, num_classes : int = 2):
        super(CNNClfWithFFT, self).__init__()
        input_size = 512
        layer_size = (32, 64, 64, 32)
        filter_size = (7, 5, 3)
        padding_size = (3, 2, 1)

        self.rgb_branch = nn.Sequential(
            nn.Conv2d(3, layer_size[0], kernel_size=filter_size[0], stride=1, padding=padding_size[0]),
            nn.BatchNorm2d(layer_size[0]),
            nn.ReLU(),
            
            nn.Conv2d(layer_size[0], layer_size[1], kernel_size=filter_size[1], stride=1, padding=padding_size[1]),
            nn.BatchNorm2d(layer_size[1]),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(layer_size[1], layer_size[2], kernel_size=filter_size[2], stride=1, padding=padding_size[2]),
            nn.BatchNorm2d(layer_size[2]),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.magnitude_branch = nn.Sequential(
            nn.Conv2d(3, layer_size[0], kernel_size=filter_size[0], stride=1, padding=padding_size[0]),
            nn.BatchNorm2d(layer_size[0]),
            nn.ReLU(),
            
            nn.Conv2d(layer_size[0], layer_size[1], kernel_size=filter_size[1], stride=1, padding=padding_size[1]),
            nn.BatchNorm2d(layer_size[1]),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(layer_size[1], layer_size[3], kernel_size=filter_size[2], stride=1, padding=padding_size[2]),
            nn.BatchNorm2d(layer_size[3]),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.phase_branch = nn.Sequential(
            nn.Conv2d(3, layer_size[0], kernel_size=filter_size[0], stride=1, padding=padding_size[0]),
            nn.BatchNorm2d(layer_size[0]),
            nn.ReLU(),
            
            nn.Conv2d(layer_size[0], layer_size[1], kernel_size=filter_size[1], stride=1, padding=padding_size[1]),
            nn.BatchNorm2d(layer_size[1]),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(layer_size[1], layer_size[3], kernel_size=filter_size[2], stride=1, padding=padding_size[2]),
            nn.BatchNorm2d(layer_size[3]),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        fc_input_size = layer_size[2] + 2*layer_size[3]
        image_size = input_size // 4


        self.fc = nn.Sequential(
            nn.Linear(fc_input_size*image_size*image_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x_mag, x_pha = fft_2d(x)

        f_rgb = self.rgb_branch(x)
        f_mag = self.magnitude_branch(x)
        f_pha = self.phase_branch(x)

        fused = torch.cat([f_rgb, f_mag, f_pha], dim=1)

        output = self.fc(fused)

        return output