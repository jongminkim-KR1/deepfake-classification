import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
import torchvision.transforms.functional as F

def fft_2d(img_rgb: torch.Tensor):

    if img_rgb.shape[-3] != 3:
        raise ValueError(f"입력 이미지의 채널 차원(-3)이 3이 아닙니다. (..., 3, H, W) 형태가 필요합니다. 현재 shape: {img_rgb.shape}")
        
    y_cb_cr = F.rgb_to_ycbcr(img_rgb)
    # Y 채널 추출 (첫 번째 채널)
    Y = y_cb_cr.select(dim=-3, index=0) # (..., H, W)
    
    fft_result = torch.fft.fft2(Y)

    # 주파수 영역의 중심을 0으로 이동
    fft_shifted = torch.fft.fftshift(fft_result)

    # 크기(Magnitude)와 위상(Phase) 계산
    magnitude = torch.abs(fft_shifted)
    phase = torch.angle(fft_shifted)

    magnitude = magnitude.unsqueeze(-3) # (..., 1, H, W)
    phase = phase.unsqueeze(-3)       # (..., 1, H, W)
    
    return magnitude, phase


class DeepclfwithResNet(nn.Module):
    def __init__(self, num_classes :  int = 2):
        super(DeepclfwithResNet, self).__init__()
        input_size = 224
        layer_size = (32, 64, 64, 128, 256)

        main_backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.rgb_branch = nn.Sequential(*list(main_backbone.children())[:-2])
        rgb_output_channels = 512

        filter_size = (7, 5, 3, 3, 3)
        padding_size = (3, 2, 1, 1, 1)

        self.magnitude_branch = nn.Sequential(
            nn.Conv2d(1, layer_size[0], kernel_size=filter_size[0], stride=1, padding=padding_size[0]),
            nn.BatchNorm2d(layer_size[0]),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(layer_size[0], layer_size[1], kernel_size=filter_size[1], stride=1, padding=padding_size[1]),
            nn.BatchNorm2d(layer_size[1]),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(layer_size[1], layer_size[2], kernel_size=filter_size[2], stride=1, padding=padding_size[2]),
            nn.BatchNorm2d(layer_size[2]),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(layer_size[2], layer_size[3], kernel_size=filter_size[3], stride=1, padding=padding_size[3]),
            nn.BatchNorm2d(layer_size[3]),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(layer_size[3], layer_size[4], kernel_size=filter_size[4], stride=1, padding=padding_size[4]),
            nn.BatchNorm2d(layer_size[4]),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.phase_branch = nn.Sequential(
            nn.Conv2d(1, layer_size[0], kernel_size=filter_size[0], stride=1, padding=padding_size[0]),
            nn.BatchNorm2d(layer_size[0]),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(layer_size[0], layer_size[1], kernel_size=filter_size[1], stride=1, padding=padding_size[1]),
            nn.BatchNorm2d(layer_size[1]),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(layer_size[1], layer_size[2], kernel_size=filter_size[2], stride=1, padding=padding_size[2]),
            nn.BatchNorm2d(layer_size[2]),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(layer_size[2], layer_size[3], kernel_size=filter_size[3], stride=1, padding=padding_size[3]),
            nn.BatchNorm2d(layer_size[3]),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(layer_size[3], layer_size[4], kernel_size=filter_size[4], stride=1, padding=padding_size[4]),
            nn.BatchNorm2d(layer_size[4]),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        fc_input_size = rgb_output_channels + 2*layer_size[4]
        image_size = input_size // 32
        
        self.fc = nn.Sequential(
            nn.Linear(fc_input_size * image_size * image_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x_mag, x_pha = fft_2d(x)

        f_rgb = self.rgb_branch(x)
        f_mag = self.magnitude_branch(x_mag)
        f_pha = self.phase_branch(x_pha)

        fused = torch.cat([f_rgb.flatten(1), f_mag.flatten(1), f_pha.flatten(1)], dim=1)

        output = self.fc(fused)

        return output
