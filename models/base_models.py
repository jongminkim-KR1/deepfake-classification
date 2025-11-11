# model.py (가정)
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms # [추가] Grayscale 변환을 위해 import

def fft_2d(img: torch.Tensor):
    # ... (원본과 동일) ...
    fft_result = torch.fft.fft2(img)
    fft_shifted = torch.fft.fftshift(fft_result)
    magnitude = torch.abs(fft_shifted)
    phase = torch.angle(fft_shifted)
    return magnitude, phase

class CNNClfWithFFT(nn.Module):
    def __init__(self, num_classes : int = 2):
        # ... (원본 __init__과 동일) ...
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
            nn.MaxPool2d(2),

            nn.Conv2d(layer_size[2], layer_size[2], kernel_size=filter_size[2], stride=1, padding=padding_size[2]),
            nn.BatchNorm2d(layer_size[2]),
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

            nn.Conv2d(layer_size[1], layer_size[3], kernel_size=filter_size[2], stride=1, padding=padding_size[2]),
            nn.BatchNorm2d(layer_size[3]),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(layer_size[3], layer_size[3], kernel_size=filter_size[2], stride=1, padding=padding_size[2]),
            nn.BatchNorm2d(layer_size[3]),
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

            nn.Conv2d(layer_size[1], layer_size[3], kernel_size=filter_size[2], stride=1, padding=padding_size[2]),
            nn.BatchNorm2d(layer_size[3]),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(layer_size[3], layer_size[3], kernel_size=filter_size[2], stride=1, padding=padding_size[2]),
            nn.BatchNorm2d(layer_size[3]),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        fc_input_size = layer_size[2] + 2*layer_size[3]
        image_size = input_size // 8

        self.fc = nn.Sequential(
            nn.Linear(fc_input_size*image_size*image_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        # x shape: [B, 3, 512, 512]

        # --- [수정 1] FFT를 위해 그레이스케일 이미지 생성 ---
        # (B, 3, 512, 512) -> (B, 1, 512, 512)
        x_gray = transforms.Grayscale(num_output_channels=1)(x)

        # --- [수정 2] 그레이스케일 이미지로 FFT 수행 ---
        x_mag, x_pha = fft_2d(x_gray) 
        # x_mag, x_pha shape: [B, 1, 512, 512]

        # 3개 브랜치 처리
        f_rgb = self.rgb_branch(x)      # 원본 3채널 x 사용
        f_mag = self.magnitude_branch(x_mag) # 1채널 x_mag 사용
        f_pha = self.phase_branch(x_pha)   # 1채널 x_pha 사용

        # 피처맵 퓨전
        fused = torch.cat([f_rgb, f_mag, f_pha], dim=1) # [B, 128, 64, 64]

        # --- [수정 3] FC 레이어에 입력하기 전 flatten ---
        fused_flat = torch.flatten(fused, 1) # [B, 128 * 64 * 64]

        output = self.fc(fused_flat)
        # --- [수정 끝] ---

        return output