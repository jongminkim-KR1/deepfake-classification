import numpy as np
import torch
import torch.nn as nn


def fft_2d(img_rgb: torch.Tensor):

    if img_rgb.shape[-3] != 3:
        raise ValueError(f"입력 이미지의 채널 차원(-3)이 3이 아닙니다. (..., 3, H, W) 형태가 필요합니다. 현재 shape: {img_rgb.shape}")
        
    device = img_rgb.device
    dtype = img_rgb.dtype

    # Y 채널 계산을 위한 가중치 (ITU-R BT.601 표준)
    # (3,) 텐서를 생성
    weights_y = torch.tensor([0.299, 0.587, 0.114], device=device, dtype=dtype)
    
    # 가중치 텐서를 입력 텐서와 브로드캐스팅 가능하도록 shape 변경
    # (3,) -> (3, 1, 1) 또는 (1, 3, 1, 1) 등으로 변환
    
    # 입력 텐서의 차원 수에 맞게 shape을 동적으로 조정
    # (B, 3, H, W) 라면 -> [1, 3, 1, 1]
    # (3, H, W) 라면 -> [3, 1, 1]
    weight_shape = [1] * img_rgb.dim() # 예: [1, 1, 1, 1] 또는 [1, 1, 1]
    channel_dim = -3 # 채널 차원의 인덱스
    weight_shape[channel_dim] = 3 # 예: [1, 3, 1, 1] 또는 [3, 1, 1]
    
    weights_y = weights_y.view(*weight_shape)

    # Y 채널 계산
    # (..., 3, H, W) * (..., 3, 1, 1) -> (..., 3, H, W) (각 채널에 가중치 곱셈)
    # torch.sum(..., dim=-3) -> (..., H, W) (채널 차원을 합산)
    Y = torch.sum(img_rgb * weights_y, dim=channel_dim)
    
    # Y 채널에 대한 FFT 수행
    # Y 텐서 (..., H, W)에 대해 fft2 수행
    fft_result = torch.fft.fft2(Y)

    # 주파수 영역의 중심을 0으로 이동
    fft_shifted = torch.fft.fftshift(fft_result)

    # 크기(Magnitude)와 위상(Phase) 계산
    magnitude = torch.abs(fft_shifted)
    phase = torch.angle(fft_shifted)
    
    return magnitude, phase

class CNNClfWithFFT(nn.Module):
    def __init__(self, num_classes : int = 2):
        super(CNNClfWithFFT, self).__init__()
        input_size = 224
        layer_size = (32, 64, 64, 32)
        filter_size = (7, 5, 3)
        padding_size = (3, 2, 1)

        self.rgb_branch = nn.Sequential(
            nn.Conv2d(3, layer_size[0], kernel_size=filter_size[0], stride=1, padding=padding_size[0]),
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

            nn.Conv2d(layer_size[2], layer_size[2], kernel_size=filter_size[2], stride=1, padding=padding_size[2]),
            nn.BatchNorm2d(layer_size[2]),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.magnitude_branch = nn.Sequential(
            nn.Conv2d(1, layer_size[0], kernel_size=filter_size[0], stride=1, padding=padding_size[0]),
            nn.BatchNorm2d(layer_size[0]),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
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
            nn.MaxPool2d(2),
            
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

        fused = torch.cat([f_rgb.flatten(1), f_mag.flatten(1), f_pha.flatten(1)], dim=1)

        output = self.fc(fused)

        return output