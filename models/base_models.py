import numpy as np
import torch
import torch.nn as nn

class CNNClfWithFFT(nn.Module):
    def __init__(self, num_classes=2, input_size=):
        super(CNNClfWithFFT, self).__init__()

        self.rgb_branch = nn.Sequential(
            
        )

        self.magnitude_branch = nn.Sequential(

        )

        self.phase_branch = nn.Sequential(

        )
    
    def forward(self, x):
        return