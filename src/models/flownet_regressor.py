# src/models/flownet_regressor.py
import torch
import torch.nn as nn
from torchvision.models import optical_flow

class FlowNetRegressor(nn.Module):
    def __init__(self, sequence_length=2):
        super().__init__()
        self.sequence_length = sequence_length
        self.flownet = optical_flow.raft_large(pretrained=True)
        
        # Modify input channels based on sequence length
        self.regressor = nn.Sequential(
            nn.Conv2d(2*(sequence_length-1), 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*64*64, 256),
            nn.ReLU(),
            nn.Linear(256, 6)
        )

    def forward(self, imgs):
        """Process sequence of frames"""
        flows = []
        for i in range(1, self.sequence_length):
            flow = self.flownet(imgs[i-1], imgs[i])
            flows.append(flow)
        
        combined_flow = torch.cat(flows, dim=1)
        return self.regressor(combined_flow)