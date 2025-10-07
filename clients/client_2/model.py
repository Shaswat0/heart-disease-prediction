# clients/client_1/model.py

"""
Combined model: CNN branch for ECG images + MLP for tabular features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.config import INPUT_DIM, DEVICE

class CombinedModel(nn.Module):
    def __init__(self, tab_in=13, image_channels=3, image_feat=64, hidden_dim=64, out_dim=2):
        super(CombinedModel, self).__init__()
        # CNN branch
        self.cnn = nn.Sequential(
            nn.Conv2d(image_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x32
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4,4))
        )
        self.cnn_fc = nn.Linear(64 * 4 * 4, image_feat)

        # Tabular MLP branch
        self.mlp = nn.Sequential(
            nn.Linear(tab_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU()
        )

        # Combined head
        self.head = nn.Sequential(
            nn.Linear((hidden_dim//2) + image_feat, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        )

    def forward(self, tabular, image):
        # image: (B, C, H, W)
        img_feat = self.cnn(image)
        img_feat = img_feat.view(img_feat.size(0), -1)
        img_feat = self.cnn_fc(img_feat)

        tab_feat = self.mlp(tabular)
        combined = torch.cat([tab_feat, img_feat], dim=1)
        out = self.head(combined)
        return out
