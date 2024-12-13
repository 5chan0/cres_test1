import torch.nn as nn
import torch
from torchvision import models
import torch.nn.functional as F

class CSRNet(nn.Module):
    def __init__(self, load_weights=False):
        super(CSRNet, self).__init__()
        self.seen = 0
        
        # EfficientNet B5 모델 로드
        # pretrained 인자를 통해 가중치 로드 여부를 결정
        effnet = models.efficientnet_b5(pretrained=not load_weights)

        # EfficientNet의 features 부분을 frontend로 사용
        self.frontend = effnet.features

        # EfficientNet B5의 마지막 features 출력 채널 수가 2048이라 가정 (torchvision 공식 구현 참고)
        # 기존 backend 유지
        self.backend = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
        )

        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        if load_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, std=0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
