import torch.nn as nn
import torch
from torchvision import models
import torch.nn.functional as F

class CSRNet(nn.Module):
    def __init__(self, load_weights=False):
        super(CSRNet, self).__init__()
        self.seen = 0
        
        # ConvNeXt X-Large 모델 로드 (최신 torchvision에서 지원)
        convnext = models.convnext_xlarge(pretrained=not load_weights)
        self.frontend = convnext.features

        # ConvNeXt X-Large의 마지막 feature 출력 채널 수는 2048
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
        # ConvNeXt features 추출 (1/32 scale)
        x = self.frontend(x)
        
        # 1/32 -> 1/8로 업샘플 (scale_factor=4)
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        
        # backend 처리 (해상도 유지)
        x = self.backend(x)
        x = self.output_layer(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, std=0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
