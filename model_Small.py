import torch.nn as nn
import torch
from torchvision import models
import torch.nn.functional as F

class CSRNet(nn.Module):
    def __init__(self, load_weights=False):
        super(CSRNet, self).__init__()
        self.seen = 0
        
        # ConvNeXt Small 로드
        # torchvision 공식 구현에서 convnext_small의 마지막 features 출력 채널 수는 768
        convnext = models.convnext_small(pretrained=not load_weights)
        self.frontend = convnext.features

        # convnext_small의 마지막 features 출력 채널 수: 768
        self.backend = nn.Sequential(
            nn.Conv2d(768, 512, kernel_size=3, padding=2, dilation=2),
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
        # ConvNeXt Small: 입력 대비 약 1/32 스케일 출력
        x = self.frontend(x)
        # 1/32 -> 1/8 스케일로 업샘플
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        x = self.backend(x)
        x = self.output_layer(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, std=0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
