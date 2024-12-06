import torch.nn as nn
import torch
from torchvision import models
import torch.nn.functional as F


class CSRNet(nn.Module):
    def __init__(self, load_weights=False):
        super(CSRNet, self).__init__()
        self.seen = 0

        # 기존 ResNet-50 사용 부분 제거
        # resnet = models.resnet50(pretrained=not load_weights)
        # modules = list(resnet.children())[:-2]

        # dilation 적용 부분 삭제 (DenseNet에는 해당 레이어 구조가 다름)
        # resnet.layer3.apply(self._nostride_dilate(2))
        # resnet.layer4.apply(self._nostride_dilate(4))

        # DenseNet-121 사용
        densenet = models.densenet121(pretrained=not load_weights)
        # DenseNet의 feature 추출 파트만 사용
        self.frontend = densenet.features

        # DenseNet-121의 마지막 feature 출력 채널 수는 1024
        # 이에 맞춰 backend 첫 Conv 레이어의 입력 채널 수정 (기존 2048 -> 1024)
        self.backend = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=2, dilation=2),
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

        # interpolate 적용 부분은 그대로 유지 (필요에 따라 조정 가능)
        target_size = (x.size(2) - 1, x.size(3))
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, std=0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

    def _nostride_dilate(self, dilate):
        # 이전에 사용하던 dilation 함수는 더 이상 필요 없음
        # 필요 시 DenseNet의 특정 레이어에 dilation을 적용하는 별도 로직을 구현할 수 있으나
        # 기본 DenseNet 구조로도 충분히 시도 가능
        def apply_layer(m):
            pass
        return apply_layer
