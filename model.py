import torch.nn as nn
import torch
from torchvision import models
import torch.nn.functional as F  # 추가 필요


class CSRNet(nn.Module):
    def __init__(self, load_weights=False):
        super(CSRNet, self).__init__()
        self.seen = 0

        # ResNet-50 모델을 로드하고 필요한 레이어를 수정합니다.
        resnet = models.resnet50(pretrained=not load_weights)

        # avgpool과 fc 레이어를 제거하여 필요한 특징 추출 부분만 사용합니다.
        modules = list(resnet.children())[:-2]

        # 레이어 3와 레이어 4에서 dilated convolution을 적용하여 공간 해상도를 유지합니다.
        resnet.layer3.apply(self._nostride_dilate(2))
        resnet.layer4.apply(self._nostride_dilate(4))

        self.frontend = nn.Sequential(*modules)

        # backend를 정의합니다.
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

      # 출력 크기를 타겟 크기와 맞춤
      target_size = (x.size(2) - 1, x.size(3))  # 타겟과 동일한 크기로 조정
      x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)

      return x

    def _initialize_weights(self):
        # 초기 가중치 설정 (필요한 경우)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, std=0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

    def _nostride_dilate(self, dilate):
        # 레이어의 stride를 1로 설정하고 dilation을 적용하는 함수
        def apply_layer(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                if m.stride == (2, 2):
                    m.stride = (1, 1)
                    if m.kernel_size == (3, 3):
                        m.dilation = (dilate, dilate)
                        m.padding = (dilate, dilate)
        return apply_layer
