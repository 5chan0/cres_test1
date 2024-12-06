import torch.nn as nn
import torch
from torchvision import models
import torch.nn.functional as F

class CSRNet(nn.Module):
    def __init__(self, load_weights=False):
        super(CSRNet, self).__init__()
        self.seen = 0

        # DeepLabv3 모델 로드 (ResNet-50 백본)
        deeplab = models.segmentation.deeplabv3_resnet50(pretrained=not load_weights)
        
        # DeepLabv3는 backbone과 classifier로 구성
        # backbone은 features 추출, classifier는 segmentation map 생성
        # backbone 출력 특징 맵을 CSRNet의 backend로 연결
        # DeepLabv3의 backbone 마지막 출력 채널 수는 2048
        backbone = deeplab.backbone

        self.frontend = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4
        )

        # CSRNet backend (1/8 해상도 정도에 맞춰 dilated conv 유지)
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

        # 필요 시 target 해상도 맞추기 위해 interpolate 유지
        # 단, DeepLabv3는 이미 dilated conv를 사용해 해상도가 이전보다 보존됨
        # 상황에 따라 이 부분을 제거하거나 조정 가능
        #target_size = (x.size(2) - 1, x.size(3))
        #x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, std=0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
