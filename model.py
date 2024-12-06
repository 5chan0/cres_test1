import torch.nn as nn
import torch
import torch.nn.functional as F
# HRNet 백본을 로드하기 위한 추가 import (아래 2번에서 설명)
from hrnet_backbone import get_hrnet_model

class CSRNet(nn.Module):
    def __init__(self, load_weights=False):
        super(CSRNet, self).__init__()
        self.seen = 0
        
        # HRNet 백본 로드 (가중치 사전 학습 여부는 get_hrnet_model에서 처리)
        # get_hrnet_model 함수는 HRNet을 초기화하고 pretrained 가중치를 로드한 뒤
        # 마지막에 고해상도 피처맵을 반환하도록 구성할 예정입니다.
        hrnet = get_hrnet_model(pretrained=not load_weights)

        # frontend: HRNet이 생성한 고해상도 특징 추출 부분
        self.frontend = hrnet

        # HRNet-W32 기준, 마지막 스테이지 출력 채널 수는 약 480ch 정도(4개 branch 32/64/128/256채널 결합 시)
        # 이 출력을 backend 입력으로 사용
        # backend 첫 Conv에서 stride=2를 줘서 1/4 -> 1/8 해상도로 다운샘플링
        self.backend = nn.Sequential(
            nn.Conv2d(480, 512, kernel_size=3, stride=2, padding=1),  # 해상도 1/2 축소 -> 1/8 달성
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
        x = self.frontend(x)  # HRNet 특징 추출
        x = self.backend(x)
        x = self.output_layer(x)
        # HRNet + backend 구성으로 이미 1/8 해상도 출력 -> 추가적인 interpolate 불필요
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, std=0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
