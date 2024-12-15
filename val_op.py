import h5py
import scipy.io as io
import PIL.Image as Image
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter 
import scipy
import json
import torchvision.transforms.functional as F
from matplotlib import cm as CM
from image import *
from model import CSRNet
import torch
import argparse
import sys

from torchvision import datasets, transforms

# `%matplotlib inline`은 주피터 노트북에서만 사용되므로 주석 처리합니다.
# %matplotlib inline

def main():
    # 1. argparse를 사용하여 명령줄 인자 처리
    parser = argparse.ArgumentParser(description='Validation for CSRNet')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to the model file (e.g., best_model.pth.tar)')
    args = parser.parse_args()
    
    # 2. 데이터 전처리 정의 (수정하지 않음)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    
    root = '.'
    
    # 3. 검증할 이미지 경로 설정 (수정하지 않음)
    part_A_test = os.path.join(root, 'part_A_final/test_data', 'images')
    part_B_test = os.path.join(root, 'part_B_final/test_data', 'images')
    path_sets = [part_A_test, part_B_test]
    
    img_paths = []
    for path in path_sets:
        for img_path in glob.glob(os.path.join(path, '*.jpg')):
            img_paths.append(img_path)
    
    # 4. 모델 초기화 및 GPU로 이동 (수정하지 않음)
    model = CSRNet()
    model = model.cuda()
    
    # 5. 체크포인트 로딩 코드 제거 및 명시적 모델 파일 로딩
    # 기존 코드:
    # checkpoint = torch.load('model_best.pth.tar')
    # model.load_state_dict(checkpoint['state_dict'])
    
    # 수정된 코드: --model 인자로 전달된 파일에서 state_dict 직접 로딩
    state_dict = torch.load(args.model)
    
    # 모델 파일이 state_dict만 포함하는 경우:
    model.load_state_dict(state_dict)
    
    # 만약 모델 파일이 checkpoint 형식으로 저장되었다면 아래 주석을 해제하고 사용하세요.
    # state_dict = torch.load(args.model)['state_dict']
    # model.load_state_dict(state_dict)
    
    model.eval()  # 모델을 평가 모드로 설정
    
    mae = 0
    for i in range(len(img_paths)):
        # 6. 이미지 로딩 및 전처리 (xrange -> range로 수정)
        img = 255.0 * F.to_tensor(Image.open(img_paths[i]).convert('RGB'))
        
        # 수동 정규화 (수정하지 않음)
        img[0, :, :] = img[0, :, :] - 92.8207477031
        img[1, :, :] = img[1, :, :] - 95.2757037428
        img[2, :, :] = img[2, :, :] - 104.877445883
        img = img.cuda()
        
        # 7. Ground Truth 밀도 맵 로딩 (수정하지 않음)
        gt_path = img_paths[i].replace('.jpg', '.h5').replace('images', 'ground_truth')
        gt_file = h5py.File(gt_path, 'r')
        groundtruth = np.asarray(gt_file['density'])
        gt_file.close()
        
        # 8. 추론 수행 (torch.no_grad() 
