import argparse
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
from torchvision import datasets, transforms

# Python3용 transform (원본과 동일하게 유지)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def main():
    parser = argparse.ArgumentParser(description='PyTorch CSRNet Validation')
    # --model 인자를 통해 원하는 모델 파일을 지정할 수 있게 함
    parser.add_argument('--model', type=str, default='model_best.pth.tar',
                        help='path to the model file (xx.pth.tar)')
    args = parser.parse_args()
    
    root = '.'
    
    # 원본 코드와 동일한 폴더 구조
    part_A_train = os.path.join(root, 'part_A_final/train_data', 'images')
    part_A_test  = os.path.join(root, 'part_A_final/test_data', 'images')
    part_B_train = os.path.join(root, 'part_B_final/train_data', 'images')
    part_B_test  = os.path.join(root, 'part_B_final/test_data', 'images')
    
    # val.py 예시에서는 part_A_test만 사용 (원본 코드 동일 로직)
    path_sets = [part_A_test]

    img_paths = []
    for path in path_sets:
        for img_path in glob.glob(os.path.join(path, '*.jpg')):
            img_paths.append(img_path)

    model = CSRNet()
    model = model.cuda()

    # checkpoint 대신, 최종 모델 state_dict만 로드
    # args.model 파일에는 {'state_dict': ...} 형태가 저장되어 있다고 가정
    ckpt = torch.load(args.model)
    model.load_state_dict(ckpt['state_dict'])

    mae = 0.0
    
    # xrange를 range로 변경 (Python3)
    for i in range(len(img_paths)):
        # 원본 코드에서 사용하던 방식 그대로 유지
        img = 255.0 * F.to_tensor(Image.open(img_paths[i]).convert('RGB'))
        # BGR 통계량 보정 (원본 코드와 동일)
        img[0,:,:] = img[0,:,:] - 92.8207477031
        img[1,:,:] = img[1,:,:] - 95.2757037428
        img[2,:,:] = img[2,:,:] - 104.877445883
        
        img = img.cuda()
        
        gt_file = h5py.File(img_paths[i].replace('.jpg','.h5').replace('images','ground_truth'), 'r')
        groundtruth = np.asarray(gt_file['density'])
        
        output = model(img.unsqueeze(0))
        mae += abs(output.detach().cpu().sum().item() - np.sum(groundtruth))
        
        print(i, mae)
    
    print('Final MAE:', mae / len(img_paths))

if __name__ == '__main__':
    main()
