import argparse
import h5py
import PIL.Image as Image
import numpy as np
import os
import glob
from model import CSRNet
import torch
import torchvision.transforms.functional as F
from image import load_data  # load_data 함수가 image.py에 정의되어 있다고 가정

def main():
    # Argument parser 설정
    parser = argparse.ArgumentParser(description='Validate CSRNet model')
    parser.add_argument('--model', '-m', type=str, required=True,
                        help='Path to the model .pth.tar file')
    parser.add_argument('--data_dir', '-d', type=str, default='./part_A_final/test_data/images',
                        help='Path to the validation images directory')
    parser.add_argument('--gpu', '-g', type=str, default='0',
                        help='GPU id to use. Default is 0')
    args = parser.parse_args()

    # GPU 설정
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 모델 로드
    model = CSRNet()
    model = model.to(device)
    
    if not os.path.isfile(args.model):
        print(f"Model file '{args.model}' not found.")
        return

    checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # 검증 데이터 경로 수집
    img_paths = glob.glob(os.path.join(args.data_dir, '*.jpg'))

    if not img_paths:
        print(f"No images found in directory '{args.data_dir}'.")
        return

    mae = 0
    for i, img_path in enumerate(img_paths):
        # 이미지 및 Ground Truth 로드
        img, groundtruth = load_data(img_path, train=False)

        # 이미지 전처리
        img = F.to_tensor(img).unsqueeze(0).to(device)  # 배치 차원 추가
        img = img * 255.0  # 원본 코드와 동일하게 스케일링
        img[0, 0, :, :] -= 92.8207477031
        img[0, 1, :, :] -= 95.2757037428
        img[0, 2, :, :] -= 104.877445883
        # 원본 코드의 주석 처리된 transform 사용 대신 직접 전처리

        # Forward Pass
        with torch.no_grad():
            output = model(img)
            output_sum = output.detach().cpu().sum().item()
        
        # Ground Truth 카운트 계산
        gt_count = np.sum(groundtruth)
        
        # MAE 계산
        mae += abs(output_sum - gt_count)
        print(f'Image {i+1}/{len(img_paths)} - Current MAE: {mae/(i+1):.4f}')

    # 최종 MAE 출력
    final_mae = mae / len(img_paths)
    print(f'\nFinal MAE: {final_mae:.4f}')

if __name__ == '__main__':
    main()
