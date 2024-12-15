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

    # 모델 초기화 및 로드
    model = CSRNet()
    model = model.to(device)
    
    if not os.path.isfile(args.model):
        print(f"Error: Model file '{args.model}' not found.")
        return

    try:
        checkpoint = torch.load(args.model, map_location=device)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
            print(f"Loaded checkpoint from '{args.model}' (epoch {checkpoint.get('epoch', 'N/A')})")
        else:
            # checkpoint가 dict가 아니거나 'state_dict' 키가 없는 경우, 전체를 state_dict로 간주
            model.load_state_dict(checkpoint)
            print(f"Loaded state_dict from '{args.model}'")
    except Exception as e:
        print(f"Error loading the model file: {e}")
        return

    model.eval()

    # 검증 데이터 경로 수집
    img_paths = glob.glob(os.path.join(args.data_dir, '*.jpg'))

    if not img_paths:
        print(f"No images found in directory '{args.data_dir}'.")
        return

    mae_total = 0
    mae_list = []
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
        mae = abs(output_sum - gt_count)
        mae_list.append(mae)
        mae_total += mae

        print(f'Image {i+1}/{len(img_paths)} - MAE: {mae:.4f}')

    # 최종 평균 MAE 계산 및 출력
    final_mae = mae_total / len(img_paths)
    print(f'\nFinal Average MAE: {final_mae:.4f}')

if __name__ == '__main__':
    main()