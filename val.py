import argparse
import h5py
import PIL.Image as Image
import numpy as np
import os
import glob
from model import CSRNet
import torch
import torchvision.transforms.functional as F

def parse_args():
    parser = argparse.ArgumentParser(description='Validate CSRNet model')
    parser.add_argument('--model', '-m', type=str, required=True,
                        help='Path to the model .pth.tar file')
    parser.add_argument('--data_dir', '-d', type=str, default='./part_A_final/test_data/images',
                        help='Path to the validation images directory')
    parser.add_argument('--gpu', '-g', type=str, default='0',
                        help='GPU id to use. Default is 0')
    return parser.parse_args()

def load_model(model_path, device):
    model = CSRNet()
    model = model.to(device)
    
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found.")
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
            print(f"Loaded checkpoint from '{model_path}' (epoch {checkpoint.get('epoch', 'N/A')})")
        else:
            # Assume the entire file is the state_dict
            model.load_state_dict(checkpoint)
            print(f"Loaded state_dict from '{model_path}'")
    except Exception as e:
        raise RuntimeError(f"Error loading the model file: {e}")
    
    model.eval()
    return model

def get_image_paths(data_dir):
    img_paths = glob.glob(os.path.join(data_dir, '*.jpg'))
    if not img_paths:
        raise FileNotFoundError(f"No images found in directory '{data_dir}'.")
    return img_paths

def validate(model, device, img_paths):
    mae = 0.0
    for i, img_path in enumerate(img_paths):
        # Load image and ground truth
        img = 255.0 * F.to_tensor(Image.open(img_path).convert('RGB'))
        
        # Subtract mean values (as per original code)
        img[0, :, :] -= 92.8207477031
        img[1, :, :] -= 95.2757037428
        img[2, :, :] -= 104.877445883
        
        img = img.to(device)
        
        # Load ground truth density map
        gt_path = img_path.replace('.jpg', '.h5').replace('images', 'ground_truth')
        if not os.path.isfile(gt_path):
            print(f"Warning: Ground truth file '{gt_path}' not found. Skipping this image.")
            continue
        gt_file = h5py.File(gt_path, 'r')
        groundtruth = np.asarray(gt_file['density'])
        gt_file.close()
        
        # Forward pass
        with torch.no_grad():
            output = model(img.unsqueeze(0))
            output_sum = output.detach().cpu().sum().item()
        
        gt_count = np.sum(groundtruth)
        
        # Calculate MAE
        mae += abs(output_sum - gt_count)
        print(f'Image {i+1}/{len(img_paths)} - Current MAE: {mae/(i+1):.4f}')
    
    final_mae = mae / len(img_paths)
    print(f'\nFinal MAE: {final_mae:.4f}')

def main():
    args = parse_args()
    
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(f"Using GPU: {args.gpu}")
    else:
        print("GPU not available. Using CPU.")
    
    # Load model
    try:
        model = load_model(args.model, device)
    except Exception as e:
        print(e)
        return
    
    # Get image paths
    try:
        img_paths = get_image_paths(args.data_dir)
    except Exception as e:
        print(e)
        return
    
    # Validate
    validate(model, device, img_paths)

if __name__ == '__main__':
    main()
