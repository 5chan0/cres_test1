import math
import os
import random
import numpy as np
import h5py
from PIL import Image, ImageFilter, ImageDraw, ImageOps
import cv2

def load_data(img_path, train=True):
    gt_path = img_path.replace('.jpg','.h5').replace('images','ground_truth')
    img = Image.open(img_path).convert('RGB')
    gt_file = h5py.File(gt_path)
    target = np.asarray(gt_file['density'])

    # (원본) 데이터 증강 코드가 if False로 묶여 실제로는 미적용
    if False:
        crop_size = (img.size[0]/2, img.size[1]/2)
        if random.randint(0,9) <= -1:
            dx = int(random.randint(0,1) * img.size[0] * 1./2)
            dy = int(random.randint(0,1) * img.size[1] * 1./2)
        else:
            dx = int(random.random() * img.size[0] * 1./2)
            dy = int(random.random() * img.size[1] * 1./2)

        img = img.crop((dx, dy, crop_size[0]+dx, crop_size[1]+dy))
        target = target[dy:int(crop_size[1]+dy), dx:int(crop_size[0]+dx)]

        if random.random() > 0.8:
            target = np.fliplr(target)
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # 1) 32의 배수로 만들기 위한 새 크기 계산 (패딩용)
    w, h = img.size
    new_w = int(math.ceil(w / 32.0) * 32)
    new_h = int(math.ceil(h / 32.0) * 32)

    # 2) 패딩할 픽셀 수 계산
    pad_left = (new_w - w) // 2
    pad_right = new_w - w - pad_left
    pad_top = (new_h - h) // 2
    pad_bottom = new_h - h - pad_top

    # 3) 필요하다면 이미지와 target에 패딩 적용
    if (new_w != w) or (new_h != h):
        # PIL 이미지 패딩 (검은색 or 원하는 색으로 채울 수 있음)
        img = ImageOps.expand(img, border=(pad_left, pad_top, pad_right, pad_bottom), fill=(0,0,0))
        
        # target 패딩 (numpy 배열)
        target = np.pad(target, 
                        ((pad_top, pad_bottom), (pad_left, pad_right)), 
                        mode='constant', 
                        constant_values=0)

    # 4) target을 (1/8) 축소 후 *64 (원본 로직 그대로)
    #    target.shape: (H, W) -> 리사이즈 시 (W/8, H/8)
    resized_h = int(target.shape[0] / 8)
    resized_w = int(target.shape[1] / 8)
    target = cv2.resize(target, (resized_w, resized_h), interpolation=cv2.INTER_CUBIC) * 64
    
    return img, target
