import random
import os
from PIL import Image,ImageFilter,ImageDraw
import numpy as np
import h5py
from PIL import ImageStat
import cv2

def load_data(img_path,train = True):
    gt_path = img_path.replace('.jpg','.h5').replace('images','ground_truth')
    img = Image.open(img_path).convert('RGB')
    gt_file = h5py.File(gt_path)
    target = np.asarray(gt_file['density'])

    # 이미지 및 target의 크기를 32의 배수로 강제 조정
    w, h = img.size
    new_w = (w // 32) * 32
    new_h = (h // 32) * 32

    if new_w < w or new_h < h:
        # Center crop
        left = (w - new_w) // 2
        top = (h - new_h) // 2
        img = img.crop((left, top, left + new_w, top + new_h))
        target = target[top:top+new_h, left:left+new_w]

    # target을 1/8 축소 후 *64
    target = cv2.resize(target, (int(target.shape[1] / 8), int(target.shape[0] / 8)), interpolation=cv2.INTER_CUBIC) * 64
    return img,target
