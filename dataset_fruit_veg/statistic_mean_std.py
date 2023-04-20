# -*- coding = utf-8 -*-
"""
@Created: 2023/3/22 17:54
@Author: Red9th
@File: statistic_mean_std.py
@Software: PyCharm
"""
# 统计数据库中所有图片的每个通道的均值和标准差（用于归一化）
import os
import glob
import numpy as np
from PIL import Image

if __name__ == '__main__':
    train_files = glob.glob(os.path.join('train', '*', '*.jpg'))

    print(f'total {len(train_files)} files for training')

    result = []
    for file in train_files:
        img = Image.open(file).convert('RGB')
        img = np.array(img).astype(np.uint8)
        img = img / 255.
        result.append(img)

    print(np.shape(result)) # [BS, H, W, C]
    mean = np.mean(result, axis=(0, 1, 2))
    std = np.std(result, axis=(0, 1, 2))
    print(mean)
    print(std)