# -*- coding = utf-8 -*-
"""
@Created: 2023/3/22 11:39
@Author: Red9th
@File: split_dataset.py
@Software: PyCharm
"""
import os
import glob
import random
import shutil
from PIL import Image

if __name__ == '__main__':
    test_split_ratio = 0.05
    desired_size = 128
    raw_path = './raw'

    dirs = glob.glob(os.path.join(raw_path, '*'))
    dirs = [d for d in dirs if os.path.isdir(d)]

    # print(f'Total: {len(dirs)} classes: {dirs}')

    for path in dirs:
        path = path.split('\\')[-1]

        os.makedirs(f'train/{path}', exist_ok=True)
        os.makedirs(f'test/{path}', exist_ok=True)

        files = glob.glob(os.path.join(raw_path, path, '*.jpg'))
        files += glob.glob(os.path.join(raw_path, path, '*.png'))

        random.shuffle(files)

        boundary = int(len(files) * test_split_ratio)

        for i, file in enumerate(files):
            img = Image.open(file).convert('RGB')
            old_size = img.size
            ratio = float(desired_size) / max(old_size)
            new_size = (int(old_size[0] * ratio), int(old_size[1] * ratio))
            im = img.resize(new_size, Image.ANTIALIAS)
            new_im = Image.new('RGB', (desired_size, desired_size))
            new_im.paste(im, ((desired_size - new_size[0]) // 2, (desired_size - new_size[1]) // 2))

            assert new_im.mode == 'RGB'

            if i <= boundary:
                new_im.save(os.path.join(f'test/{path}', file.split('\\')[-1].split('.')[0] + '.jpg'))
            else:
                new_im.save(os.path.join(f'train/{path}', file.split('\\')[-1].split('.')[0] + '.jpg'))

    # test_files = glob.glob(os.path.join('test', '*', '*.jpg'))
    # train_files = glob.glob(os.path.join('train', '*', '*.jpg'))

    # print(f'total {len(test_files)} files for testing')
    # print(f'total {len(train_files)} files for training')