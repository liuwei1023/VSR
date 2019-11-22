import numpy as np 
import os 
import torch
import torch.utils.data as data
import cv2 as cv
import random

from Logger import system_log
rng = np.random.RandomState(100)

def RandomCrop(lr_img, hr_img, hr_crop_size):
    lr_crop_size = [int(size/4) for size in hr_crop_size]
    c,h,w = lr_img.shape
    
    lr_tl_x = int(rng.randint(0,w-lr_crop_size[0]-1))
    lr_tl_y = int(rng.randint(0,h-lr_crop_size[1]-1))
    hr_tl_x = lr_tl_x * 4
    hr_tl_y = lr_tl_y * 4

    new_lr = lr_img[:, lr_tl_y:lr_tl_y+lr_crop_size[0], lr_tl_x:lr_tl_x+lr_crop_size[1]]
    new_hr = hr_img[:, hr_tl_y:hr_tl_y+hr_crop_size[0], hr_tl_x:hr_tl_x+hr_crop_size[1]]
    
    return new_lr, new_hr

def augment(lr_img, hr_img, hflip=True):
    # horizontal flip OR rotate
    hflip = hflip and random.random() < 0.5

    def _augment(img):
        if hflip:
            img = img[:, :, ::-1].copy()
        return img

    new_lr = _augment(lr_img)
    new_hr = _augment(hr_img)

    return new_lr, new_hr

class dataset_loader_sisr(data.Dataset):
    def __init__(self, train_file, lr_root, hr_root, crop_size=[256,256], flip=True, extension="bmp", is_train=True):
        system_log.WriteLine("reading dataset")
        self.is_train = is_train
        self.flip = flip
        self.crop_size = crop_size
        self.lr_seq = []
        self.hr_seq = []
        with open(train_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                element = line.strip('\n').split(' ')
                lr_video_name = element[0]
                hr_video_name = element[1]
                lr_video_folder = os.path.join(lr_root, lr_video_name)
                hr_video_folder = os.path.join(hr_root, hr_video_name)

                max_items = len([name for name in os.listdir(hr_video_folder)])
                for i in range(1, max_items+1):
                    lr_path = os.path.join(lr_video_folder, "%s_%03d.%s"%(lr_video_name, i, extension))
                    hr_path = os.path.join(hr_video_folder, "%s_%03d.%s"%(hr_video_name, i, extension))
                    self.lr_seq.append(lr_path)
                    self.hr_seq.append(hr_path)

        system_log.WriteLine(f"total frame {len(self.hr_seq)}")

    def __getitem__(self, index):
        lr_paths = self.lr_seq[index]
        hr_paths = self.hr_seq[index]

        lr_img = cv.imread(lr_paths).astype(np.float32).transpose(2,0,1)
        hr_img = cv.imread(hr_paths).astype(np.float32).transpose(2,0,1)

        if self.is_train:
            lr_img, hr_img = RandomCrop(lr_img, hr_img, self.crop_size)
            if self.flip:
                lr_img, hr_img = augment(lr_img, hr_img)

        return lr_img, hr_img

    def __len__(self):
        return len(self.hr_seq)