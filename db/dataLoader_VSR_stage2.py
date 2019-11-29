import numpy as np 
import os 
import torch
import torch.utils.data as data
import cv2 as cv
import random

from Logger import system_log
rng = np.random.RandomState(100)

def RandomCrop(list_lr, hr_img, hr_crop_size):
    lr_crop_size = [int(size) for size in hr_crop_size]
    c,h,w = list_lr[0].shape
    
    lr_tl_x = int(rng.randint(0,w-lr_crop_size[0]-1))
    lr_tl_y = int(rng.randint(0,h-lr_crop_size[1]-1))
    hr_tl_x = lr_tl_x * 4
    hr_tl_y = lr_tl_y * 4

    newlist_lr = []
    for lr in list_lr:
        new_lr = lr[:, lr_tl_y:lr_tl_y+lr_crop_size[0], lr_tl_x:lr_tl_x+lr_crop_size[1]]
        newlist_lr.append(new_lr)
    
    new_hr = hr_img[:, hr_tl_y:hr_tl_y+hr_crop_size[0], hr_tl_x:hr_tl_x+hr_crop_size[1]]
    
    return newlist_lr, new_hr

def augment(list_lr, hr_img, hflip=True):
    # horizontal flip OR rotate
    hflip = hflip and random.random() < 0.5

    def _augment(img):
        if hflip:
            img = img[:, :, ::-1].copy()
        return img

    newlist_lr = []
    for lr in list_lr:
        new_lr = _augment(lr)
        newlist_lr.append(new_lr)

    new_hr = _augment(hr_img)

    return newlist_lr, new_hr

class dataset_loader_vsr_stage2(data.Dataset):
    def __init__(self, train_file, lr_root, hr_root, depth, crop_size=[256,256], flip=True, extension="bmp", is_train=True):
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

                name_list = list(filter(lambda x: f".{extension}" in x, [name for name in os.listdir(hr_video_folder)]))
                max_items = len(name_list)
                # if hr_video_name == "10091373":
                #     print(f"max_items = {max_items}")
                #     for name in os.listdir(hr_video_folder):
                #         print(f"name = {name}")
                # print(f"hr_video_name = {hr_video_name},  max_items = {max_items}")
                
                for i in range((depth//2)+1, max_items-(depth//2)+1):
                    lr_sub_seq = []
                    for j in range(-(depth//2),(depth//2)+1):
                        lr_index = i + j
                        lr_path = os.path.join(lr_video_folder, "%s_%03d.%s"%(lr_video_name,lr_index,extension))
                        lr_sub_seq.append(lr_path)
                    self.lr_seq.append(lr_sub_seq)
                    hr_path = os.path.join(hr_video_folder, "%s_%03d.%s"%(hr_video_name,i,extension))
                    self.hr_seq.append(hr_path)

        system_log.WriteLine(f"total frame {len(self.hr_seq)}")

    def __getitem__(self, index):
        lr_paths = self.lr_seq[index]
        hr_paths = self.hr_seq[index]

        lr_list = []
        for p in lr_paths:
            cv_img = cv.imread(p)
            if cv_img is None:
                print(f"ERROR: p = {p}")
            cv_img = cv_img.astype(np.float32).transpose(2,0,1)
            lr_list.append(cv_img)
        hr_img = cv.imread(hr_paths).astype(np.float32).transpose(2,0,1)

        if self.is_train:
            lr_list, hr_img = RandomCrop(lr_list, hr_img, self.crop_size)
            if self.flip:
                lr_list, hr_img = augment(lr_list, hr_img)
        lr_list_reverse = lr_list[::-1]

        return np.array(lr_list), np.array(lr_list_reverse), hr_img

    def __len__(self):
        return len(self.hr_seq)