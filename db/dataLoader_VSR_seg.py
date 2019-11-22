import numpy as np 
import os 
import torch
import torch.utils.data as data
import cv2 as cv
import random

from Logger import system_log
rng = np.random.RandomState(100)

def RandomCrop(list_lr, hr_img, hr_crop_size):
    lr_crop_size = [int(size/4) for size in hr_crop_size]
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

class dataset_loader_vsr(data.Dataset):
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

                images_name = [name for name in os.listdir(hr_video_folder)]    # Youku_00250_h_GT_001.bmp   Youku_00250_l_001.bmp
                images_name.sort(key = lambda x: int(x[-7:-4]))
                first_image_index = int(images_name[0][-7:-4])
                end_image_index = int(images_name[-1][-7:-4])
                _lr_video_name = images_name[0][:-12]    # Youku_00250_
                _lr_video_name = f"{_lr_video_name}l"
                _hr_video_name = _lr_video_name[:-1]    # Youku_00250_
                _hr_video_name = f"{_hr_video_name}h_GT"    # Youku_00250_h_GT

                for i in range(first_image_index, end_image_index+1):
                    lr_sub_seq = []
                    first_lr_path = os.path.join(lr_video_folder, "%s_%03d.%s"%(_lr_video_name,first_image_index,extension))
                    end_lr_path = os.path.join(lr_video_folder, "%s_%03d.%s"%(_lr_video_name,end_image_index,extension))
                    for j in range(-(depth//2),(depth//2)+1):
                        lr_index = i + j
                        if lr_index > first_image_index-1 and lr_index < end_image_index+1:
                            lr_path = os.path.join(lr_video_folder, "%s_%03d.%s"%(_lr_video_name,lr_index,extension))
                            lr_sub_seq.append(lr_path)
                        elif lr_index < first_image_index:
                            lr_sub_seq.append(first_lr_path)
                        else:
                            lr_sub_seq.append(end_lr_path)
                    self.lr_seq.append(lr_sub_seq)
                    hr_path = os.path.join(hr_video_folder, "%s_%03d.%s"%(_hr_video_name,i,extension))
                    self.hr_seq.append(hr_path)

        system_log.WriteLine(f"total frame {len(self.hr_seq)}")

    def __getitem__(self, index):
        lr_paths = self.lr_seq[index]
        hr_paths = self.hr_seq[index]

        lr_list = []
        for p in lr_paths:
            cv_img = cv.imread(p).astype(np.float32).transpose(2,0,1)
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