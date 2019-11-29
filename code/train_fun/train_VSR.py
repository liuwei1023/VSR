import os
import os.path as osp
import glob
import logging
import numpy as np
import cv2
import torch
import math 
import json
from torch import nn, optim
import time

import utils.util as util
import models.modules.EDVR_arch as EDVR_arch

from utils.loss import CharbonnierLoss
from utils.math_ import AverageMeter, calc_psnr
from db.dataLoader_VSR import dataset_loader_vsr
from db.dataLoader_VSR_stage2 import dataset_loader_vsr_stage2
# from db.dataLoader_VSR_seg import dataset_loader_vsr
from Logger import system_log
from config import system_config

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

def net_forward(net, tensor_in):
    # input_frame_tensor = torch.nn.functional.pad(tensor_in, (0,0,1,1))
    # out = net(input_frame_tensor)
    # h = out.size(2)
    # indices = list(range(4, h-4))
    # indices = torch.Tensor(indices).long().cuda()
    # out = torch.index_select(out, 2, indices)

    out = net(tensor_in)
    return out

def validation(net, db):
    net.eval()
    psnr = AverageMeter()
    
    with torch.no_grad():
        for lr_seq, lr_seq_reverse, hr_seq in db:
            tensor_lr = torch.Tensor(lr_seq/255).cuda()
            tensor_hr = torch.Tensor(hr_seq/255).cuda()

            out = net_forward(net, tensor_lr)
            psnr_iter = calc_psnr(out, tensor_hr)
            psnr.update(float(psnr_iter))
        
    return psnr.avg

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate"""
    learning_mode = system_config.learning_mode
    lr = system_config.lr
    if learning_mode == "cosin":
        c = (np.cos((epoch%100)*0.02*math.pi)+1) / 2
        lr = (lr * c) + 0.000001
    elif learning_mode == "normal":
        multi_step = system_config.multi_step
        for i in range(len(multi_step)):
            if i == 0:
                if epoch <= multi_step[0]:
                    lr = lr
            else:
                if epoch > multi_step[i-1] and epoch <= multi_step[i]:
                    lr = lr * 0.5**i
            
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    system_log.WriteLine(f"update learning rate to {lr}")

def train_vsr(cfg, model_path=None):
    cfg_file = cfg
    with open(cfg_file, 'r') as f:
        system_config.update_config(json.load(f))
    system_log.set_filepath(system_config.log_path)

    lr_root = "dataset/train/540p/"
    hr_root = "dataset/train/4K/"

    if system_config.Stage2:
        if system_config.MiniTest:
            train_file = "dataset/train/miniTest.txt"
            validation_file = "dataset/train/miniTest.txt"
        else:
            train_file = "dataset/train/train.txt"
            validation_file = "dataset/train/validation.txt"

        lr_root = "dataset/train/540p/"
        hr_root = "dataset/train/4K/"
    
        
    else:
        if system_config.MiniTest:
            train_file = "dataset/train/miniTest.txt"
            validation_file = "dataset/train/miniTest.txt"
        else:
            train_file = "dataset/train/train"
            validation_file = "dataset/train/validation"
            if system_config.seg_frame:
                train_file = f"{train_file}_seg"
                validation_file = f"{validation_file}_seg"
                lr_root = f"{lr_root}_seg"
                hr_root = f"{hr_root}_seg"

            train_file = f"{train_file}.txt"
            validation_file = f"{validation_file}.txt"

        

    if system_config.Stage2:
        train_loader = torch.utils.data.DataLoader(dataset_loader_vsr_stage2(train_file, lr_root, hr_root, system_config.depth, crop_size=system_config.input_size, flip=system_config.flip, extension=system_config.extension, is_train=True), \
            batch_size=system_config.batch_size, shuffle = True, num_workers=10, pin_memory=True)
        validation_loader = torch.utils.data.DataLoader(dataset_loader_vsr_stage2(validation_file, lr_root, hr_root, system_config.depth, extension=system_config.extension, is_train=False), \
            batch_size=system_config.validation_batch_size, shuffle = True, num_workers=10, pin_memory=True)
    else:
        train_loader = torch.utils.data.DataLoader(dataset_loader_vsr(train_file, lr_root, hr_root, system_config.depth, crop_size=system_config.input_size, flip=system_config.flip, extension=system_config.extension, is_train=True), \
            batch_size=system_config.batch_size, shuffle = True, num_workers=10, pin_memory=True)
        validation_loader = torch.utils.data.DataLoader(dataset_loader_vsr(validation_file, lr_root, hr_root, system_config.depth, extension=system_config.extension, is_train=False), \
            batch_size=system_config.validation_batch_size, shuffle = True, num_workers=10, pin_memory=True)

    if system_config.net == "EDVR":
        net = EDVR_arch.EDVR(128, system_config.depth, 8, 5, 40, predeblur=False, HR_in=False)
    elif system_config.net == "EDVR_CBAM":
        net = EDVR_arch.EDVR_CBAM(128, system_config.depth, 8, 5, 40, predeblur=False, HR_in=False)
    elif system_config.net == "EDVR_CBAM_Stage2":
        net = EDVR_arch.EDVR_CBAM(128, system_config.depth, 8, 5, 40, predeblur=False, HR_in=True)
    elif system_config.net == "EDVR_DUF":
        net = EDVR_arch.EDVR_DUF(128, system_config.depth, 8, 5, 40, predeblur=False, HR_in=False)
    elif system_config.net == "EDVR_DUF_V2":
        net = EDVR_arch.EDVR_DUF_v2(128, system_config.depth, 8, 5, 40, predeblur=False, HR_in=False)
    elif system_config.net == "EDVR_V2":
        net = EDVR_arch.EDVR_v2(128, system_config.depth, 8, 10, 40, predeblur=False, HR_in=False)
    elif system_config.net == "EDVR_FUSION":
        net = EDVR_arch.EDVR_Fusion(128, system_config.depth, 8, 5, 40, predeblur=False, HR_in=False)
    elif system_config.net == "EDVR_FUSION_CBAM":
        net = EDVR_arch.EDVR_Fusion_CBAM(128, system_config.depth, 8, 5, 40, predeblur=False, HR_in=False)
    elif system_config.net == "EDVR_FUSION_WD":
        net = EDVR_arch.EDVR_Fusion_WD(128, system_config.depth, 8, 5, 40, predeblur=False, HR_in=False)
    elif system_config.net == "EDVR_Denoise":
        net = EDVR_arch.EDVR_Denoise(128, system_config.depth, 8, 5, 5, 5, 40, predeblur=False, HR_in=False)
    elif system_config.net == "EDVR_CBAM_Nonlocal":
        net = EDVR_arch.EDVR_CBAM_Nonlocal(128, system_config.depth, 8, 3, system_config.non_local[0], 2, 25, system_config.non_local[1], 10, system_config.non_local[2], 5, predeblur=False, HR_in=False)
    elif system_config.net == "EDVR_CBAM_Denoise_Nonlocal":
        net = EDVR_arch.EDVR_Denoise_Nonlocal(128, system_config.depth, 8, 5, 5, 3, system_config.non_local[0], 2, 25, system_config.non_local[1], 10, system_config.non_local[2], 5, predeblur=False, HR_in=False)
    elif system_config.net == "EDVR_CBAM_Denoise":
        net = EDVR_arch.EDVR_Denoise(128, system_config.depth, 8, 5, 5, 5, 40, predeblur=False, HR_in=False)
     
    if not model_path == None:
        net.load_state_dict(torch.load(model_path))
        system_log.WriteLine(f"loading model from {model_path}")

    net = net.cuda()
    net = torch.nn.DataParallel(net)

    train_loss_iter = AverageMeter()
    train_loss_total = AverageMeter()
    loss_fun = CharbonnierLoss()
    mse_fun = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=system_config.lr)
    min_loss = np.inf 
    max_psnr = 0
    
    system_log.WriteLine(f"config: {system_config.config_all()}")

    for epoch_idx in range(1, system_config.max_epochs+1):
        train_loss_iter.reset()
        net.train()

        adjust_learning_rate(optimizer, epoch_idx)
        # if epoch_idx in system_config.multi_step:
        #     adjust_learning_rate(optimizer, epoch_idx)

        start = time.time()
        for lr_seq, lr_seq_reverse, hr_seq in train_loader:
            iter_start = time.time()
            tensor_lr = torch.Tensor(lr_seq/255).cuda()
            tensor_hr = torch.Tensor(hr_seq/255).cuda()
            
            out = net(tensor_lr)
            loss = loss_fun(out, tensor_hr)

            if system_config.PP_loss:
                tensor_lr_reverse = torch.Tensor(lr_seq_reverse/255).cuda()
                out_reverse = net(tensor_lr_reverse)
                pp_loss = mse_fun(out, out_reverse)
                loss += pp_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_iter.update(float(loss))
            iter_end = time.time()

            system_log.WriteLine(f"Epoch[{epoch_idx}/{system_config.max_epochs}]:train_loss_epoch:{train_loss_iter.avg:.8f}, loss_iter: {loss:.8f}, cost time: {(iter_end-iter_start):.8f}sec!")

            
        end = time.time()
        train_loss_total.update(train_loss_iter.avg)
        system_log.WriteLine(f"Epoch[{epoch_idx}/{system_config.max_epochs}]: train_loss_total:{train_loss_total.avg:.8f}, train_loss_iter:{train_loss_iter.avg:.8f}, cost time: {(end-start):.8f}sec!")

        # min loss
        if min_loss > train_loss_iter.avg:
            saved_model = net.module
            torch.save(saved_model.state_dict(), system_config.min_loss_model_path)
            system_log.WriteLine(f"min loss update from {min_loss} to {train_loss_iter.avg}, save model to {system_config.min_loss_model_path}")
            min_loss = train_loss_iter.avg

        # save ckpt
        saved_model = net.module
        torch.save(saved_model.state_dict(), system_config.ckpt_path.format(epoch_idx))

        # validation
        if epoch_idx % system_config.validation_per_epochs == 0:
            val_start = time.time()
            psnr = validation(net, validation_loader)
            val_end = time.time()

            system_log.WriteLine(f"Validation: psnr:{psnr:.8f}, cost time: {(val_end-val_start):8f}sec!")
            if max_psnr < psnr:
                # save model
                saved_model = net.module
                torch.save(saved_model.state_dict(), system_config.best_model_path)
                system_log.WriteLine(f"psnr update from {max_psnr} to {psnr}, save model to {system_config.best_model_path}")
                max_psnr = psnr

    system_log.WriteLine(f"train done!")
    system_log.WriteLine(f"min_loss: {min_loss}, max_psnr: {max_psnr}")
