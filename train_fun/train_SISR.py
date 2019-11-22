import os
import os.path as osp
import glob
import logging
import numpy as np
import cv2
import torch
import json
from torch import nn, optim
import time

import utils.util as util
from models.net.WDSR_A import WDSR_A

from utils.loss import CharbonnierLoss
from utils.math_ import AverageMeter, calc_psnr
from db.dataLoader_SISR import dataset_loader_sisr

from Logger import system_log
from config import system_config

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

def net_forward(net, tensor_in):
    input_frame_tensor = torch.nn.functional.pad(tensor_in, (0,0,1,1))
    out = net(input_frame_tensor)
    h = out.size(2)
    indices = list(range(4, h-4))
    indices = torch.Tensor(indices).long().cuda()
    out = torch.index_select(out, 2, indices)
    return out

def validation(net, db):
    net.eval()
    psnr = AverageMeter()
    
    with torch.no_grad():
        for lr_seq, hr_seq in db:
            tensor_lr = torch.Tensor(lr_seq/255).cuda()
            tensor_hr = torch.Tensor(hr_seq/255).cuda()

            out = net_forward(net, tensor_lr)
            psnr_iter = calc_psnr(out, tensor_hr)
            psnr.update(float(psnr_iter))
        
    return psnr.avg

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate"""
    lr = system_config.lr
    for count, epoch_idx in enumerate(system_config.multi_step):
        if epoch == epoch_idx:
            for _ in range(count+1):
                lr = lr * 0.1

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    system_log.WriteLine(f"update learning rate to {lr}")

def train_sisr(cfg):
    cfg_file = cfg
    with open(cfg_file, 'r') as f:
        system_config.update_config(json.load(f))
    system_log.set_filepath(system_config.log_path)

    lr_root = "/dfsdata2/share-group/aisz_group/tianchi/round2/LR/images/" + system_config.extension
    hr_root = "/dfsdata2/share-group/aisz_group/tianchi/round2/HR/images/" + system_config.extension

    if not system_config.MiniTest:
        train_file = "/dfsdata2/share-group/aisz_group/tianchi/round2/train/train"
        validation_file = "/dfsdata2/share-group/aisz_group/tianchi/round2/validation/val"
        if system_config.seg_frame:
            train_file = f"{train_file}_seg"
            validation_file = f"{validation_file}_seg"
            lr_root = f"{lr_root}_seg"
            hr_root = f"{hr_root}_seg"

        train_file = f"{train_file}.txt"
        validation_file = f"{validation_file}.txt"
            
    else:
        train_file = "/dfsdata2/liuwei79_data/ImageDatabase/tianchi/round2/miniTest.txt"
        validation_file = "/dfsdata2/liuwei79_data/ImageDatabase/tianchi/round2/miniTest.txt"

    train_loader = torch.utils.data.DataLoader(dataset_loader_sisr(train_file, lr_root, hr_root, crop_size=system_config.input_size, flip=system_config.flip, extension=system_config.extension, is_train=True), \
        batch_size=system_config.batch_size, shuffle = True, num_workers=10, pin_memory=True)
    validation_loader = torch.utils.data.DataLoader(dataset_loader_sisr(validation_file, lr_root, hr_root, extension=system_config.extension, is_train=False), \
        batch_size=system_config.validation_batch_size, shuffle = True, num_workers=10, pin_memory=True)
        
    if system_config.net == "WDSR_A":
        net = WDSR_A(4, system_config.n_resblocks, 64, 192).cuda()
    
    net = torch.nn.DataParallel(net)

    train_loss_iter = AverageMeter()
    train_loss_total = AverageMeter()
    loss_fun = CharbonnierLoss()
    optimizer = optim.Adam(net.parameters(), lr=system_config.lr)
    min_loss = np.inf 
    max_psnr = 0
    
    system_log.WriteLine(f"config: {system_config.config_all()}")

    for epoch_idx in range(1, system_config.max_epochs+1):
        train_loss_iter.reset()
        net.train()

        if epoch_idx in system_config.multi_step:
            adjust_learning_rate(optimizer, epoch_idx)

        start = time.time()
        for lr_seq, hr_seq in train_loader:
            iter_start = time.time()
            tensor_lr = torch.Tensor(lr_seq/255).cuda()
            tensor_hr = torch.Tensor(hr_seq/255).cuda()

            out = net(tensor_lr)
            loss = loss_fun(out, tensor_hr)

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
