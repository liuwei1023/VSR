import os 
import torch 
import time 
import zipfile
import argparse
import json
import torchvision

import cv2 as cv 
import numpy as np 

import models.modules.EDVR_arch as EDVR_arch
from models.net.WDSR_A import WDSR_A

from config import system_config
from Logger import system_log

def ArgParser():
    
    parser = argparse.ArgumentParser(description="Test")
    parser.add_argument("MODE", help="train mode", type=str)
    parser.add_argument("cfg", help="config file", type=str)
    parser.add_argument("model_path", help="model path", type=str)
    parser.add_argument("output_folder", help="output folder", type=str)
    parser.add_argument('mode', type=str)
    parser.add_argument('split_num', type=int)
    parser.add_argument('num', type=int)
    return parser.parse_args()

def get_video_list(input_folder, args):
    
    assert args.mode in ['test', 'val']
    assert args.num <= args.split_num and args.num > 0
    
    if args.mode == 'test':
        full_list = list(filter(lambda x : '.' not in x, sorted(os.listdir(input_folder))))
        
    elif args.mode == 'val':
        with open('vsr_val.txt', 'r') as txtfile:
            full_list = list(line.split('\n')[0] for line in txtfile.readlines())
            
    div, mod = len(full_list) // args.split_num, len(full_list) % args.split_num
    
    sp = (div + 1) * (args.num - 1) if args.num <= mod else div * (args.num - 1) + mod
    K = div + 1 if mod >= args.num else div
    video_list = full_list[sp : sp + K].copy()
    
    print (video_list)
    print (len(video_list))
    
    return video_list
    
def zipDir(dirpath,outFullName):
    zip = zipfile.ZipFile(outFullName,"w",zipfile.ZIP_DEFLATED)
    for path,dirnames,filenames in os.walk(dirpath):
        fpath = path.replace(dirpath,'')

        for filename in filenames:
            zip.write(os.path.join(path,filename),os.path.join(fpath,filename))
    zip.close()

def image_forward(model, tensor_in):
    out = model(tensor_in)
    # h = out.size(2)
    # indices = list(range(4, h-4))
    # indices = torch.Tensor(indices).long().cuda()
    # out = torch.index_select(out, 2, indices)
    return out

if __name__ == "__main__":
    
    args = ArgParser()
    output_folder = args.output_folder 
    img_folder = os.path.join(output_folder, "img_output")
    final_folder = os.path.join(output_folder, "final_output")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"create dirs {output_folder}")
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)
        print(f"create dirs {img_folder}")
    if not os.path.exists(final_folder):
        os.makedirs(final_folder)
        print(f"create dirs {final_folder}")

    cfg_file = args.cfg
    with open(cfg_file, 'r') as f:
        system_config.update_config(json.load(f))
    log_path = os.path.join(output_folder, "log.log")
    system_log.set_filepath(log_path)

    input_folder = "/dfsdata2/share-group/aisz_group/kesci_ai_challenger/vsr/new_round1/test/540p"
    depth = system_config.depth
    batch_size = 2

    VSR_frame_MODE = "repeat"
    MODE = args.MODE
    model_path = args.model_path

    extension = system_config.extension
    video_list = get_video_list(input_folder, args)

    if MODE == "VSR":
        # VSR
        if system_config.net == "EDVR":
            net = EDVR_arch.EDVR(128, depth, 8, 5, 40, predeblur=False, HR_in=False)
        elif system_config.net == "EDVR_CBAM":
            net = EDVR_arch.EDVR_CBAM(128, depth, 8, 5, 40, predeblur=False, HR_in=False)
        elif system_config.net == "EDVR_Denoise":
            net = EDVR_arch.EDVR_Denoise(128, depth, 8, 5, 5, 5, 40, predeblur=False, HR_in=False)

    elif MODE == "SISR":
        # SISR
        if system_config.net == "WDSR_A":
            net = WDSR_A(4, system_config.n_resblocks, 64, 192).cuda()

    net.load_state_dict(torch.load(model_path))
    net = net.cuda()
    net = torch.nn.DataParallel(net)
    net.eval()

    with torch.no_grad():
        for video_name in video_list:
            folder_path = os.path.join(input_folder, video_name)
            frames_list = []
            for frame_path in os.listdir(folder_path):
                frames_list.append(frame_path)
            output_img_folder = os.path.join(img_folder, video_name)
            if not os.path.exists(output_img_folder):
                os.makedirs(output_img_folder)
            frames_list.sort(key=lambda x: int(x[-7: -4]))
            frame_index = 0
            while(frame_index < len(frames_list)):
                tensor_in_list = []
                for _ in range(batch_size):
                    if MODE == "VSR":
                        # VSR
                        tensor_VSR_list = []
                        for i in range(-(depth//2), (depth//2)+1):
                            cur_index = frame_index + i
                            if VSR_frame_MODE == "repeat":
                                if cur_index < 0:
                                    cur_index = 0
                                if cur_index >= len(frames_list):
                                    cur_index = len(frames_list)-1

                            img_path = os.path.join(folder_path, frames_list[cur_index]) 
                            cv_img = cv.imread(img_path).astype(np.float32).transpose(2,0,1)
                            img_tensor = torch.Tensor(cv_img/255).cuda()
                            # img_tensor = torch.nn.functional.pad(img_tensor, (0,0,1,1))
                            tensor_VSR_list.append(img_tensor)
                        tensor_VSR_list = torch.stack(tensor_VSR_list, 0)
                        tensor_in_list.append(tensor_VSR_list)           
                    elif MODE == "SISR":
                        # SISR
                        img_path = frames_list[frame_index]
                        img_path = os.path.join(folder_path, img_path)
                        cv_img = cv.imread(img_path).astype(np.float32).transpose(2,0,1)
                        img_tensor = torch.Tensor(cv_img/255).cuda()
                        # img_tensor = torch.nn.functional.pad(img_tensor, (0,0,1,1))
                        tensor_in_list.append(img_tensor)

                    frame_index += 1
                tensor_in = torch.stack(tensor_in_list, 0)
                out = image_forward(net, tensor_in)
                output_f = out.detach().cpu().numpy()

                # flip test
                # flip w
                out = image_forward(net, torch.flip(tensor_in, (-1, )))
                out = torch.flip(out, (-1, ))
                out = out.detach().cpu().numpy()
                output_f = output_f + out 

                # flip H
                out = image_forward(net, torch.flip(tensor_in, (-2, )))
                out = torch.flip(out, (-2, ))
                out = out.detach().cpu().numpy()
                output_f = output_f + out 

                # flip W and H
                out = image_forward(net, torch.flip(tensor_in, (-2, -1)))
                out = torch.flip(out, (-2, -1))
                out = out.detach().cpu().numpy()
                output_f = output_f + out 

                output_f = output_f / 4
                output_f = output_f * 255

                for i in range(batch_size):
                    cur_output_index = frame_index - batch_size + i
                    cv_img = output_f[i]
                    cv_img = cv_img.transpose(1,2,0)
                    img_name = frames_list[cur_output_index].split('.')[0] + '.png'
                    output_path = os.path.join(output_img_folder, img_name)
                    cv.imwrite(output_path, cv_img)
                    system_log.WriteLine(f"write image to {output_path}")

    # # bmp 2 mp4
    # for video_name in video_list:
    #     hr_name = f"{video_name}.mp4"
    #     hr_name = os.path.join(final_folder, hr_name)
    #     output_img_folder = os.path.join(img_folder, video_name)
    #     # shell_merge = f"ffmpeg -i {output_img_folder}/{video_name}_%03d.{extension}  -pix_fmt yuv420p  -vsync 0 {hr_name} -y"
    #     shell_merge = f"ffmpeg -i {output_img_folder}/{video_name}_%03d.{extension} -c:v libx265 -crf 1 {hr_name}"
    #     os.system(shell_merge)
    #     break

    # # zip
    # system_log.WriteLine(f"zip...")
    # zipDir(f"{final_folder}",f"{output_folder}/result.zip")


    system_log.WriteLine(f"all done")


model_zhankeng = torchvision.models.resnet50(pretrained=False).cuda()
x = torch.empty(5, 3, 224, 224).cuda()
print("zhankeng...")
while True:
    y = model_zhankeng(x)