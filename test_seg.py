import os 
import torch 
import time 
import zipfile
import argparse
import json
import cv2 as cv 
import numpy as np 

import models.modules.EDVR_arch as EDVR_arch
from models.net.WDSR_A import WDSR_A

from config import system_config
from Logger import system_log

def zipDir(dirpath,outFullName):
    zip = zipfile.ZipFile(outFullName,"w",zipfile.ZIP_DEFLATED)
    for path,dirnames,filenames in os.walk(dirpath):
        fpath = path.replace(dirpath,'')

        for filename in filenames:
            zip.write(os.path.join(path,filename),os.path.join(fpath,filename))
    zip.close()

def image_forward(model, tensor_in):
    out = model(tensor_in)
    h = out.size(2)
    indices = list(range(4, h-4))
    indices = torch.Tensor(indices).long().cuda()
    out = torch.index_select(out, 2, indices)
    return out


parser = argparse.ArgumentParser(description="Test")
parser.add_argument("MODE", help="train mode", type=str)
parser.add_argument("cfg", help="config file", type=str)
parser.add_argument("output_folder", help="output folder", type=str)
args = parser.parse_args()

if __name__ == "__main__":
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

    input_folder = "/dfsdata2/share-group/aisz_group/tianchi/round2/test/images/" + system_config.extension + "_seg"
    full_frame_txt = "/dfsdata2/share-group/aisz_group/tianchi/round2/test/images/full_seg.txt"
    sub_frame_txt = "/dfsdata2/share-group/aisz_group/tianchi/round2/test/images/sub_seg.txt"
    depth = system_config.depth

    MODE = args.MODE 
    extension = system_config.extension   

    full_frame_folder_list = []
    sub_frame_folder_list = []
    with open(full_frame_txt, 'r') as f:
        lines = f.readlines()
        for line in lines:
            full_frame_folder_list.append(line.strip('\n'))   
    with open(sub_frame_txt, 'r') as f:
        lines = f.readlines()
        for line in lines:
            sub_frame_folder_list.append(line.strip('\n'))

    model_path = system_config.best_model_path
    

    if MODE == "VSR":
        # VSR
        if system_config.net == "EDVR":
            net = EDVR_arch.EDVR(128, depth, 8, 5, 40, predeblur=False, HR_in=False)
        elif system_config.net == "EDVR_CBAM":
            net = EDVR_arch.EDVR_CBAM(128, depth, 8, 5, 40, predeblur=False, HR_in=False)
        elif system_config.net == "EDVR_DUF":
            net = EDVR_arch.EDVR_DUF(128, depth, 8, 5, 40, predeblur=False, HR_in=False)
        elif system_config.net == "EDVR_DUF_V2":
            net = EDVR_arch.EDVR_DUF_v2(128, depth, 8, 5, 40, predeblur=False, HR_in=False)
        elif system_config.net == "EDVR_V2":
            net = EDVR_arch.EDVR_v2(128, depth, 8, 10, 40, predeblur=False, HR_in=False)
        elif system_config.net == "EDVR_FUSION":
            net = EDVR_arch.EDVR_Fusion(128, depth, 8, 5, 40, predeblur=False, HR_in=False)
        elif system_config.net == "EDVR_FUSION_CBAM":
            net = EDVR_arch.EDVR_Fusion_CBAM(128, depth, 8, 5, 40, predeblur=False, HR_in=False)
        elif system_config.net == "EDVR_FUSION_WD":
            net = EDVR_arch.EDVR_Fusion_WD(128, depth, 8, 5, 40, predeblur=False, HR_in=False)

    elif MODE == "SISR":
        # SISR
        if system_config.net == "WDSR_A":
            net = WDSR_A(4, system_config.n_resblocks, 64, 192).cuda()

    net.load_state_dict(torch.load(model_path))
    net = net.cuda()
    net.eval()

    with torch.no_grad():
        # full frame
        for folder_name in full_frame_folder_list:
            folder_path = os.path.join(input_folder, folder_name)    
            frame_list = []
            for frame_path in os.listdir(folder_path):
                frame_list.append(frame_path)   # Youku_00850_l_001.bmp
            frame_list.sort(key=lambda x:int(x[-7:-4]))
            total_frame = len(frame_list)
            frame_name = frame_list[0][:13]
            output_img_folder = os.path.join(img_folder, frame_name)
            if not os.path.exists(output_img_folder):
                os.makedirs(output_img_folder)
            
            for frame_idx in range(total_frame):
                if MODE == "VSR":
                    tensor_in_list = []
                    for i in range(-(depth//2), (depth//2)+1):
                        cur_idx = frame_idx + i 
                        if cur_idx < 0:
                            cur_idx = 0
                        elif cur_idx > total_frame - 1:
                            cur_idx = total_frame - 1
                        img_path = os.path.join(folder_path, frame_list[cur_idx])
                        cv_img = cv.imread(img_path).astype(np.float32).transpose(2,0,1)
                        img_tensor = torch.Tensor(cv_img/255).cuda()
                        img_tensor = torch.nn.functional.pad(img_tensor, (0,0,1,1))
                        tensor_in_list.append(img_tensor)
                    tensor_in = torch.stack(tensor_in_list, 0)
                elif MODE == "SISR":
                    img_path = os.path.join(folder_path, frame_list[frame_idx])
                    cv_img = cv.imread(img_path).astype(np.float32).transpose(2,0,1)
                    img_tensor = torch.Tensor(cv_img/255).cuda()
                    tensor_in = torch.nn.functional.pad(img_tensor, (0,0,1,1))

                tensor_in = torch.unsqueeze(tensor_in, 0)
                out = image_forward(net, tensor_in)
                output_f = out.detach().cpu().numpy()[0]

                # flip test
                # flip w
                out = image_forward(net, torch.flip(tensor_in, (-1, )))
                out = torch.flip(out, (-1, ))
                out = out.detach().cpu().numpy()[0]
                output_f = output_f + out 

                # flip H
                out = image_forward(net, torch.flip(tensor_in, (-2, )))
                out = torch.flip(out, (-2, ))
                out = out.detach().cpu().numpy()[0]
                output_f = output_f + out 

                # flip W and H
                out = image_forward(net, torch.flip(tensor_in, (-2, -1)))
                out = torch.flip(out, (-2, -1))
                out = out.detach().cpu().numpy()[0]
                output_f = output_f + out 

                output_f = output_f / 4
                output_f = output_f * 255
                output_f = output_f.transpose(1,2,0)

                image_index = int(frame_list[frame_idx][-7:-4])
                output_path = os.path.join(output_img_folder, "%s_%03d.%s"%(folder_name[:-2], image_index, extension))
                cv.imwrite(output_path, output_f)
                system_log.WriteLine(f"Write Image to {output_path}")

        # sub frame
        calc_index = [1, 26, 51, 76]
        for folder_name in sub_frame_folder_list:
            folder_path = os.path.join(input_folder, folder_name)
            frame_list = []
            for frame_path in os.listdir(folder_path):
                frame_list.append(frame_path) # Youku_00850_l_001.bmp
            

            frame_list.sort(key=lambda x:int(x[-7:-4]))
            first_index = int(frame_list[0][-7:-4])
            last_index = int(frame_list[-1][-7:-4])
            frame_name = frame_list[0][:13]
            output_img_folder = os.path.join(img_folder, frame_name)
            if not os.path.exists(output_img_folder):
                os.makedirs(output_img_folder)

            for c_idx in calc_index:
                img_path = os.path.join(folder_path, "%s_%03d.%s"%(frame_name, c_idx, extension))
                if os.path.exists(img_path):
                    # exist
                    if MODE == "VSR":
                        tensor_in_list = []
                        for i in range(-(depth//2), (depth//2)+1):
                            cur_idx = c_idx + i 
                            if cur_idx < first_index:
                                cur_idx = first_index
                            elif cur_idx > last_index:
                                cur_idx = last_index
                            img_path = os.path.join(folder_path, "%s_%03d.%s"%(frame_name, cur_idx, extension))
                            cv_img = cv.imread(img_path).astype(np.float32).transpose(2,0,1)
                            img_tensor = torch.Tensor(cv_img/255).cuda()
                            img_tensor = torch.nn.functional.pad(img_tensor, (0,0,1,1))
                            tensor_in_list.append(img_tensor)
                        tensor_in = torch.stack(tensor_in_list, 0)
                    elif MODE == "SISR":
                        cv_img = cv.imread(img_path).astype(np.float32).transpose(2,0,1)
                        img_tensor = torch.Tensor(cv_img/255).cuda()
                        tensor_in = torch.nn.functional.pad(img_tensor, (0,0,1,1))
                
                    tensor_in = torch.unsqueeze(tensor_in, 0)
                    out = image_forward(net, tensor_in)
                    output_f = out.detach().cpu().numpy()[0]

                    # flip test
                    # flip w
                    out = image_forward(net, torch.flip(tensor_in, (-1, )))
                    out = torch.flip(out, (-1, ))
                    out = out.detach().cpu().numpy()[0]
                    output_f = output_f + out 

                    # flip H
                    out = image_forward(net, torch.flip(tensor_in, (-2, )))
                    out = torch.flip(out, (-2, ))
                    out = out.detach().cpu().numpy()[0]
                    output_f = output_f + out 

                    # flip W and H
                    out = image_forward(net, torch.flip(tensor_in, (-2, -1)))
                    out = torch.flip(out, (-2, -1))
                    out = out.detach().cpu().numpy()[0]
                    output_f = output_f + out 

                    output_f = output_f / 4
                    output_f = output_f * 255
                    output_f = output_f.transpose(1,2,0)

                    output_path = os.path.join(output_img_folder, "%s_%03d.%s"%(frame_name, (c_idx/25)+1, extension))
                    cv.imwrite(output_path, output_f)
                    system_log.WriteLine(f"Write Image to {output_path}")

    # bmp 2 y4m
    for video_name in full_frame_folder_list:      
        hr_name = video_name[:12]   # Youku_00850_
        hr_name = f"{hr_name}h_Res.y4m" # Youku_00850_h_Res.y4m
        hr_name = os.path.join(final_folder, hr_name)
        if not os.path.exists(hr_name):
            lr_name = f"{video_name[:12]}l" # Youku_00850_l
            output_img_folder = os.path.join(img_folder, lr_name)
            shell_merge = f"ffmpeg -i {output_img_folder}/{lr_name}_%03d.{extension}  -pix_fmt yuv420p  -vsync 0 {hr_name} -y"
            os.system(shell_merge)

    for video_name in sub_frame_folder_list:
        hr_name = video_name[:12]   # Youku_00850_
        hr_name = f"{hr_name}h_Sub25_Res.y4m"   # Youku_00850_h_Sub25_Res.y4m
        hr_name = os.path.join(final_folder, hr_name)
        if not os.path.exists(hr_name):
            lr_name = f"{video_name[:12]}l"     # Youku_00850_l
            output_img_folder = os.path.join(img_folder, lr_name)
            shell_merge = f"ffmpeg -i {output_img_folder}/{lr_name}_%03d.{extension}  -pix_fmt yuv420p  -vsync 0 {hr_name} -y"
            os.system(shell_merge)

    # zip
    system_log.WriteLine(f"zip...")
    zipDir(f"{final_folder}",f"{output_folder}/result.zip")

    system_log.WriteLine(f"all done")

                            


                    




