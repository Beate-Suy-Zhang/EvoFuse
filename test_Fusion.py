import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


import os
print(os.getcwd())
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import sys
import time
import kornia
from tqdm import tqdm
import pygad
import numpy as np
import warnings
import logging
import copy

import cv2
# from path_config import get_path_config
from evo_path_config.fusion_Path import get_path_config
from futools.utils.testloader import get_test_loader

from futools.netdraevo import Encoder, Decoder, FusionModel, DRA1BR, prominent

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Initialize the device used
TQDM_BAR_FORMAT = '{l_bar}{bar:50}| {n_fmt}/{total_fmt} {elapsed}'
TQDM_BAR_FORMAT_2 = '{l_bar}{bar:30}| {n_fmt}/{total_fmt} {elapsed}'

ckpt_path = r"ckpt/evofuse.pth"


Fusion_Model = FusionModel().to(device)
Fusion_Model.load_state_dict(torch.load(ckpt_path)['Fusion_Model'])
# best temp for one single epoch of Evolutionary Learning
best_Model = copy.deepcopy(Fusion_Model).to(device)

# train data loader for fusion training 
trainloader = None

evodata = [
    
    "M3FD",
    # "TNO",
    # "RoadScene",    
    "FMB",
    ]
(project_root, train_folder_list, train_irfolder_list, train_vifolder_list, train_loader_list, val_folder_list, 
val_irfolder_list, val_vifolder_list, val_loader_list, project_root_list) = get_path_config(evodata)




# generate RGB images
def generate_color_images(Fusion_Model, test_loaders, out_paths, iter_save=""):
    print("Generating RGB images...")
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.CRITICAL)
    
    
    eval_copy_Model = copy.deepcopy(Fusion_Model).to(device)
    for m in Fusion_Model.modules():
        if isinstance(m, DRA1BR):
            m.fuse_convs()
            m.forward = m.forward_fuse  # update forward
        if isinstance(m, Decoder):
            m.fuse_convs()
            m.forward = m.forward_fuse  # update forward
    eval_copy_Model.eval()
    
    
    for i in range(len(out_paths)):
        if iter_save == "":
            test_out_folder = out_paths[i]
        else:
            test_out_folder=os.path.join(out_paths[i] + "evos", iter_save)
        # Model inference
        if not os.path.exists(test_out_folder):
            os.makedirs(test_out_folder)
        test_loader  =  get_test_loader(ir_root=test_loaders[i][0], 
                                        vis_root=test_loaders[i][1], 
                                        num_workers=test_loaders[i][2], 
                                        image_size=None,
                                        # image_size=test_loaders[i][3],   ###############################################################################################
                                        )
        with torch.no_grad():
            for (irimage, visimage_rgb, image_name) in tqdm(test_loader, bar_format=TQDM_BAR_FORMAT_2):
            # for (irimage, visimage_rgb, _, _, image_name) in tqdm(test_loader, bar_format=TQDM_BAR_FORMAT_2):
                data_IR = irimage.to(device)
                data_Ycbcr = visimage_rgb.to(device)
                data_VIS = data_Ycbcr[:, 0:1, :, :]

                data_Fuse = eval_copy_Model(data_VIS, data_IR)
                
                Y = data_Fuse * 255.
                Cb = data_Ycbcr[:, 1:2, :, :] * 255.
                Cr = data_Ycbcr[ :, 2:3, :, :] * 255.
                
                R = Y + 1.402 * (Cr - 128)
                G = Y - 0.344136 * (Cb - 128) - 0.714136 * (Cr - 128)
                B = Y + 1.772 * (Cb - 128)
                rgb_tensor = torch.cat((B, G, R), dim=1)
                rgb_tensor = torch.clamp(rgb_tensor, 0, 255)  # 确保值范围在 [0, 255]
                
                data_rgb_np = rgb_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
                output_path = os.path.join(test_out_folder, f"{os.path.split(image_name[0])[1][:-4]}.png")
                cv2.imwrite(output_path, data_rgb_np)
        del test_loader
    
    
if __name__ == '__main__':
    
    print("ckpt root: ", ckpt_path)
    # from thop import profile
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"    
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # fusion_model = FusionModel().to(device)
    # for m in fusion_model.modules():
    #     if isinstance(m, RepConv):
    #         m.fuse_convs()
    #         m.forward = m.forward_fuse  # update forward
    # test_info_VI = torch.randn(1, 1, 1024, 768).to(device)
    # test_info_IR = torch.randn(1, 1, 1024, 768).to(device)
    # flops, params = profile(fusion_model, inputs=(test_info_VI, test_info_IR))
    # print('FLOPs: ', flops)
    # print('Params: ', params)
    
    generate_color_images(Fusion_Model, val_loader_list, val_folder_list)
