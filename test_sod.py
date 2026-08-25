#!/usr/bin/python
# -*- encoding: utf-8 -*-
#-****************************************************************-#
# Author: WangDi
# Email:  diwang1211@mail.dlut.edu.cn or wangdi_1211@njust.edu.cn
#-****************************************************************-#

import sys
sys.path.append("..")
import numpy as np
from decimal import Decimal
import os
import cv2
import torch
import torch.backends.cudnn
import torch.nn.functional as F
import torch.utils.data

from sodutils.dataloader.fus_sod_dataloader import test_dataset, get_sod_train_loader
from sodutils.models.fgccnet import FGCCNet
from tqdm import tqdm

import argparse
import warnings
warnings.filterwarnings('ignore')
import time
import statistics


def normalize(pred):
    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    return pred


def f_measure(pred, gt, beta=0.3):
    gt = gt.astype(np.bool_)
    pred = normalize(pred)
    thresholds = np.linspace(0, 1, 256)
    f_scores = []
    for thresh in thresholds:
        bin_pred = pred >= thresh
        tp = (bin_pred & gt).sum()
        fp = (bin_pred & ~gt).sum()
        fn = (~bin_pred & gt).sum()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f = (1 + beta) * precision * recall / (beta * precision + recall + 1e-8)
        f_scores.append(f)
    return max(f_scores)


def s_measure(pred, gt):
    # simplified version of structure similarity
    pred = normalize(pred)
    gt = gt.astype(np.float32)
    mean_pred = pred.mean()
    mean_gt = gt.mean()
    alpha = 0.5
    # object-aware similarity
    O = 1 - np.abs(mean_pred - mean_gt)
    # region-aware similarity (simplified, not full structural measure)
    x = np.abs(pred - gt).mean()
    R = 1 - x
    return alpha * O + (1 - alpha) * R


def e_measure(pred, gt):
    pred = normalize(pred)
    gt = gt.astype(np.float32)
    fm = pred - pred.mean()
    gm = gt - gt.mean()
    score = 1 - np.mean(np.sqrt((fm - gm) ** 2))
    return score


Method='images'
# print("Method:", Method)
# dataset='VT5000'

def test_sod(dataset='VT5000', ckpt='Comparison/'+Method+'/VT5000/iter0/model_sod.pth', num=0):
# def test_sod(dataset='VT1000', ckpt='./checkpoint/SOD/bievofuse/model_sod.pth', num=0):
    TQDM_BAR_FORMAT_2 = '{l_bar}{bar:30}| {n_fmt}/{total_fmt} {elapsed}'
    # todo: set the device for training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    # todo: define network
    sodnet = FGCCNet().to(device)

    # todo: load model parameters
    # load_path = os.path.join(opt.load, str(num), 'model_sod.pth')
    load_path = os.path.join(ckpt)
    sodnet.load_state_dict(torch.load(load_path))
    print('===> Loading pretrained model from {} sucessfully~')
    sodnet.eval()

    # todo: testing datasets
    # test_datasets = ['VT821','VT1000', 'VT5000']
    # test_datasets = ['VT5000']
    # path = './results/'
    # save_path = os.path.join(path, dataset)
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    
    # save_path = 'ViSOD/' + dataset + '/' + Method + '/'
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)

    test_path = "./datasets/"
    rgb_root     = test_path + dataset + '/val/RGB/'
    thermal_root = test_path + dataset + '/val/T/'
    fus_root     = test_path + dataset + '/val/images/'  
    # fus_root     = test_path + dataset + '/val/'+Method+'/'
    gt_root      = test_path + dataset + '/val/GT/'

    val_loader = test_dataset(rgb_root, thermal_root, fus_root, gt_root, 288)
    
    # val_loader = get_sod_train_loader(rgb_root, thermal_root, fus_root, gt_root,
    #                           batchsize=1,
    #                           trainsize=288,
    #                           shuffle=False,
    #                           split='val'
    #                           )
    
    with torch.no_grad():
        mae_sum = 0
        # simplified version of structure similarity
        S_sum = 0
        F_sum = 0
        E_sum = 0
        for i in tqdm(range(val_loader.size), bar_format=TQDM_BAR_FORMAT_2):
            rgb, thermal, fus, gt, name = val_loader.load_data()
        # for i, (rgb, thermal, fus, gt, name) in tqdm(enumerate(val_loader), bar_format=TQDM_BAR_FORMAT_2):
            # rgb = rgb.cuda()
            # thermal = thermal.cuda()
            # fus = fus.cuda()
            # gt = gt.cuda()
            
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)

            rgb = rgb.cuda()
            thermal = thermal.cuda()
            fus = fus.cuda()

            sal_input = torch.cat((rgb, thermal, fus), dim=0)
            s_coarse, rgb_map, tma_map, y, s_output = sodnet(sal_input)
            res = s_output
            res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            
            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])
            S_sum = S_sum + s_measure(res, gt)
            F_sum = F_sum + f_measure(res, gt)
            E_sum = E_sum + e_measure(res, gt)
            
            # cv2.imwrite(save_path + name, res * 255)

        mae = mae_sum / val_loader.size
        ssim = S_sum / val_loader.size
        fscore = F_sum / val_loader.size
        eam = E_sum / val_loader.size
        
        # mae = mae_sum / len(val_loader)
        # ssim = S_sum / len(val_loader)
        # fscore = F_sum / len(val_loader)
        # eam = E_sum / len(val_loader)
    
    print("MAE: {:.6f}, S: {:.6f}, F: {:.6f}, E: {:.6f}".format(mae, ssim, fscore, eam))
    fit_score = 1. / (1e-8 + mae) + ssim * 10 + fscore * 10 + eam * 10
    return fit_score
    return {"MAE": mae, "S": ssim, "F": fscore, "E": eam}, fit_score
    # return fit_score * 1.5

if __name__ == "__main__":
    test_sod(num=0)
