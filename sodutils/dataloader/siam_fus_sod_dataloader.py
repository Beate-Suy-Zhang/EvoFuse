import os
import cv2
import numpy as np
from PIL import Image
import random

import torch.utils.data
import torchvision.transforms as transforms


def cv_random_flip(rgb, ycbcr, thermal, gt):
    flip_flag = random.randint(0, 1)
    # flip_flag2= random.randint(0,1)
    #left right flip
    if flip_flag == 1:
        rgb     = rgb.transpose(Image.FLIP_LEFT_RIGHT)
        thermal = thermal.transpose(Image.FLIP_LEFT_RIGHT)
        gt      = gt.transpose(Image.FLIP_LEFT_RIGHT)
        ycbcr   = ycbcr.transpose(Image.FLIP_LEFT_RIGHT)

    #top bottom flip
    # if flip_flag2==1:
    #     img = img.transpose(Image.FLIP_TOP_BOTTOM)
    #     label = label.transpose(Image.FLIP_TOP_BOTTOM)
    #     depth = depth.transpose(Image.FLIP_TOP_BOTTOM)
    return rgb, ycbcr, thermal, gt

def cv_random_flip_0(gt, edge):
    flip_flag = random.randint(0, 1)
    # flip_flag2= random.randint(0,1)
    #left right flip
    if flip_flag == 1:
        gt      = gt.transpose(Image.FLIP_LEFT_RIGHT)
        edge    = gt.transpose(Image.FLIP_LEFT_RIGHT)

    #top bottom flip
    # if flip_flag2==1:
    #     img = img.transpose(Image.FLIP_TOP_BOTTOM)
    #     label = label.transpose(Image.FLIP_TOP_BOTTOM)
    #     depth = depth.transpose(Image.FLIP_TOP_BOTTOM)
    return gt, edge

def get_sod_loader(rgb_root, thermal_root, gt_root, batchsize, trainsize, shuffle=True, num_workers=8, pin_memory=False, split='train'):
    if split == 'train':
        dataset = SalObjDataset(rgb_root, thermal_root, gt_root, trainsize)
    else:
        dataset = SalObjDataset_val(rgb_root, thermal_root, gt_root, trainsize)
        
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)

    return data_loader

class SalObjDataset(torch.utils.data.Dataset):
    def __init__(self, rgb_root, thermal_root, gt_root, trainsize):
        self.trainsize = trainsize
        self.rgb      = [rgb_root + '/' + f       for f in os.listdir(rgb_root)   if f.endswith('.jpg') or f.endswith('.png')]
        self.thermal  = [thermal_root + '/' + f   for f in os.listdir(thermal_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gt       = [gt_root+ '/' + f         for f in os.listdir(gt_root)      if f.endswith('.jpg') or f.endswith('.png')]
        self.name     = [f                        for f in os.listdir(gt_root)      if f.endswith('.jpg') or f.endswith('.png')]

        self.rgb      = sorted(self.rgb)
        self.thermal  = sorted(self.thermal)
        self.gt       = sorted(self.gt)
        self.name     = sorted(self.name)

        self.size = len(self.rgb)

        self.rgb_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.thermal_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])

        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])

        self.index = 0


    def __getitem__(self, index):
        rgb     = self.rgb_loader(self.rgb[index])
        ycbcr   = self.ycbcr_loader(self.rgb[index])
        thermal = self.rgb_loader(self.thermal[index])
        gt      = self.binary_loader(self.gt[index])
        name    = self.name[index]

        rgb, ycbcr, thermal, gt = cv_random_flip(rgb, ycbcr, thermal, gt)

        rgb     = self.rgb_transform(rgb)
        ycbcr     = self.rgb_transform(ycbcr)
        thermal = self.thermal_transform(thermal)
        gt      = self.gt_transform(gt)

        return rgb, ycbcr, thermal, gt, name


    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def ycbcr_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB').convert('YCbCr')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def resize(self, img, gt, thermal):
        assert img.size == gt.size and gt.size==thermal.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST),thermal.resize((w, h), Image.NEAREST)
        else:
            return img, gt, thermal

    def __len__(self):
        return self.size


class SalObjDataset_val(torch.utils.data.Dataset):
    def __init__(self, rgb_root, thermal_root, gt_root, trainsize):
        self.trainsize = trainsize
        self.rgb     = [rgb_root     + f for f in os.listdir(rgb_root)       if f.endswith('.jpg') or f.endswith('.png')]
        self.thermal = [thermal_root + f for f in os.listdir(thermal_root)   if f.endswith('.jpg') or f.endswith('.png')]
        self.gt      = [gt_root            + f for f in os.listdir(gt_root)  if f.endswith('.jpg') or f.endswith('.png')]
        self.name    = [f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]

        self.rgb      = sorted(self.rgb)
        self.gt       = sorted(self.gt)
        self.thermal  = sorted(self.thermal)
        self.name     = sorted(self.name)

        self.size = len(self.rgb)

        self.rgb_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        self.thermal_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])

        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])

        self.index = 0

    def __getitem__(self, index):
        rgb     = self.rgb_loader(self.rgb[index])
        thermal = self.rgb_loader(self.thermal[index])
        gt      = self.binary_loader(self.gt[index])
        name    = self.name[index]

        rgb     = self.rgb_transform(rgb)
        thermal = self.thermal_transform(thermal)
        gt      = self.gt_transform(gt)

        return rgb, thermal, gt, name

    def load_data(self):
        rgb = self.rgb_loader(self.rgb[self.index])
        rgb = self.rgb_transform(rgb).unsqueeze(0)

        thermal = self.rgb_loader(self.thermal[self.index])
        thermal = self.thermal_transform(thermal).unsqueeze(0)

        gt = self.binary_loader(self.gt[self.index])

        name = self.rgb[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        self.index = self.index % self.size

        return rgb, thermal, gt, name


    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')


    def __len__(self):
        return self.size

#test dataset and loader
class test_original_dataset:
    def __init__(self, rgb_root, thermal_root, gt_root, testsize):
        self.testsize = testsize
        self.rgb     = [rgb_root     + f for f in os.listdir(rgb_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.thermal = [thermal_root + f for f in os.listdir(thermal_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gt      = [gt_root      + f for f in os.listdir(gt_root)  if f.endswith('.jpg') or f.endswith('.png')]

        self.rgb     = sorted(self.rgb)
        self.thermal = sorted(self.thermal)
        self.gt      = sorted(self.gt)

        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        )

        self.thermal_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor()]
        )

        self.gt_transform = transforms.ToTensor()

        self.size = len(self.rgb)
        self.index = 0

    def load_data(self):
        rgb = self.rgb_loader(self.rgb[self.index])
        rgb = self.transform(rgb).unsqueeze(0)

        thermal = self.rgb_loader(self.thermal[self.index])
        thermal = self.thermal_transform(thermal).unsqueeze(0)

        gt = self.binary_loader(self.gt[self.index])

        name = self.rgb[self.index].split('/')[-1]

        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        self.index = self.index % self.size
        return rgb, thermal, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
    def __len__(self):
        return self.size


def get_ctd_loader(gt_root, edge_root, batchsize, trainsize, shuffle=True, num_workers=8, pin_memory=False, split='train'):
    if split == 'train':
        dataset = CTDSalObjDataset(gt_root, edge_root, trainsize)
    else:
        dataset = CTDSalObjDataset_val(gt_root, trainsize)

    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)

    return data_loader

class CTDSalObjDataset(torch.utils.data.Dataset):
    def __init__(self, gt_root, edge_root, trainsize):
        self.trainsize = trainsize
        self.gt       = [gt_root   + '/' + f      for f in os.listdir(gt_root)  if f.endswith('.jpg') or f.endswith('.png')]
        self.edge     = [edge_root + '/' + f      for f in os.listdir(edge_root)if f.endswith('.jpg') or f.endswith('.png')]
        self.name     = [f                        for f in os.listdir(gt_root)  if f.endswith('.jpg') or f.endswith('.png')]

        self.gt       = sorted(self.gt)
        self.edge     = sorted(self.edge)
        self.name     = sorted(self.name)

        self.size = len(self.gt)

        self.rgb_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.thermal_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])

        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])

        self.index = 0


    def __getitem__(self, index):
        gt      = self.binary_loader(self.gt[index])
        edge    = self.binary_loader(self.edge[index])
        name    = self.name[index]

        gt      = self.gt_transform(gt)
        edge    = self.gt_transform(edge)

        return gt, edge, name


    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def resize(self, img, gt, thermal):
        assert img.size == gt.size and gt.size==thermal.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST),thermal.resize((w, h), Image.NEAREST)
        else:
            return img, gt, thermal

    def __len__(self):
        return self.size


class CTDSalObjDataset_val(torch.utils.data.Dataset):
    def __init__(self, gt_root, trainsize):
        self.trainsize = trainsize
        self.gt      = [gt_root            + f for f in os.listdir(gt_root)  if f.endswith('.jpg') or f.endswith('.png')]
        self.name    = [f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]

        self.gt       = sorted(self.gt)
        self.name     = sorted(self.name)

        self.size = len(self.gt)

        self.rgb_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        self.thermal_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])

        self.index = 0

    def __getitem__(self, index):
        gt      = self.binary_loader(self.gt[index])
        name    = self.name[index]

        gt      = self.gt_transform(gt)

        return gt, name

    def load_data(self):

        gt = self.binary_loader(self.gt[self.index])

        name = self.gt[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        self.index = self.index % self.size

        return gt, name


    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')


    def __len__(self):
        return self.size

#test dataset and loader
class test_edge_dataset:
    def __init__(self, gt_root, testsize):
        self.testsize = testsize
        self.gt      = [gt_root      + f for f in os.listdir(gt_root)  if f.endswith('.jpg') or f.endswith('.png')]

        self.gt      = sorted(self.gt)

        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        )

        self.thermal_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor()]
        )

        self.gt_transform = transforms.ToTensor()

        self.size = len(self.gt)
        self.index = 0

    def load_data(self):

        gt = self.binary_loader(self.gt[self.index])

        name = self.gt[self.index].split('/')[-1]

        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        self.index = self.index % self.size
        return gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
    def __len__(self):
        return self.size
