import os

import cv2
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms


image_WIDTH = 1024 // 2
image_HEIGHT = 768 // 2

image_WIDTH = 1389
image_HEIGHT = 926



def get_test_loader(ir_root, vis_root, num_workers=0, image_size=(0,0), batchsize=1, testsize=320,
                    shuffle=False, pin_memory=True):
    # print(image_size)
    # dataset = TestDataset(ir_root=ir_root, vis_root=vis_root, testsize=testsize)
    dataset = TestFusionDataset(ir_root=ir_root, vis_root=vis_root, testsize=image_size)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


def rgb_loader(path, image_size):
    with open(path, 'rb') as f:
        img = Image.open(f)
        if image_size is not None:
            img = img.resize((image_size[0], image_size[1]), Image.BILINEAR)
        return img.convert('RGB').convert('YCbCr')
        return img.convert('L')


def rgb_grey_loader(path, image_size):
    with open(path, 'rb') as f:
        img = Image.open(f)
        if image_size is not None:
            img = img.resize((image_size[0], image_size[1]), Image.BILINEAR)
        return img.convert('L')


class TestFusionDataset(data.Dataset):
    def __init__(self, ir_root, vis_root, testsize):
        self.testsize = testsize
        # get filenames
        self.irimages = [os.path.join(ir_root, f) for f in os.listdir(ir_root)
                         if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.bmp')]
        self.visimages = [os.path.join(vis_root, f) for f in os.listdir(vis_root)
                          if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.bmp')]

        # sorted files
        self.irimages = sorted(self.irimages)
        self.visimages = sorted(self.visimages)

        # transforms

        self.img_transform = transforms.Compose([transforms.ToTensor()])
        self.toPIL = transforms.ToPILImage()
        self.size = len(self.visimages)
        if len(self.visimages) != len(self.irimages):
            raise ValueError('ir and vis img num is different.')

    def __getitem__(self, index):
        # read imgs
        irimage = self.gray_loader(self.irimages[index], self.testsize)
        visimage_rgb = rgb_loader(self.visimages[index], self.testsize)
        # visimage_rgb_grey = rgb_grey_loader(self.visimages[index], self.testsize)
        
        irimage = self.img_transform(irimage)
        visimage_rgb = self.img_transform(visimage_rgb)
        # visimage_rgb_grey = self.img_transform(visimage_rgb_grey)
        return irimage, visimage_rgb, self.irimages[index]
        
        
        visimage_bri, visimage_clr = self.bri_clr_loader(self.visimages[index], self.testsize)
        visimage_bri = self.toPIL(visimage_bri)
        visimage_clr = self.toPIL(visimage_clr)
        visimage_bri = self.img_transform(visimage_bri)
        visimage_clr = self.img_transform(visimage_clr)
        return irimage, visimage_rgb, visimage_bri, visimage_clr, self.irimages[index]
        return irimage, visimage_rgb, visimage_rgb_grey, visimage_bri, visimage_clr, self.irimages[index]

    def bri_clr_loader(self, path, image_size):
        img1 = cv2.imread(path)
        if image_size is not None:
            img1 = cv2.resize(img1, (image_size[0], image_size[1]),
                            interpolation=cv2.INTER_LINEAR)
        img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2HSV)
        color = img1[:, :, 0:2]
        brightness = img1[:, :, 2]
        return brightness, color

    def gray_loader(self, path, image_size):
        with open(path, 'rb') as f:
            img = Image.open(f)
            if image_size is not None:
                img = img.resize((image_size[0], image_size[1]), Image.BILINEAR)
            return img.convert('L')

    def binary_loader(self, path, image_size):
        with open(path, 'rb') as f:
            img = Image.open(f)
            if image_size is not None:
                img = img.resize((image_size[0], image_size[1]), Image.BILINEAR)
            return img.convert('L')

    def __len__(self):
        return self.size