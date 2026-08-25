import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy



class Fusionloss(nn.Module):
    def __init__(self):
        super(Fusionloss, self).__init__()
        self.sobelconv=Sobelxy()

    def forward(self,image_vis,image_ir,generate_img):
        image_y=image_vis[:,:1,:,:]
        x_in_max=torch.max(image_y,image_ir)
        loss_in=F.l1_loss(x_in_max,generate_img)
        y_grad=self.sobelconv(image_y)
        ir_grad=self.sobelconv(image_ir)
        generate_img_grad=self.sobelconv(generate_img)
        x_grad_joint=torch.max(y_grad,ir_grad)
        loss_grad=F.l1_loss(x_grad_joint,generate_img_grad)
        loss_grad = 10*loss_grad
        loss_total=loss_in+loss_grad
        return loss_total,loss_in,loss_grad

class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)


def cc(img1, img2):
    eps = torch.finfo(torch.float32).eps
    """Correlation coefficient for (N, C, H, W) image; torch.float32 [0.,1.]."""
    N, C, _, _ = img1.shape
    img1 = img1.reshape(N, C, -1)
    img2 = img2.reshape(N, C, -1)
    img1 = img1 - img1.mean(dim=-1, keepdim=True)
    img2 = img2 - img2.mean(dim=-1, keepdim=True)
    cc = torch.sum(img1 * img2, dim=-1) / (eps + torch.sqrt(torch.sum(img1 **
                                                                      2, dim=-1)) * torch.sqrt(torch.sum(img2**2, dim=-1)))
    cc = torch.clamp(cc, -1., 1.)
    return cc.mean()



    


def pmt_loss(img1, img2):
    eps = torch.finfo(torch.float32).eps
    """Correlation coefficient for (N, C, H, W) image; torch.float32 [0.,1.]."""
    N, C, h, w = img1.shape
    act = nn.Sigmoid()
    actimg1 = act(img1)
    
    # gapm = nn.AdaptiveAvgPool2d((8, 8))
    # gap = nn.functional.interpolate(gapm(actimg1), (h, w), mode='nearest')  #
    # scalar = act(actimg1.pow(2) / gap).pow(4)
    
    # scalar = act(actimg1.pow(2) / actimg1.mean(dim=[2,3], keepdim=True)).pow(4)
    scalar = (torch.clamp(act(actimg1.pow(2) / actimg1.mean(dim=[2, 3], keepdim=True)), 0.00, 1.)
              * torch.clamp(act(actimg1.pow(2) / actimg1.mean(dim=[1, 3], keepdim=True)), 0.00, 1.)
              * torch.clamp(act(actimg1.pow(2) / actimg1.mean(dim=[1, 2], keepdim=True)), 0.00, 1.) 
    ).pow(4)
    
    pmtimg1 = img1 * scalar
    pmtimg1 = nn.SiLU()(pmtimg1)
    
    # mu_x = img1.mean(dim=[2, 3], keepdim=True)
    # sigma_x = torch.sqrt((img1 - mu_x).mean(dim=[2, 3], keepdim=True).pow(2))
    # mu_x_ = pmtimg1.mean(dim=[2, 3], keepdim=True)
    # sigma_x_ = torch.sqrt((pmtimg1 - mu_x_).mean(dim=[2, 3], keepdim=True).pow(2))
    # pmtimg1 = (pmtimg1 - mu_x_) * sigma_x / sigma_x_ + mu_x
        
        
    pmtimg1=(pmtimg1-torch.min(pmtimg1))/(torch.max(pmtimg1)-torch.min(pmtimg1)) * 1.0
    # pmtimg1 = torch.clamp(pmtimg1, 0., 1.)
    
    # img1 is the original image
    # img2 is the fusion image
    
    pmt_loss=F.l1_loss(img2,pmtimg1)*1.
    # pmt_loss=torch.mean((img2 - pmtimg1) ** 2)*1.2
    
    
    return pmt_loss
    
    # img1 = img1.reshape(N, C, -1)
    # img2 = img2.reshape(N, C, -1)
    # pmt = img2 - (img1 * act(actimg1 / actimg1.mean(dim=-1, keepdim=True)))
    # pmt_loss = torch.norm(img2 - img1, dim=-1) 
    # return pmt_loss.mean() / 200





    

    

def vsm(img_in):
    # 将输入数据移到 GPU 上
    img = copy.deepcopy(img_in).to('cuda')

    # 预先为显著性图和历史数组分配内存
    sal = torch.zeros(256, dtype=torch.int, device='cuda')
    map = torch.zeros_like(img, dtype=torch.int, device='cuda')

    # 计算直方图（使用 GPU）
    for b in range(img.shape[0]):
        # 使用 torch.histc 计算直方图，结果已经是 GPU 上的
        his = torch.histc(img[b], bins=256, min=0, max=1)

        # 使用 torch.arange 和 torch.abs 计算显著性图
        # 先计算显著性图，避免重复计算
        for i in range(256):
            sal[i] = torch.sum(torch.abs(torch.arange(256, device='cuda') - i) * his)

        # 在 map 中应用显著性图
        # 这里通过一次性操作，避免在循环中多次调用 torch.where
        map = torch.round(img[b] * 255).int()  # 转换为整数并乘以255
        for i in range(256):
            map[map == i] = sal[i]  # 为显著性区域赋值

        # 如果显著性图的最大值为 0，直接设置为 0
        if map.max() == 0:
            img[b] = torch.zeros_like(img[b], dtype=torch.int, device='cuda')
        else:
            # 归一化显著性图并转换为浮点数
            img[b] = map.float() / map.max()

    return img

    
    # img = img_in.to('cuda')
    
    for b in range(img.shape[0]):
        # 计算直方图，img[b] 在 GPU 上
        his = torch.histc(img[b], bins=256, min=0, max=1)
        
        # 创建 sal 数组，放到 GPU 上
        sal = torch.zeros(256, dtype=torch.int, device='cuda')
        
        # 计算显著性图
        for i in range(256):
            sal[i] = torch.sum(torch.abs(torch.arange(256, device='cuda') - i) * his)
        
        # 创建 map 数组，放到 GPU 上
        map = torch.zeros_like(img[b], dtype=torch.int, device='cuda')
        
        # 将图像中符合条件的像素值赋予显著性值
        for i in range(256):
            map[torch.where(torch.round(img[b]*255) == i)] = sal[i]
        
        # 如果显著性图的最大值为 0，直接设置为 0
        if map.max() == 0:
            img[b] = torch.zeros_like(img[b], dtype=torch.int, device='cuda')
        else:
            # 归一化显著性图并转换为浮点数
            img[b] = map.float() / map.max()

    # 返回 GPU 上的结果
    return img


    
    img = copy.deepcopy(img_in)
    for b in range(img.shape[0]):
        his = torch.histc(img[b], bins=256, min=0, max=1)
        
        sal = torch.zeros(256, dtype=torch.int)
        for i in range(256):
            # ii = torch.zeros(1)
            # ii.weight=i
            # print(torch.arange(256).device)
            # print(ii.device)
            sal[i] = torch.sum(torch.abs(torch.arange(256) - i) * his.cpu())
        
        map = torch.zeros_like(img[b], dtype=torch.int)
        for i in range(256):
            map[torch.where(torch.round(img[b]*255) == i)] = sal[i]
        
        if map.max() == 0:
            print(11111111111111111111111111111111111111111111111111)
            img[b] = torch.zeros_like(img[b], dtype=torch.int)
        img[b] = map.float() / map.max()

    return img

    
    # B, C, H, W = img.shape
    # his = torch.zeros((B, 256))
    # for i in range(B):
    #     his[i] = torch.histc(img[i], bins=256, min=0, max=1)
    # num_rows = 256
    # num_cols = 256
    # matrix_i = torch.arange(num_cols).repeat(num_rows, 1)
    # matrix_j = torch.transpose(matrix_i, 0, 1)
    # M_absij = torch.abs(matrix_i - matrix_j)
    # sal_M = his.unsqueeze(2) * M_absij.unsqueeze(0)
    # sal = torch.zeros((B, 256))
    # for i in range(B):
    #     for j in range(256):
    #         sal[i] = torch.sum(sal_M[i][j])

        
    # map = torch.zeros_like(img)
    # for i in range(B):
    #     for j in range(256):
    #         map[i][torch.where(img[i] == j)] = sal[i][j]
    # if map.max()==0:
    #     return torch.zeros_like(img)
    # return map / (torch.max(map, 0, keepdim=True)[0])


def vsm_loss(img1, img2):
    
    map1 = vsm(img1)
    # print(img1.shape)
    # print(img2.shape)
    # print(map1.shape)
    # map2 = vsm(img2)

    vsmloss = F.l1_loss(img1 * map1, img2 * map1)
    return vsmloss


