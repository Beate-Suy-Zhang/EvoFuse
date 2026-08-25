
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from einops import rearrange
import numpy as np


class prominent(nn.Module):
    def __init__(self, channels=0, partnum=1, e=1.5):
        super(prominent, self).__init__()
        self.e_lambda = 1e-5
        self.act = nn.Sigmoid()
        self.act2 = nn.SiLU(inplace=True)

    def forward(self, x):
        y = self.act(x)
        return self.act2(x * self.act(y.pow(2) / (self.e_lambda + y.mean(dim=[2, 3], keepdim=True))))


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class RepConv(nn.Module):
    default_act = nn.ReLU(inplace=True)  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True):
        super().__init__()
        assert (k == 3 or k == (3, 3)) and p == 1
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.conv1 = nn.Conv2d(c1, c2, k, s, 1, groups=g, bias=False)
        self.conv2 = nn.Conv2d(c1, c2, 1, s, 0, groups=g, bias=False)
        self.ln = LayerNorm(c2, "withbias")
        # self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        # self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)
        
    def forward_fuse(self, x):
        return self.act(self.ln(self.conv(x)))
    
    def forward(self, x):
        return self.act(self.ln(self.conv1(x) + self.conv2(x)))
    
    def get_equivalent_kernel_bias(self):
        """Returns equivalent kernel and bias by adding 3x3 kernel, 1x1 kernel and identity kernel with their biases."""
        kernel3x3 = self.conv1.weight
        kernel1x1 = self.conv2.weight
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1)
    
    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        """Pads a 1x1 tensor to a 3x3 tensor."""
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, [1, 1, 1, 1])  # (left, right, top, bottom)
        
    def fuse_convs(self):
        """Combines two convolution layers into a single layer and removes unused attributes from the class."""
        if hasattr(self, 'conv'):
            return
        kernel = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(in_channels=self.conv1.in_channels,
                              out_channels=self.conv1.out_channels,
                              kernel_size=self.conv1.kernel_size,
                              stride=self.conv1.stride,
                              padding=self.conv1.padding,
                              dilation=self.conv1.dilation,
                              groups=self.conv1.groups,
                              bias=False).requires_grad_(False)
        self.conv.weight.data = kernel
        for para in self.parameters():
            para.detach_()
        self.__delattr__('conv1')
        self.__delattr__('conv2')
        if hasattr(self, 'nm'):
            self.__delattr__('nm')
        if hasattr(self, 'bn'):
            self.__delattr__('bn')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        


class DRA1BR(nn.Module):
    default_act = nn.ReLU(inplace=True)  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True):
        super().__init__()
        assert (k == 3 or k == (3, 3)) and p == 1
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.conv1 = nn.Conv2d(c1, c2, k, s, 1, groups=g, bias=False)               # high-frequency
        self.conv2 = nn.Conv2d(c1, c2, 1, s, 0, groups=g, bias=False)               # identity
        self.conv3 = nn.Conv2d(c1, c2, (k, 1), s, 0, groups=g, bias=False)          # flat
        self.conv4 = nn.Conv2d(c1, c2, (1, k), s, 0, groups=g, bias=False)          # vertical
        self.conv5 = nn.Conv2d(c1, c2, (k - 1, k - 1), s, 0, groups=g, bias=False)  # low-frequency
        self.weight = nn.Parameter(torch.ones(1, c2, 1, 1))
        self.ln = LayerNorm(c2, "withbias")
        
    def forward_fuse(self, x):
        return self.act(self.ln(self.conv(x)))
    
    def forward(self, x):
        return self.act(self.ln((self.conv1(x) + 
                                self.conv2(x) + 
                                self.conv3(F.pad(x, [0, 0, 1, 1])) + 
                                self.conv4(F.pad(x, [1, 1, 0, 0])) + 
                                self.conv5(F.pad(x, [0, 1, 0, 1]))) * self.weight))
    
    def get_equivalent_kernel_bias(self):
        """Returns equivalent kernel and bias."""
        c2 = self.weight.shape[1]  # number of output channels
        kernel3x3 = self.conv1.weight                                   # high-frequency
        kernel1x1 = self._pad_1x1_to_3x3_tensor(self.conv2.weight)      # identity
        kernel3x1 = self._pad_3x1_to_3x3_tensor(self.conv3.weight)      # flat
        kernel1x3 = self._pad_1x3_to_3x3_tensor(self.conv4.weight)      # vertical
        kernel2x2 = self._pad_2x2_to_3x3_tensor(self.conv5.weight)      # low-frequency
        return (kernel3x3 + kernel1x1 + kernel3x1 + kernel1x3 + kernel2x2) * self.weight.reshape(c2, 1, 1, 1)
    
    
    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        """Pads a 1x1 tensor to a 3x3 tensor."""
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, [1, 1, 1, 1])  # (left, right, top, bottom)
    
    def _pad_1x3_to_3x3_tensor(self, kernel1x3):
        """Pads a 1x1 tensor to a 3x3 tensor."""
        if kernel1x3 is None:
            return 0
        else:
            return F.pad(kernel1x3, [0, 0, 1, 1])  # (left, right, top, bottom)
    
    def _pad_3x1_to_3x3_tensor(self, kernel3x1):
        """Pads a 1x1 tensor to a 3x3 tensor."""
        if kernel3x1 is None:
            return 0
        else:
            return F.pad(kernel3x1, [1, 1, 0, 0])  # (left, right, top, bottom)
    
    def _pad_2x2_to_3x3_tensor(self, kernel2x2):
        """Pads a 1x1 tensor to a 3x3 tensor."""
        if kernel2x2 is None:
            return 0
        else:
            return F.pad(kernel2x2, [1, 0, 1, 0])  # (left, right, top, bottom)
        
    def fuse_convs(self):
        """Combines multiple convolution layers into a single layer and removes unused attributes from the class."""
        if hasattr(self, 'conv'):
            return
        kernel = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(in_channels=self.conv1.in_channels,
                              out_channels=self.conv1.out_channels,
                              kernel_size=self.conv1.kernel_size,
                              stride=self.conv1.stride,
                              padding=self.conv1.padding,
                              dilation=self.conv1.dilation,
                              groups=self.conv1.groups,
                              bias=False).requires_grad_(False)
        self.conv.weight.data = kernel
        for para in self.parameters():
            para.detach_()
        self.__delattr__('conv1')
        self.__delattr__('conv2')
        self.__delattr__('conv3')
        self.__delattr__('conv4')
        self.__delattr__('conv5')
        self.__delattr__('weight')


class Conv(nn.Module):
    default_act = nn.ReLU(inplace=True)  # default activation
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=True)
        self.ln = LayerNorm(c2, "withbias")
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.ln(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=False, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        
        self.cv1 = DRA1BR(c1, c_, k[0], 1)
        self.cv2 = DRA1BR(c_, c2, k[1], 1, g=g)

        self.add = shortcut and c1 == c2
    def forward(self, x):
        return x + (self.cv2(self.cv1(x))) if self.add else (self.cv2(self.cv1(x)))
    
class C2f(nn.Module):
    def __init__(self, c1, c2, n=2, shortcut=True, g=1, e=0.5, partnum=1):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in self.m)
        return self.cv2(torch.cat(y, 1))

# =============================================================================

# =============================================================================
import numbers
##########################################################################
## Layer Norm
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias    
    
    
class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
class Encoder(nn.Module):
    def __init__(self, inp_channels=2, dim=64, num_blocks=1, num=2,):
        super(Encoder, self).__init__()
        self.embed = Conv(inp_channels, dim, 3, 1)
        self.extractor = nn.Sequential(*[C2f(c1=dim, c2=dim, n=num, g=4, e=0.5) for _ in range(num_blocks)])
    
    def forward(self, inp_img):
        embed_features = self.embed(inp_img)
        encode_features = self.extractor(embed_features) + embed_features
        return encode_features

class Decoder(nn.Module):
    def __init__(self, out_channels=1, dim=64, num_blocks=1, num=2, bias=False):
        super(Decoder, self).__init__()
        self.id1 = nn.Conv2d(int(2), int(1), kernel_size=1, padding=0, bias=bias)
        self.id3 = nn.Conv2d(int(2), int(1), kernel_size=3, padding=1, bias=bias)
        self.id5 = nn.Conv2d(int(2), int(1), kernel_size=5, padding=2, bias=bias)
        self.id7 = nn.Conv2d(int(2), int(1), kernel_size=7, padding=3, bias=bias)
        self.extractor = nn.Sequential(*[C2f(c1=dim, c2=dim, n=num, g=4, e=0.5) for _ in range(num_blocks)])
        self.output = nn.Sequential(
            nn.Conv2d(int(dim), out_channels, kernel_size=3,
                    stride=1, padding=1, bias=bias),
            )
        self.sigmoid = nn.Sigmoid()              

    def forward(self, inp_img1, inp_img2, base_feature):
        out_enc_level1 = self.extractor(base_feature)
        id1 = self.id1(torch.cat((inp_img1, inp_img2), dim=1))
        id3 = self.id3(torch.cat((inp_img1, inp_img2), dim=1))
        id5 = self.id5(torch.cat((inp_img1, inp_img2), dim=1))
        id7 = self.id7(torch.cat((inp_img1, inp_img2), dim=1))
        
        out_enc_level1 = self.output(out_enc_level1) + id1 + id3 + id5 + id7
        return self.sigmoid(out_enc_level1)
        
    def forward_fuse(self, inp_img1, inp_img2, base_feature):
        out_enc_level1 = self.extractor(base_feature)
        id = self.idconv(torch.cat((inp_img1, inp_img2), dim=1))
        
        out_enc_level1 = self.output(out_enc_level1) + id
        return self.sigmoid(out_enc_level1)
    
    def get_equivalent_kernel_bias(self):
        kernel7x7 = self.id7.weight                            
        kernel1x1 = self._pad_to_7x7_tensor(self.id1.weight, 3)  
        kernel3x3 = self._pad_to_7x7_tensor(self.id3.weight, 2)     
        kernel5x5 = self._pad_to_7x7_tensor(self.id5.weight, 1)     
        return kernel7x7 + kernel1x1 + kernel3x3 + kernel5x5
    
    
    def _pad_to_7x7_tensor(self, kernel, pad):
        if kernel is None:
            return 0
        else:
            return F.pad(kernel, [pad, pad, pad, pad])  # (left, right, top, bottom)
        
    def fuse_convs(self):
        if hasattr(self, 'idconv'):
            return
        kernel = self.get_equivalent_kernel_bias()
        self.idconv = nn.Conv2d(in_channels=self.id7.in_channels,
                              out_channels=self.id7.out_channels,
                              kernel_size=self.id7.kernel_size,
                              stride=self.id7.stride,
                              padding=self.id7.padding,
                              dilation=self.id7.dilation,
                              groups=self.id7.groups,
                              bias=False).requires_grad_(False)
        self.idconv.weight.data = kernel
        for para in self.parameters():
            para.detach_()
        self.__delattr__('id1')
        self.__delattr__('id3')
        self.__delattr__('id5')
        self.__delattr__('id7')
    
    
class FusionModel(nn.Module):
    def __init__(self, ):
        super(FusionModel, self).__init__()
        self.encoder = Encoder(dim=32)
        self.decoder = Decoder(dim=32)
        
    def forward(self, inp_img1, inp_img2):
        base_feature = self.encoder(torch.cat((inp_img1, inp_img2), dim=1))
        output_gray_images = self.decoder(inp_img1, inp_img2, base_feature)
        return output_gray_images
    
if __name__ == '__main__':
    height = 128
    width = 128
    window_size = 8
    modelE = Encoder().cuda()
    modelD = Decoder().cuda()
