import torch
import torch.nn as nn
from torch.nn import functional as F

__all__ = ('ChannelAttention', 'SpatialAttention', 'CBAM', 'TripletAttention', 'SEBlock', 'eca_layer',
           'simam_module', 'NAM', 'prominent')


class prominent(nn.Module):
    def __init__(self, channels, partnum=4, e=1.5):
        super(prominent, self).__init__()

        self.act = nn.Sigmoid()
        self.act2 = nn.SiLU()
        self.gaps = nn.AdaptiveAvgPool2d((1, 1))
        #self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        #self.weight2 = nn.Parameter(torch.ones(1, channels, 1, 1))

    def forward(self, x):
        #return self.nam(x)
        
        y = self.act(x)
        
        return self.act2(x * self.act(y / y.mean(dim=[1, 2, 3], keepdim=True)) *
                         self.act(y / self.gaps(y)))
        
        return (x * self.act(y / y.mean(dim=[1, 2, 3], keepdim=True)).pow(self.weight) *
                self.act(y / self.gaps(y)).pow(self.weight2))
        
        return self.act2(x * self.act(self.act(x) / self.gaps(self.act(x))))
        
        return self.act2(x * (self.act(y / y.mean(dim=[1, 2, 3], keepdim=True)) * self.act(y / self.gaps(y))).pow(self.repeats))
        
        return self.act2(x * self.act(y / y.mean(dim=[1, 2, 3], keepdim=True)) *
                         self.act(y / self.gaps(y)))
        
        return self.act2(x * self.act(y / y.mean(dim=[1, 2, 3], keepdim=True)).pow(self.weight) *
                         self.act(y / self.gaps(y)).pow(self.weight2))
        
        return x * self.act(y / y.mean(dim=[1, 2, 3], keepdim=True)) * self.act(y / self.gaps(y))


class BasicConv(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        relu=True,
        bn=True,
        bias=False,
    ):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = (
            nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
            if bn
            else None
        )
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1
        )


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(
            2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False
        )

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale


class TripletAttention(nn.Module):
    def __init__(
        self,
        gate_channels,
        reduction_ratio=16,
        pool_types=["avg", "max"],
        no_spatial=False,
    ):
        super(TripletAttention, self).__init__()
        self.ChannelGateH = SpatialGate()
        self.ChannelGateW = SpatialGate()
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.ChannelGateH(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.ChannelGateW(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
        if not self.no_spatial:
            x_out = self.SpatialGate(x)
            x_out = (1 / 3) * (x_out + x_out11 + x_out21)
        else:
            x_out = (1 / 2) * (x_out11 + x_out21)
        return x_out


class eca_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class SEBlock(nn.Module):
    # conv+sigmoid(-1)   conv+in+relu(0)    conv+in+sigmoid(1)
    def __init__(self, planes):
        super().__init__()
        self.planes = planes
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.silu = nn.SiLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Conv2d(self.planes, self.planes // 16, 1)  # Use nn.Conv2d instead of nn.Linear
        self.fc2 = nn.Conv2d(self.planes // 16, self.planes, 1)

    def forward(self, x):
        out = x
        # gap = self.gap(x)

        # gap = gap.view(gap.size(0), -1)

        # gap = self.relu(self.fc1(gap))
        # gap = self.sigmoid(self.fc2(gap))

        # gap = gap.view(out.size(0), out.size(1), 1, 1)

        out = out * self.sigmoid(self.fc2(self.silu(self.fc1(self.gap(x)))))
        return out


class ChannelAttention(nn.Module):
    """Channel-attention module https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet."""

    def __init__(self, channels: int) -> None:
        """Initializes the class and sets the basic configurations and instance variables required."""
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies forward pass using activation on convolutions of the input, optionally using batch normalization."""
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    """Spatial-attention module."""

    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module with kernel size argument."""
        super().__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply channel and spatial attention on input for feature recalibration."""
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, c1, kernel_size=7):
        """Initialize CBAM with given input channel (c1) and kernel size."""
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """Applies the forward pass through C1 module."""
        return self.spatial_attention(self.channel_attention(x))


class simam_module(torch.nn.Module):
    def __init__(self, channels = None, e_lambda = 1e-4):
        super(simam_module, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        # print(4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n))
        # print(y)

        return x * self.activaton(y)


class Channel_Att(nn.Module):
    def __init__(self, channels, t=16):
        super(Channel_Att, self).__init__()
        self.channels = channels
      
        self.bn2 = nn.BatchNorm2d(self.channels, affine=True)


    def forward(self, x):
        residual = x

        x = self.bn2(x)
        weight_bn = self.bn2.weight.data.abs() / torch.sum(self.bn2.weight.data.abs())
        x = x.permute(0, 2, 3, 1).contiguous()
        x = torch.mul(weight_bn, x)
        x = x.permute(0, 3, 1, 2).contiguous()
        
        x = torch.sigmoid(x) * residual #
        
        return x


class NAM(nn.Module):
    def __init__(self, channels, out_channels=None, no_spatial=True):
        super(NAM, self).__init__()
        self.Channel_Att = Channel_Att(channels)
  
    def forward(self, x):
        x_out1=self.Channel_Att(x)
 
        return x_out1  

