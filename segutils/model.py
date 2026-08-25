
import torch
import torch.nn as nn
import torch.nn.functional as F 
from mmseg.models.backbones.mit import MixVisionTransformer
from mmseg.models.decode_heads import SegformerHead

class WeTr(nn.Module):
    def __init__(self, backbone, num_classes=20, embedding_dim=256, pretrained=None, pretrained_path=None):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.feature_strides = [4, 8, 16, 32]
        #self.in_channels = [32, 64, 160, 256]
        #self.in_channels = [64, 128, 320, 512]
        
        norm_cfg = dict(type='SyncBN', requires_grad=True)

        self.backbone = MixVisionTransformer(
                in_channels=3,
                embed_dims=32,
                num_stages=4,
                num_layers=[2, 2, 2, 2],
                num_heads=[1, 2, 5, 8],
                patch_sizes=[7, 3, 3, 3],
                sr_ratios=[8, 4, 2, 1],
                out_indices=(0, 1, 2, 3),
                mlp_ratio=4,
                qkv_bias=True,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                drop_path_rate=0.1)
        # self.in_channels = self.encoder.embed_dims
        self.decode_head = SegformerHead(
                    in_channels=[32, 64, 160, 256],
                    in_index=[0, 1, 2, 3],
                    channels=256,
                    dropout_ratio=0.1,
                    num_classes=num_classes,
                    norm_cfg=norm_cfg,
                    align_corners=False,
                    loss_decode=dict(
                        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))
        # self.classifier = nn.Conv2d(in_channels=self.in_channels[-1], out_channels=self.num_classes, kernel_size=1, bias=False)

        ## initilize encoder
        if pretrained:
            # state_dict = torch.load('pretrained/'+backbone+'.pth')
            # state_dict = torch.load("./FMB/8000iters/iter_60000_best_0.491828.pth")
            # state_dict = torch.load("./best_mIoU_iter_36000.pth")['state_dict']
            # state_dict = torch.load("./FMB/8000iters/iter_25000_best_0.615577.pth")
            state_dict = torch.load(pretrained_path)['state_dict']
            # state_dict.pop('head.weight')
            # state_dict.pop('head.bias')
            self.load_state_dict(state_dict,)
            
    def forward(self, x):
        x = self.backbone(x)
        x = self.decode_head(x)
        return x

    def get_param_groups(self):
        param_groups = [[], [], []] # 
        for name, param in list(self.backbone.named_parameters()):
            if "norm" in name:
                param_groups[1].append(param)
            else:
                param_groups[0].append(param)
        for param in list(self.decode_head.parameters()):
            param_groups[2].append(param)
        # param_groups[2].append(self.classifier.weight)
        return param_groups


class original_WeTr(nn.Module):
    def __init__(self, backbone, num_classes=20, embedding_dim=256, pretrained=None):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.feature_strides = [4, 8, 16, 32]
        #self.in_channels = [32, 64, 160, 256]
        #self.in_channels = [64, 128, 320, 512]

        self.encoder = getattr(mix_transformer, backbone)()
        self.in_channels = self.encoder.embed_dims
        self.decoder = SegFormerHead(feature_strides=self.feature_strides, in_channels=self.in_channels, embedding_dim=self.embedding_dim, num_classes=self.num_classes)
        self.classifier = nn.Conv2d(in_channels=self.in_channels[-1], out_channels=self.num_classes, kernel_size=1, bias=False)

        ## initilize encoder
        if pretrained:
            # state_dict = torch.load('pretrained/'+backbone+'.pth')
            # state_dict = torch.load("./FMB/8000iters/iter_60000_best_0.491828.pth")
            state_dict = torch.load("./best_mIoU_iter_36000.pth")['state_dict']
            # state_dict.pop('head.weight')
            # state_dict.pop('head.bias')
            self.load_state_dict(state_dict,)

        

    def _forward_cam(self, x):
        
        cam = F.conv2d(x, self.classifier.weight)
        cam = F.relu(cam)
        
        return cam

    def get_param_groups(self):

        param_groups = [[], [], []] # 
        
        for name, param in list(self.encoder.named_parameters()):
            if "norm" in name:
                param_groups[1].append(param)
            else:
                param_groups[0].append(param)

        for param in list(self.decoder.parameters()):

            param_groups[2].append(param)
        
        param_groups[2].append(self.classifier.weight)

        return param_groups

    def forward(self, x):

        _x = self.encoder(x)
        _x1, _x2, _x3, _x4 = _x
        cls = self.classifier(_x4)

        return self.decoder(_x)
