import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import numpy as np

from layers import *


class ResidualConvUnit(nn.Module):
    """Residual convolution module."""

    def __init__(self, features):
        """Init.
        Args:
            features (int): number of features
        """
        super().__init__()

        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )

        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """Forward pass.
        Args:
            x (tensor): input
        Returns:
            tensor: output
        """
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + x

    
class AttentionConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super(AttentionConv, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups

        assert self.out_channels % self.groups == 0, "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"

        self.rel_h = nn.Parameter(torch.randn(1, 1, 1, kernel_size, 1), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn(1, 1, 1, 1, kernel_size), requires_grad=True)
        
        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

        self.reset_parameters()

    def forward(self, x):
        batch, channels, height, width = x.size()

        padded_x = F.pad(x, [self.padding, self.padding, self.padding, self.padding])
        q_out = self.query_conv(x)
        k_out = self.key_conv(padded_x)
        v_out = self.value_conv(padded_x)
        
        k_out = k_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        v_out = v_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)

        k_out_h, k_out_w = k_out.split(self.out_channels // 2, dim=1)
        k_out = torch.cat((k_out_h + self.rel_h, k_out_w + self.rel_w), dim=1)
#         k_out_h = k_out + self.rel_h
#         k_out_w = k_out + self.rel_w

        k_out = k_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)
#         k_out_h = k_out_h.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)
#         k_out_w = k_out_w.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)
        
        v_out = v_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)

        q_out = q_out.view(batch, self.groups, self.out_channels // self.groups, height, width, 1)

        out = q_out * k_out # (k_out_h + k_out_w)
        out = F.softmax(out, dim=-1)
        out = torch.einsum('bnchwk,bnchwk -> bnchw', out, v_out).view(batch, -1, height, width)

        return out

    def reset_parameters(self):
        init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.value_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.query_conv.weight, mode='fan_out', nonlinearity='relu')

        init.normal_(self.rel_h, 0, 1)
        init.normal_(self.rel_w, 0, 1)
    
    
class ResidualAttentionUnit(nn.Module):
    """Residual self-attention module."""

    def __init__(self, features):
        """Init.
        Args:
            features (int): number of features
        """
        super().__init__()

        self.atten1 = AttentionConv(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )

        self.atten2 = AttentionConv(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """Forward pass.
        Args:
            x (tensor): input
        Returns:
            tensor: output
        """
        out = self.relu(x)
        out = self.atten1(out)
        out = self.relu(out)
        out = self.atten2(out)

        return out + x
    
    
class FeatureFusionBlock(nn.Module):
    """Feature fusion block."""

    def __init__(self, features):
        """Init.
        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()

        self.resConfUnit1 = ResidualAttentionUnit(features)
        self.resConfUnit2 = ResidualAttentionUnit(features)
        self.resConfUnit3 = ResidualAttentionUnit(features)
        
        self.conv3x3 = Conv3x3(2, 1)

    def forward(self, dt, upt, dt_1, dt_2):
        """Forward pass.
        Returns:
            tensor: output
        """
        dt_upt = torch.cat([dt, upt], dim=1)
        context = torch.cat([dt_1, dt_2], dim=1)
        
        output = self.resConfUnit1(dt_upt)
        output += self.resConfUnit2(context)

        output = self.conv3x3(self.resConfUnit3(output))

        output_up = nn.functional.interpolate(
            output, scale_factor=2, mode="bilinear", align_corners=True
        )

        return output, output_up


class FeatureFusionBlock_v2(nn.Module):
    """Feature fusion block."""

    def __init__(self, features, scale, init_scale=False):
        """Init.
        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock_v2, self).__init__()
        
        self.scale = scale
        self.init_scale = init_scale
        
        if self.init_scale:
            self.conv_init = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True)

        self.resConfUnit1 = ResidualAttentionUnit(features)
        self.resConfUnit2 = ResidualAttentionUnit(features)
        self.resConfUnit3 = ResidualAttentionUnit(features)
        
        if self.scale == 1:
            self.conv3x3 = Conv3x3(features, 1)
        else:
            self.upscale_1 = UpscalePS(features, 1, self.scale)
            self.upscale_2 = UpscalePS(features, features // 4, 2)
        

    def forward(self, dt, upt, dt_1, dt_2):
        if upt is None:
            dt_upt = self.conv_init(dt)
        else:
            dt_upt = torch.cat([dt, upt], dim=1)
            
        context = torch.cat([dt_1, dt_2], dim=1)
        
        output = self.resConfUnit1(dt_upt)
        output += self.resConfUnit2(context)

        output = self.resConfUnit3(output)
        
        if self.scale == 1:
            output_depth = self.conv3x3(output)
            output_up = None
        else:
            output_depth = self.upscale_1(output)
            output_up = self.upscale_2(output)

        return output_depth, output_up


class UpscalePS(nn.Module):
    
    def __init__(self, input_ch, output_ch, scale):
        super().__init__()
        self.conv = nn.Conv2d(
            input_ch, output_ch * scale ** 2, kernel_size=3, stride=1, padding=1, bias=True)
        self.ps = nn.PixelShuffle(scale)
    
    def forward(self, x):
        return self.ps(torch.tanh(self.conv(x)))
        
        
    
class Fusion(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.fusion_block_1 = FeatureFusionBlock_v2(features=256, scale=8, init_scale=True)
        self.fusion_block_2 = FeatureFusionBlock_v2(features=128, scale=4)
        self.fusion_block_3 = FeatureFusionBlock_v2(features=64, scale=2)
        self.fusion_block_4 = FeatureFusionBlock_v2(features=32, scale=1)
        
    def forward(self, depth_dec_outputs):
        dec_outputs = {}
        dec_outputs_t_1 = {}
        dec_outputs_t_2 = {}
        
        for k, v in depth_dec_outputs.items():
            dec_outputs[k], dec_outputs_t_1[k], dec_outputs_t_2[k] = v.split(len(v) // 3)       
                
        outputs = {}

        outputs[("disp", 3)], output_up = self.fusion_block_1(dt=dec_outputs[("pre_disp", 3)], 
                                                    upt=None, 
                                                    dt_1=dec_outputs_t_1[("pre_disp", 3)], 
                                                    dt_2=dec_outputs_t_2[("pre_disp", 3)])
        outputs[("disp", 2)], output_up = self.fusion_block_2(dt=dec_outputs[("pre_disp", 2)], 
                                                    upt=output_up, 
                                                    dt_1=dec_outputs_t_1[("pre_disp", 2)], 
                                                    dt_2=dec_outputs_t_2[("pre_disp", 2)])
        outputs[("disp", 1)], output_up = self.fusion_block_3(dt=dec_outputs[("pre_disp", 1)], 
                                                    upt=output_up, 
                                                    dt_1=dec_outputs_t_1[("pre_disp", 1)], 
                                                    dt_2=dec_outputs_t_2[("pre_disp", 1)])
        outputs[("disp", 0)], output_up = self.fusion_block_4(dt=dec_outputs[("pre_disp", 0)], 
                                                    upt=output_up, 
                                                    dt_1=dec_outputs_t_1[("pre_disp", 0)], 
                                                    dt_2=dec_outputs_t_2[("pre_disp", 0)])
        
        return outputs

    
        
class FeatureFusionBlock_v3(nn.Module):
    """Feature fusion block."""

    def __init__(self, features, attention=True, init_scale=False):
        """Init.
        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock_v3, self).__init__()
        
        self.init_scale = init_scale
        
        if self.init_scale:
            self.conv_1 = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1, bias=True)
        
        if attention:
            self.resConfUnit1 = ResidualAttentionUnit(features)
            self.resConfUnit2 = ResidualAttentionUnit(features)
            self.resConfUnit3 = ResidualAttentionUnit(features * 2)
        else:
            self.resConfUnit1 = ResidualConvUnit(features)
            self.resConfUnit2 = ResidualConvUnit(features)
            self.resConfUnit3 = ResidualConvUnit(features * 2)
        
        self.conv3x3 = Conv3x3(features * 2, 1)
        self.upscale = UpscalePS(features * 2, 1, 2)

        
    def forward(self, dt, upt, dt_1, dt_2):
        
        if self.init_scale: # upt is None:
            dt_upt = self.conv_1(dt)
        else:
            dt_upt = torch.cat([dt, upt], dim=1)
            
        context = torch.cat([dt_1, dt_2], dim=1)
        output = torch.cat([self.resConfUnit1(dt_upt), self.resConfUnit2(context)], dim=1)

        output = self.resConfUnit3(output)
        
        output_depth = self.conv3x3(output)
        output_up = self.upscale(output)
        
        return output_depth, output_up
      
        
class Fusion_v3(nn.Module):
    
    def __init__(self, attention=True):
        super().__init__()
        print('attention', attention)
        self.fusion_block_1 = FeatureFusionBlock_v3(features=2, attention=attention, init_scale=True)
        self.fusion_block_2 = FeatureFusionBlock_v3(features=2, attention=attention)
        self.fusion_block_3 = FeatureFusionBlock_v3(features=2, attention=attention)
        self.fusion_block_4 = FeatureFusionBlock_v3(features=2, attention=attention)
            
        
    def forward(self, depth_dec_outputs):
        dec_outputs = {}
        dec_outputs_t_1 = {}
        dec_outputs_t_2 = {}
        
        for k, v in depth_dec_outputs.items():
            dec_outputs[k], dec_outputs_t_1[k], dec_outputs_t_2[k] = v.split(len(v) // 3)       
                
        outputs = {}

        outputs[("disp", 3)], output_up = self.fusion_block_1(dt=dec_outputs[("disp", 3)], 
                                                    upt=None, 
                                                    dt_1=dec_outputs_t_1[("disp", 3)], 
                                                    dt_2=dec_outputs_t_2[("disp", 3)])
        outputs[("disp", 2)], output_up = self.fusion_block_2(dt=dec_outputs[("disp", 2)], 
                                                    upt=output_up, 
                                                    dt_1=dec_outputs_t_1[("disp", 2)], 
                                                    dt_2=dec_outputs_t_2[("disp", 2)])
        outputs[("disp", 1)], output_up = self.fusion_block_3(dt=dec_outputs[("disp", 1)], 
                                                    upt=output_up, 
                                                    dt_1=dec_outputs_t_1[("disp", 1)], 
                                                    dt_2=dec_outputs_t_2[("disp", 1)])
        outputs[("disp", 0)], output_up = self.fusion_block_4(dt=dec_outputs[("disp", 0)], 
                                                    upt=output_up, 
                                                    dt_1=dec_outputs_t_1[("disp", 0)], 
                                                    dt_2=dec_outputs_t_2[("disp", 0)])
        
        return outputs