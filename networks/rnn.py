import torch
import torch.nn as nn

import math
import numpy as np
import torch.nn.functional as F
import torch.nn.init as init

import numpy as np
import math



class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out


class ConvLSTMCell_v1(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell_v1, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next
    
    
class ConvLSTMModel_v1(nn.Module):
    def __init__(self, dim_dict, kernel_size, bias, device):
        super(ConvLSTMModel_v1, self).__init__()

        self.clstm_1 = ConvLSTMCell_v1(dim_dict['input_dim'], dim_dict['hidden_dim_1'], kernel_size, bias)
        self.h0_layer1, self.c0_layer1 = self.init_hidden(dim_dict['hidden_dim_1'], dim_dict['height'], dim_dict['width'], device)
        self.height = dim_dict['height']
        self.width = dim_dict['width']
        self.hidden_dim = dim_dict['hidden_dim_1']

    def init_hidden(self, hidden_dim, height, width, device):
        return (nn.Parameter(torch.zeros(1, hidden_dim, height, width, device=device), requires_grad=True),
                nn.Parameter(torch.zeros(1, hidden_dim, height, width, device=device), requires_grad=True))

    def forward(self, x, hidden_state_1):
        return self.clstm_1(x, hidden_state_1)
    
    

class ConvGRUCell(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias):
        super(ConvGRUCell, self).__init__()

        self.height, self.width = input_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.hidden_dim = hidden_dim
        self.bias = bias
        self.conv_gates = nn.Conv2d(in_channels=input_dim + hidden_dim,
                                    out_channels=2*self.hidden_dim,  # for update_gate,reset_gate respectively
                                    kernel_size=kernel_size,
                                    padding=self.padding,
                                    bias=self.bias)

        self.conv_can = nn.Conv2d(in_channels=input_dim+hidden_dim,
                              out_channels=self.hidden_dim, # for candidate neural memory
                              kernel_size=kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_dim, self.height, self.width)

    def forward(self, input_tensor, h_cur):
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv_gates(combined)

        gamma, beta = torch.split(combined_conv, self.hidden_dim, dim=1)
        reset_gate = torch.sigmoid(gamma)
        update_gate = torch.sigmoid(beta)

        combined = torch.cat([input_tensor, reset_gate*h_cur], dim=1)
        cc_cnm = self.conv_can(combined)
        cnm = torch.tanh(cc_cnm)
#         cnm = F.elu(cc_cnm)
        h_next = (1 - update_gate) * h_cur + update_gate * cnm
        
        return h_next

    
    
class ConvGRUModel_v1(nn.Module):
    def __init__(self, dim_dict, kernel_size, bias, device):
        super(ConvGRUModel_v1, self).__init__()

        self.cgru_1 = ConvGRUCell((dim_dict['height'], dim_dict['width']), 
                                  dim_dict['input_dim'], dim_dict['hidden_dim_1'], kernel_size, bias)
        self.h0_layer1 = self.init_hidden(dim_dict['hidden_dim_1'], 
                                          dim_dict['height'], dim_dict['width'], device)
        self.height = dim_dict['height']
        self.width = dim_dict['width']
        self.hidden_dim = dim_dict['hidden_dim_1']

    def init_hidden(self, hidden_dim, height, width, device):
        return nn.Parameter(torch.zeros(1, hidden_dim, height, width, device=device), requires_grad=True)

    def forward(self, x, hidden_state_1):
        return self.cgru_1(x, hidden_state_1)
    
    
    
class ConvGRUModel_v2(nn.Module):
    def __init__(self, dim_dict, kernel_size, bias, device):
        super(ConvGRUModel_v2, self).__init__()

        self.cgru_1 = ConvGRUCell((dim_dict['height'], dim_dict['width']), 
                                  dim_dict['input_dim'], dim_dict['hidden_dim_1'], kernel_size, bias)
        self.cgru_2 = ConvGRUCell((dim_dict['height'], dim_dict['width']), 
                                  dim_dict['hidden_dim_1'], dim_dict['hidden_dim_2'], kernel_size, bias)
        self.h0_layer1 = self.init_hidden(dim_dict['hidden_dim_1'], dim_dict['height'], dim_dict['width'], device)
        self.h0_layer2 = self.init_hidden(dim_dict['hidden_dim_2'], dim_dict['height'], dim_dict['width'], device)
        self.height = dim_dict['height']
        self.width = dim_dict['width']
        self.hidden_dim = dim_dict['hidden_dim_1']

    def init_hidden(self, hidden_dim, height, width, device):
        return nn.Parameter(torch.zeros(1, hidden_dim, height, width, device=device), requires_grad=True)

    def forward(self, x, hidden_state_1, hidden_state_2):
        hidden_state_1 = self.cgru_1(x, hidden_state_1)
        hidden_state_2 = self.cgru_2(hidden_state_1, hidden_state_2)
        
        return hidden_state_1, hidden_state_2


    
class ConvGRUBlocks(nn.Module):
    """GRU based on disparity (input with channel = 1)
    """
    def __init__(self, kernel_size, bias, device):
        super(ConvGRUBlocks, self).__init__()
        
        # scale 0
        dim_dict_0 = {
            'input_dim':1,
            'width': 640,
            'height': 192,
            'hidden_dim_1':1,
            'hidden_dim_2':1
        }
        
        self.cgru_0 = ConvGRUModel_v1(dim_dict_0, kernel_size, bias, device)
        
        # scale 1
        dim_dict_1 = {
            'input_dim':1,
            'width': 320,
            'height': 96,
            'hidden_dim_1':1,
            'hidden_dim_2':1
        }
        
        self.cgru_1 = ConvGRUModel_v1(dim_dict_1, kernel_size, bias, device)

        # scale 2
        dim_dict_2 = {
            'input_dim':1,
            'width': 160,
            'height': 48,
            'hidden_dim_1':1,
            'hidden_dim_2':1
        }
        
        self.cgru_2 = ConvGRUModel_v1(dim_dict_2, kernel_size, bias, device)
        
        # scale 3
        dim_dict_3 = {
            'input_dim':1,
            'width': 80,
            'height': 24,
            'hidden_dim_1':1,
            'hidden_dim_2':1
        }
        
        self.cgru_3 = ConvGRUModel_v1(dim_dict_3, kernel_size, bias, device)  
        
        self.conv3x3_0 = Conv3x3(1, 1)
        self.conv3x3_1 = Conv3x3(1, 1)
        self.conv3x3_2 = Conv3x3(1, 1)
        self.conv3x3_3 = Conv3x3(1, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, dec_outputs, hidden_states):
        hidden_states_new = []
        disp = {}
        
        
        hidden_states_new.append(self.cgru_0(dec_outputs[('disp', 0)], hidden_states[0]))
        hidden_states_new.append(self.cgru_1(dec_outputs[('disp', 1)], hidden_states[1]))
        hidden_states_new.append(self.cgru_2(dec_outputs[('disp', 2)], hidden_states[2]))
        hidden_states_new.append(self.cgru_3(dec_outputs[('disp', 3)], hidden_states[3]))
        
        x = hidden_states_new[0]#.permute(0, 2, 3, 1)
        disp[('disp', 0)] = self.sigmoid(self.conv3x3_0(x))
        
        x = hidden_states_new[1]#.permute(0, 2, 3, 1)
        disp[('disp', 1)] = self.sigmoid(self.conv3x3_1(x))
        
        x = hidden_states_new[2]#.permute(0, 2, 3, 1)
        disp[('disp', 2)] = self.sigmoid(self.conv3x3_2(x))
        
        x = hidden_states_new[3]#.permute(0, 2, 3, 1)
        disp[('disp', 3)] = self.sigmoid(self.conv3x3_3(x))
        
        return hidden_states_new, disp
     
        


class ConvGRUBlocks_v2(nn.Module):
    """GRU based on disparity concated with upscaled depth 
    from previous scale (input with channel = 2)
    """
    def __init__(self, kernel_size, bias, device, attention=True):
        super(ConvGRUBlocks_v2, self).__init__()
        
        # scale 0
        dim_dict_0 = {
            'input_dim':2,
            'width': 640,
            'height': 192,
            'hidden_dim_1':2,
        }
        
        self.cgru_0 = ConvGRUModel_v1(dim_dict_0, kernel_size, bias, device)
        
        # scale 1
        dim_dict_1 = {
            'input_dim':2,
            'width': 320,
            'height': 96,
            'hidden_dim_1':2,
        }
        
        self.cgru_1 = ConvGRUModel_v1(dim_dict_1, kernel_size, bias, device)

        # scale 2
        dim_dict_2 = {
            'input_dim':2,
            'width': 160,
            'height': 48,
            'hidden_dim_1':2,
        }
        
        self.cgru_2 = ConvGRUModel_v1(dim_dict_2, kernel_size, bias, device)
        
        # scale 3
        dim_dict_3 = {
            'input_dim':1, # first level has no prev upscaled depth 
            'width': 80,
            'height': 24,
            'hidden_dim_1':2,
        }
        
        self.cgru_3 = ConvGRUModel_v1(dim_dict_3, kernel_size, bias, device)  
        
        # fusion
        self.fusion_0 = FeatureFusionBlock(2, up=False, attention=attention)
        self.fusion_1 = FeatureFusionBlock(2, up=True, attention=attention)
        self.fusion_2 = FeatureFusionBlock(2, up=True, attention=attention)
        self.fusion_3 = FeatureFusionBlock(2, up=True, attention=attention)
        
    def forward(self, dec_outputs, hidden_states):
        hidden_states_new = [] 
        disp = {}
        
        # scale 3
        cgru_input = dec_outputs[('disp', 3)]  
        hidden_new_3 = self.cgru_3(cgru_input, hidden_states[3])
        fusion_input_1 = torch.cat([dec_outputs[('disp', 3)], dec_outputs[('disp', 3)]], 1)
        fusion_input_2 = hidden_new_3
        out, up_out = self.fusion_3(fusion_input_1, fusion_input_2) 
        disp[('disp', 3)] = out
        
        
        # scale 2
        cgru_input = torch.cat([dec_outputs[('disp', 2)], up_out], 1) 
        hidden_new_2 = self.cgru_2(cgru_input, hidden_states[2])
        fusion_input_1 = cgru_input
        fusion_input_2 = hidden_new_2
        out, up_out = self.fusion_2(fusion_input_1, fusion_input_2) 
        disp[('disp', 2)] = out
        
        
        # scale 1
        cgru_input =  torch.cat([dec_outputs[('disp', 1)], up_out], 1) 
        hidden_new_1 = self.cgru_1(cgru_input, hidden_states[1])
        fusion_input_1 = cgru_input
        fusion_input_2 = hidden_new_1
        out, up_out = self.fusion_1(fusion_input_1, fusion_input_2)
        disp[('disp', 1)] = out
        

        # scale 0
        cgru_input = torch.cat([dec_outputs[('disp', 0)], up_out], 1)
        hidden_new_0 = self.cgru_0(cgru_input, hidden_states[0])
        fusion_input_1 = cgru_input
        fusion_input_2 = hidden_new_0
        out = self.fusion_0(fusion_input_1, fusion_input_2)
        disp[('disp', 0)] = out
        

        hidden_states_new.append(hidden_new_0)
        hidden_states_new.append(hidden_new_1)
        hidden_states_new.append(hidden_new_2)
        hidden_states_new.append(hidden_new_3)

        return hidden_states_new, disp  

    
class ConvGRUBlocks_v8(nn.Module):
    """GRU based on disparity concated with upscaled depth 
    from previous scale (input with channel = 2)
    """
    def __init__(self, kernel_size, bias, device, attention=True):
        super(ConvGRUBlocks_v8, self).__init__()
        
        # scale 0
        dim_dict_0 = {
            'input_dim':16 * 2,
            'width': 640,
            'height': 192,
            'hidden_dim_1':16 * 2,
        }
        
        self.cgru_0 = ConvLSTMModel_v1(dim_dict_0, kernel_size, bias, device)
        
        # scale 1
        dim_dict_1 = {
            'input_dim':32 * 2,
            'width': 320,
            'height': 96,
            'hidden_dim_1':32 * 2,
        }
        
        self.cgru_1 = ConvLSTMModel_v1(dim_dict_1, kernel_size, bias, device)

        # scale 2
        dim_dict_2 = {
            'input_dim':64 * 2,
            'width': 160,
            'height': 48,
            'hidden_dim_1':64 * 2,
        }
        
        self.cgru_2 = ConvLSTMModel_v1(dim_dict_2, kernel_size, bias, device)
        
        # scale 3
        dim_dict_3 = {
            'input_dim':128, # 1 # first level has no prev upscaled depth 
            'width': 80,
            'height': 24,
            'hidden_dim_1':128 * 2,
        }
        
        self.cgru_3 = ConvLSTMModel_v1(dim_dict_3, kernel_size, bias, device)  
        
        # fusion
        self.fusion_0 = FeatureFusionBlock_v2(16 * 2, up=False, attention=attention)
        self.fusion_1 = FeatureFusionBlock_v2(32 * 2, up=True, attention=attention)
        self.fusion_2 = FeatureFusionBlock_v2(64 * 2, up=True, attention=attention)
        self.fusion_3 = FeatureFusionBlock_v2(128 * 2, up=True, attention=attention)
        
    def forward(self, dec_outputs, hidden_states):
        hidden_states_new = [] 
        disp = {}

        # scale 3
        cgru_input = dec_outputs[('disp', 3)]
        hidden_new_3 = self.cgru_3(cgru_input, hidden_states[3])
        fusion_input_1 = torch.cat([dec_outputs[('disp', 3)], dec_outputs[('disp', 3)]], 1)
        fusion_input_2 = (hidden_states[3][0] + hidden_new_3[0]) / 2 # we take hidden (not cell state)
        out, up_out = self.fusion_3(fusion_input_1, fusion_input_2) 
        disp[('disp', 3)] = out

        
        
        # scale 2
        cgru_input = torch.cat([dec_outputs[('disp', 2)], up_out], 1) 
        hidden_new_2 = self.cgru_2(cgru_input, hidden_states[2])
        fusion_input_1 = cgru_input
        fusion_input_2 = (hidden_states[2][0] + hidden_new_2[0]) / 2
        out, up_out = self.fusion_2(fusion_input_1, fusion_input_2) 
        disp[('disp', 2)] = out
        
        
        # scale 1
        cgru_input =  torch.cat([dec_outputs[('disp', 1)], up_out], 1) 
        hidden_new_1 = self.cgru_1(cgru_input, hidden_states[1])
        fusion_input_1 = cgru_input
        fusion_input_2 = (hidden_states[1][0] + hidden_new_1[0]) / 2
        out, up_out = self.fusion_1(fusion_input_1, fusion_input_2)
        disp[('disp', 1)] = out
        

        # scale 0
        cgru_input = torch.cat([dec_outputs[('disp', 0)], up_out], 1)
        hidden_new_0 = self.cgru_0(cgru_input, hidden_states[0])
        fusion_input_1 = cgru_input
        fusion_input_2 = (hidden_states[0][0] + hidden_new_0[0]) / 2
        out = self.fusion_0(fusion_input_1, fusion_input_2)
        disp[('disp', 0)] = out
        
        hidden_states_new.append(hidden_new_0)
        hidden_states_new.append(hidden_new_1)
        hidden_states_new.append(hidden_new_2)
        hidden_states_new.append(hidden_new_3)

        return hidden_states_new, disp 
    

class ConvGRUBlocks_v9(nn.Module):
    """GRU based on disparity concated with upscaled depth 
    from previous scale (input with channel = 2)
    """
    def __init__(self, kernel_size, bias, device, attention=True):
        super(ConvGRUBlocks_v9, self).__init__()
        
        # scale 0
        dim_dict_0 = {
            'input_dim':16 * 2,
            'width': 640,
            'height': 192,
            'hidden_dim_1':16 * 2,
        }
        
        self.cgru_0 = ConvGRUModel_v1(dim_dict_0, kernel_size, bias, device)
        
        # scale 1
        dim_dict_1 = {
            'input_dim':32 * 2,
            'width': 320,
            'height': 96,
            'hidden_dim_1':32 * 2,
        }
        
        self.cgru_1 = ConvGRUModel_v1(dim_dict_1, kernel_size, bias, device)

        # scale 2
        dim_dict_2 = {
            'input_dim':64 * 2,
            'width': 160,
            'height': 48,
            'hidden_dim_1':64 * 2,
        }
        
        self.cgru_2 = ConvGRUModel_v1(dim_dict_2, kernel_size, bias, device)
        
        # scale 3
        dim_dict_3 = {
            'input_dim':128, # 1 # first level has no prev upscaled depth 
            'width': 80,
            'height': 24,
            'hidden_dim_1':128 * 2,
        }
        
        self.cgru_3 = ConvGRUModel_v1(dim_dict_3, kernel_size, bias, device)  
        
        # fusion
        self.fusion_0 = FeatureFusionBlock_v2(16 * 2, up=False, attention=attention)
        self.fusion_1 = FeatureFusionBlock_v2(32 * 2, up=True, attention=attention)
        self.fusion_2 = FeatureFusionBlock_v2(64 * 2, up=True, attention=attention)
        self.fusion_3 = FeatureFusionBlock_v2(128 * 2, up=True, attention=attention)
        
    def forward(self, dec_outputs, hidden_states):
        hidden_states_new = [] 
        disp = {}

        # scale 3
        cgru_input = dec_outputs[('disp', 3)]
        hidden_new_3 = self.cgru_3(cgru_input, hidden_states[3])
        fusion_input_1 = torch.cat([dec_outputs[('disp', 3)], dec_outputs[('disp', 3)]], 1)
        fusion_input_2 = (hidden_states[3] + hidden_new_3) / 2
        out, up_out = self.fusion_3(fusion_input_1, fusion_input_2) 
        disp[('disp', 3)] = out

        
        # scale 2
        cgru_input = torch.cat([dec_outputs[('disp', 2)], up_out], 1) 
        hidden_new_2 = self.cgru_2(cgru_input, hidden_states[2])
        fusion_input_1 = cgru_input
        fusion_input_2 = (hidden_states[2] + hidden_new_2) / 2
        out, up_out = self.fusion_2(fusion_input_1, fusion_input_2) 
        disp[('disp', 2)] = out
        
        
        # scale 1
        cgru_input =  torch.cat([dec_outputs[('disp', 1)], up_out], 1) 
        hidden_new_1 = self.cgru_1(cgru_input, hidden_states[1])
        fusion_input_1 = cgru_input
        fusion_input_2 = (hidden_states[1] + hidden_new_1) / 2
        out, up_out = self.fusion_1(fusion_input_1, fusion_input_2)
        disp[('disp', 1)] = out
        

        # scale 0
        cgru_input = torch.cat([dec_outputs[('disp', 0)], up_out], 1)
        hidden_new_0 = self.cgru_0(cgru_input, hidden_states[0])
        fusion_input_1 = cgru_input
        fusion_input_2 = (hidden_states[0] + hidden_new_0) / 2
        out = self.fusion_0(fusion_input_1, fusion_input_2)
        disp[('disp', 0)] = out
        
        hidden_states_new.append(hidden_new_0)
        hidden_states_new.append(hidden_new_1)
        hidden_states_new.append(hidden_new_2)
        hidden_states_new.append(hidden_new_3)

        return hidden_states_new, disp 
    
    
    
    
class AttentionConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, groups=1, bias=False):
        super(AttentionConv, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = 3 #kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups

        assert self.out_channels % self.groups == 0, \
        "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"

        self.rel_h = nn.Parameter(torch.randn(out_channels // 2, 1, 1, self.kernel_size, 1), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn(out_channels // 2, 1, 1, 1, self.kernel_size), requires_grad=True)

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
        
        k_out = k_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)
        v_out = v_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)

        q_out = q_out.view(batch, self.groups, self.out_channels // self.groups, height, width, 1)

        out = q_out * k_out
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
    
class FeatureFusionBlock(nn.Module):
    """Feature fusion block."""

    def __init__(self, features, up=True, attention=True):
        """Init.
        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()

        self.up = up
        
        if attention:
            self.resConfUnit1 = ResidualAttentionUnit(features)
            self.resConfUnit2 = ResidualAttentionUnit(features)
            self.resConfUnit3 = ResidualAttentionUnit(features)
        else:
            self.resConfUnit1 = ResidualConvUnit(features)
            self.resConfUnit2 = ResidualConvUnit(features)
            self.resConfUnit3 = ResidualConvUnit(features)
            
        self.conv3x3 = Conv3x3(2, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input_1, input_2):
        """Forward pass.
        Returns:
            tensor: output
        """
        output = self.resConfUnit1(input_1)
        output += self.resConfUnit2(input_2)
        
        output = self.sigmoid(self.conv3x3(self.resConfUnit3(output)))

        if self.up:
            output_up = nn.functional.interpolate(
                output, scale_factor=2, mode="bilinear", align_corners=True
            )
            return output, output_up
        else:
            return output

    
class FeatureFusionBlock_v2(nn.Module):
    """Feature fusion block."""

    def __init__(self, features, up=True, attention=True):
        """Init.
        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock_v2, self).__init__()

        self.up = up
        
        if attention:
            self.resConfUnit1 = ResidualAttentionUnit(features)
            self.resConfUnit2 = ResidualAttentionUnit(features)
        else:
            self.resConfUnit1 = ResidualConvUnit(features)
            self.resConfUnit2 = ResidualConvUnit(features)
            
        self.conv3x3 = Conv3x3(features, 1)
        self.sigmoid = nn.Sigmoid()
        self.upscale = UpscalePS(2)
        
    def forward(self, input_1, input_2):
        """Forward pass.
        Returns:
            tensor: output
        """
        pre_output = self.resConfUnit1(input_1)
        pre_output += input_2
        
        output = self.sigmoid(self.conv3x3(self.resConfUnit2(pre_output)))

        if self.up:
            output_up = self.upscale(pre_output)
#             output_up = nn.functional.interpolate(
#                 pre_output, scale_factor=2, mode="bilinear", align_corners=True
#             )
            return output, output_up
        else:
            return output


        
class UpscalePS(nn.Module):
    def __init__(self, scale): # input_ch, output_ch,
        super().__init__()
#         self.conv = nn.Conv2d(
#             input_ch, input_ch, kernel_size=3, stride=1, padding=1, bias=True)
        self.ps = nn.PixelShuffle(scale)
    
    def forward(self, x):
        return self.ps(torch.tanh(x))
#         return self.ps(torch.tanh(self.conv(x)))

    
        
class ConvGRUBlocks_v3(nn.Module):
    """GRU based on pre disp
    """
    def __init__(self, kernel_size, bias, device):
        super(ConvGRUBlocks_v3, self).__init__()
        
        # scale 0
        dim_dict_0 = {
            'input_dim':16,
            'width': 640,
            'height': 192,
            'hidden_dim_1':16,
        }
        
        self.cgru_0 = ConvGRUModel_v1(dim_dict_0, kernel_size, bias, device)
        
        # scale 1
        dim_dict_1 = {
            'input_dim':32,
            'width': 320,
            'height': 96,
            'hidden_dim_1':32,
        }
        
        self.cgru_1 = ConvGRUModel_v1(dim_dict_1, kernel_size, bias, device)

        # scale 2
        dim_dict_2 = {
            'input_dim':64,
            'width': 160,
            'height': 48,
            'hidden_dim_1':64,
        }
        
        self.cgru_2 = ConvGRUModel_v1(dim_dict_2, kernel_size, bias, device)
        
        # scale 3
        dim_dict_3 = {
            'input_dim':128, # first level has no prev upscaled depth 
            'width': 80,
            'height': 24,
            'hidden_dim_1':128,
        }
        
        self.cgru_3 = ConvGRUModel_v1(dim_dict_3, kernel_size, bias, device)  
        
        # head layers
        self.conv3x3_0 = Conv3x3(16, 1)
        self.conv3x3_1 = Conv3x3(32, 1)
        self.conv3x3_2 = Conv3x3(64, 1)
        self.conv3x3_3 = Conv3x3(128, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, dec_outputs, hidden_states):
        hidden_states_new = []
        disp = {}
        
        hidden_states_new.append(self.cgru_0(dec_outputs[('disp', 0)], hidden_states[0]))
        hidden_states_new.append(self.cgru_1(dec_outputs[('disp', 1)], hidden_states[1]))
        hidden_states_new.append(self.cgru_2(dec_outputs[('disp', 2)], hidden_states[2]))
        hidden_states_new.append(self.cgru_3(dec_outputs[('disp', 3)], hidden_states[3]))
        
        x = hidden_states_new[0]
        disp[('disp', 0)] = self.sigmoid(self.conv3x3_0(x))
        
        x = hidden_states_new[1]
        disp[('disp', 1)] = self.sigmoid(self.conv3x3_1(x))
        
        x = hidden_states_new[2]
        disp[('disp', 2)] = self.sigmoid(self.conv3x3_2(x))
        
        x = hidden_states_new[3]
        disp[('disp', 3)] = self.sigmoid(self.conv3x3_3(x))
        
        return hidden_states_new, disp  
     
        
        
class ConvGRUBlocks_v4(nn.Module):
    """GRU based on pre disparity. 
    Head must follow this block because we need to fuse Ht
    """
    def __init__(self, kernel_size, bias, device, fuse=True):
        super(ConvGRUBlocks_v4, self).__init__()
        
        self.fuse = fuse
        
        # scale 0
        dim_dict_0 = {
            'input_dim':16,
            'width': 640,
            'height': 192,
            'hidden_dim_1':16,
        }
        
        self.cgru_0 = ConvGRUModel_v1(dim_dict_0, kernel_size, bias, device)
        
        # scale 1
        dim_dict_1 = {
            'input_dim':32,
            'width': 320,
            'height': 96,
            'hidden_dim_1':32,
        }
        
        self.cgru_1 = ConvGRUModel_v1(dim_dict_1, kernel_size, bias, device)

        # scale 2
        dim_dict_2 = {
            'input_dim':64,
            'width': 160,
            'height': 48,
            'hidden_dim_1':64,
        }
        
        self.cgru_2 = ConvGRUModel_v1(dim_dict_2, kernel_size, bias, device)
        
        # scale 3
        dim_dict_3 = {
            'input_dim':128, # first level has no prev upscaled depth 
            'width': 80,
            'height': 24,
            'hidden_dim_1':128,
        }
        
        self.cgru_3 = ConvGRUModel_v1(dim_dict_3, kernel_size, bias, device)  
        
        
    def forward(self, dec_outputs, hidden_states):
        hidden_states_new = []
        
        hidden_states_new.append(self.cgru_0(dec_outputs[('disp', 0)], hidden_states[0]))
        hidden_states_new.append(self.cgru_1(dec_outputs[('disp', 1)], hidden_states[1]))
        hidden_states_new.append(self.cgru_2(dec_outputs[('disp', 2)], hidden_states[2]))
        hidden_states_new.append(self.cgru_3(dec_outputs[('disp', 3)], hidden_states[3]))
        
        return hidden_states_new 
        
        
class Head_v4(nn.Module):
    def __init__(self):
        super(Head_v4, self).__init__()

        # head layers
        self.conv3x3_0 = Conv3x3(16, 1)
        self.conv3x3_1 = Conv3x3(32, 1)
        self.conv3x3_2 = Conv3x3(64, 1)
        self.conv3x3_3 = Conv3x3(128, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, outputs, zero_scale_only=False):
        disp = {}
        
        if zero_scale_only:
            disp[('disp', 0)] = self.sigmoid(self.conv3x3_0(outputs[('disp', 0)]))
        else:
            disp[('disp', 0)] = self.sigmoid(self.conv3x3_0(outputs[('disp', 0)]))
            disp[('disp', 1)] = self.sigmoid(self.conv3x3_1(outputs[('disp', 1)]))
            disp[('disp', 2)] = self.sigmoid(self.conv3x3_2(outputs[('disp', 2)]))
            disp[('disp', 3)] = self.sigmoid(self.conv3x3_3(outputs[('disp', 3)]))
        
        return disp        

    
class ConvGRUBlocks_v5(nn.Module):
    """GRU inside skip connections
    """
    def __init__(self, kernel_size, bias, device, fuse=True):
        super(ConvGRUBlocks_v5, self).__init__()
        
        self.fuse = fuse
        
        # scale 0
        dim_dict_0 = {
            'input_dim':64,
            'width': 320,
            'height': 96,
            'hidden_dim_1':64,
        }
        
        self.cgru_0 = ConvGRUModel_v1(dim_dict_0, kernel_size, bias, device)
        
        # scale 1
        dim_dict_1 = {
            'input_dim':64,
            'width': 160,
            'height': 48,
            'hidden_dim_1':64,
        }
        
        self.cgru_1 = ConvGRUModel_v1(dim_dict_1, kernel_size, bias, device)

        # scale 2
        dim_dict_2 = {
            'input_dim':128,
            'width': 80,
            'height': 24,
            'hidden_dim_1':128,
        }
        
        self.cgru_2 = ConvGRUModel_v1(dim_dict_2, kernel_size, bias, device)
        
        # scale 3
        dim_dict_3 = {
            'input_dim':256, # first level has no prev upscaled depth 
            'width': 40,
            'height': 12,
            'hidden_dim_1':256,
        }
        
        self.cgru_3 = ConvGRUModel_v1(dim_dict_3, kernel_size, bias, device)

        # scale 4
        dim_dict_4 = {
            'input_dim':512, # first level has no prev upscaled depth 
            'width': 20,
            'height': 6,
            'hidden_dim_1':512,
        }
        
        self.cgru_4 = ConvGRUModel_v1(dim_dict_4, kernel_size, bias, device)  
        
        
    def forward(self, encoder_features, hidden_states):
        hidden_states_new = []
        
        hidden_states_new.append(self.cgru_0(encoder_features[0], hidden_states[0]))
        hidden_states_new.append(self.cgru_1(encoder_features[1], hidden_states[1]))
        hidden_states_new.append(self.cgru_2(encoder_features[2], hidden_states[2]))
        hidden_states_new.append(self.cgru_3(encoder_features[3], hidden_states[3]))
        hidden_states_new.append(self.cgru_4(encoder_features[4], hidden_states[4]))
        
        return hidden_states_new 

    
    
class ConvGRUBlocks_v7(nn.Module):
    """GRU based on pre disparity. 
    Head must follow this block because we need to fuse Ht
    """
    def __init__(self, kernel_size, bias, device, fuse=True):
        super(ConvGRUBlocks_v7, self).__init__()
        
        self.fuse = fuse
        
        # scale 0
        dim_dict_0 = {
            'input_dim':16 * 2,
            'width': 640,
            'height': 192,
            'hidden_dim_1':16 * 2,
        }
        
        self.cgru_0 = ConvGRUModel_v1(dim_dict_0, kernel_size, bias, device)
        
        # scale 1
        dim_dict_1 = {
            'input_dim':32 * 2,
            'width': 320,
            'height': 96,
            'hidden_dim_1':32 * 2,
        }
        
        self.cgru_1 = ConvGRUModel_v1(dim_dict_1, kernel_size, bias, device)
        self.ps_1 = nn.PixelShuffle(2)

        # scale 2
        dim_dict_2 = {
            'input_dim':64 * 2,
            'width': 160,
            'height': 48,
            'hidden_dim_1':64 * 2,
        }
        
        self.cgru_2 = ConvGRUModel_v1(dim_dict_2, kernel_size, bias, device)
        self.ps_2 = nn.PixelShuffle(2)
        
        # scale 3
        dim_dict_3 = {
            'input_dim':128, # first level has no prev upscaled depth 
            'width': 80,
            'height': 24,
            'hidden_dim_1':128 * 2, # therefore hidden will be 2 times larger than input
        }
        self.cgru_3 = ConvGRUModel_v1(dim_dict_3, kernel_size, bias, device) 
        self.ps_3 = nn.PixelShuffle(2)
        
    def forward(self, dec_outputs, hidden_states):
        hidden_states_new = []
        
        out_3 = self.cgru_3(dec_outputs[('disp', 3)], hidden_states[3])
        up_out_3 = self.ps_3(out_3)

        in_2 = torch.cat([dec_outputs[('disp', 2)], up_out_3], 1)
        out_2 = self.cgru_2(in_2, hidden_states[2])
        up_out_2 = self.ps_2(out_2)
 
        in_1 = torch.cat([dec_outputs[('disp', 1)], up_out_2], 1)
        out_1 = self.cgru_1(in_1, hidden_states[1])
        up_out_1 = self.ps_1(out_1)

        in_0 = torch.cat([dec_outputs[('disp', 0)], up_out_1], 1)
        out_0 = self.cgru_0(in_0, hidden_states[0])
        
        hidden_states_new.append(out_0)
        hidden_states_new.append(out_1)
        hidden_states_new.append(out_2)
        hidden_states_new.append(out_3)
        
        return hidden_states_new 

    
class Head_v7(nn.Module):
    def __init__(self):
        super(Head_v7, self).__init__()

        # head layers
        self.conv3x3_0 = Conv3x3(16 * 2, 1)
        self.conv3x3_1 = Conv3x3(32 * 2, 1)
        self.conv3x3_2 = Conv3x3(64 * 2, 1)
        self.conv3x3_3 = Conv3x3(128 * 2, 1)
        self.sigmoid = nn.Sigmoid() 
        
    def forward(self, outputs, zero_scale_only=False):
        disp = {}
        
        if zero_scale_only:
            disp[('disp', 0)] = self.sigmoid(self.conv3x3_0(outputs[0]))
        else:
            disp[('disp', 0)] = self.sigmoid(self.conv3x3_0(outputs[0]))
            disp[('disp', 1)] = self.sigmoid(self.conv3x3_1(outputs[1]))
            disp[('disp', 2)] = self.sigmoid(self.conv3x3_2(outputs[2]))
            disp[('disp', 3)] = self.sigmoid(self.conv3x3_3(outputs[3]))
        
        return disp


    
# class AttentionConv(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, groups=1, bias=False):
#         super(AttentionConv, self).__init__()
#         self.out_channels = out_channels
#         self.kernel_size = 3 #kernel_size
#         self.stride = stride
#         self.padding = padding
#         self.groups = groups

#         assert self.out_channels % self.groups == 0, \
#         "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"

#         self.rel_h = nn.Parameter(torch.randn(out_channels // 2, 1, 1, self.kernel_size, 1), requires_grad=True)
#         self.rel_w = nn.Parameter(torch.randn(out_channels // 2, 1, 1, 1, self.kernel_size), requires_grad=True)

#         self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
#         self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
#         self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

#         self.reset_parameters()

#     def forward(self, x):
#         batch, channels, height, width = x.size()

#         padded_x = F.pad(x, [self.padding, self.padding, self.padding, self.padding])
#         q_out = self.query_conv(x)
#         k_out = self.key_conv(padded_x)
#         v_out = self.value_conv(padded_x)

#         k_out = k_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
#         v_out = v_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
#         k_out_h, k_out_w = k_out.split(self.out_channels // 2, dim=1)
#         k_out = torch.cat((k_out_h + self.rel_h, k_out_w + self.rel_w), dim=1)
        
#         k_out = k_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)
#         v_out = v_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)

#         q_out = q_out.view(batch, self.groups, self.out_channels // self.groups, height, width, 1)

#         out = q_out * k_out
#         out = F.softmax(out, dim=-1)
#         out = torch.einsum('bnchwk,bnchwk -> bnchw', out, v_out).view(batch, -1, height, width)

#         return out

#     def reset_parameters(self):
#         init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='relu')
#         init.kaiming_normal_(self.value_conv.weight, mode='fan_out', nonlinearity='relu')
#         init.kaiming_normal_(self.query_conv.weight, mode='fan_out', nonlinearity='relu')

#         init.normal_(self.rel_h, 0, 1)
#         init.normal_(self.rel_w, 0, 1)
        
        
        
        
        
        
        
        
        
        
        
# class ConvGRUCellAttention(nn.Module):
#     def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias):
#         super(ConvGRUCellAttention, self).__init__()

#         self.height, self.width = input_size
#         self.padding = kernel_size[0] // 2, kernel_size[1] // 2
#         self.hidden_dim = hidden_dim
#         self.bias = bias
#         self.conv_gates = AttentionConv(in_channels=input_dim + hidden_dim,
#                                     out_channels=2*self.hidden_dim,  # for update_gate,reset_gate respectively
#                                     kernel_size=kernel_size,
#                                     bias=self.bias)

#         self.conv_can = nn.Conv2d(in_channels=input_dim+hidden_dim,
#                               out_channels=self.hidden_dim, # for candidate neural memory
#                               kernel_size=kernel_size,
#                               padding=self.padding,
#                               bias=self.bias)

#     def init_hidden(self, batch_size):
#         return torch.zeros(batch_size, self.hidden_dim, self.height, self.width)

#     def forward(self, input_tensor, h_cur):
#         combined = torch.cat([input_tensor, h_cur], dim=1)
#         combined_conv = self.conv_gates(combined)

#         gamma, beta = torch.split(combined_conv, self.hidden_dim, dim=1)
#         reset_gate = torch.sigmoid(gamma)
#         update_gate = torch.sigmoid(beta)

#         combined = torch.cat([input_tensor, reset_gate*h_cur], dim=1)
#         cc_cnm = self.conv_can(combined)
#         #cnm = torch.tanh(cc_cnm)
#         cnm = F.elu(cc_cnm)
#         h_next = (1 - update_gate) * h_cur + update_gate * cnm
        
#         return h_next
    
# class ConvGRUModel_v1_Attention(nn.Module):
#     def __init__(self, dim_dict, kernel_size, bias, device):
#         super(ConvGRUModel_v1_Attention, self).__init__()

#         self.cgru_1 = ConvGRUCellAttention((dim_dict['height'], dim_dict['width']), dim_dict['input_dim'], dim_dict['hidden_dim_1'], kernel_size, bias)
#         self.h0_layer1 = self.init_hidden(dim_dict['hidden_dim_1'], dim_dict['height'], dim_dict['width'], device)
#         self.height = dim_dict['height']
#         self.width = dim_dict['width']
#         self.hidden_dim = dim_dict['hidden_dim_1']

#     def init_hidden(self, hidden_dim, height, width, device):
#         return nn.Parameter(torch.zeros(1, hidden_dim, height, width, device=device), requires_grad=True)

#     def forward(self, x, hidden_state_1):
#         return self.cgru_1(x, hidden_state_1)   
    
    
# class ConvGRUBlocks_v4_Attention(nn.Module):
#     def __init__(self, kernel_size, bias, device, fuse=True):
#         super(ConvGRUBlocks_v4_Attention, self).__init__()
        
#         self.fuse = fuse
        
#         # scale 0
#         dim_dict_0 = {
#             'input_dim':16,
#             'width': 640,
#             'height': 192,
#             'hidden_dim_1':16,
#         }
        
#         self.cgru_0 = ConvGRUModel_v1_Attention(dim_dict_0, kernel_size, bias, device)
        
#         # scale 1
#         dim_dict_1 = {
#             'input_dim':32,
#             'width': 320,
#             'height': 96,
#             'hidden_dim_1':32,
#         }
        
#         self.cgru_1 = ConvGRUModel_v1_Attention(dim_dict_1, kernel_size, bias, device)

#         # scale 2
#         dim_dict_2 = {
#             'input_dim':64,
#             'width': 160,
#             'height': 48,
#             'hidden_dim_1':64,
#         }
        
#         self.cgru_2 = ConvGRUModel_v1_Attention(dim_dict_2, kernel_size, bias, device)
        
#         # scale 3
#         dim_dict_3 = {
#             'input_dim':128, # first level has no prev upscaled depth 
#             'width': 80,
#             'height': 24,
#             'hidden_dim_1':128,
#         }
        
#         self.cgru_3 = ConvGRUModel_v1_Attention(dim_dict_3, kernel_size, bias, device)  
        
        
#     def forward(self, dec_outputs, hidden_states):
#         hidden_states_new = []
        
#         hidden_states_new.append(self.cgru_0(dec_outputs[('disp', 0)], hidden_states[0]))
#         hidden_states_new.append(self.cgru_1(dec_outputs[('disp', 1)], hidden_states[1]))
#         hidden_states_new.append(self.cgru_2(dec_outputs[('disp', 2)], hidden_states[2]))
#         hidden_states_new.append(self.cgru_3(dec_outputs[('disp', 3)], hidden_states[3]))
        
#         return hidden_states_new 