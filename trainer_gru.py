# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import json

# import GPUtil

from utils import *
from kitti_utils import *
from layers import *

import datasets
import networks
from IPython import embed

from collections import defaultdict

from gru_utils import count_scene_frames_func, generate_frame_seq_func, get_mask_func, get_context_vectors, get_context_vector


class Trainer:
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []

        self.device = torch.device(f'cuda:{self.opt.main_gpu_id}')

        print('Device:', self.device)

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])
        
        # initialize models
        self.opt.mono_pretrained = True
        self.opt.gru_pre_disp = True
        # self.opt.gru_version = 'v2'
        self.opt.fuse = True
        
        if self.opt.mono_pretrained:
            print('---> Load monodepth2 weights')

        if self.opt.gru_pre_disp:
            print('---> GRU will process pre_disp')
            
        if self.opt.gru_version:
            print('---> Use GRU version', self.opt.gru_version)        
        
        if self.opt.fuse:
            print('---> Fuse', self.opt.fuse)
            
        print('Initialize depth net')
        
        self.models["encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained")
        if self.opt.mono_pretrained:
            path = 'models/mono_640x192/encoder.pth'
            model_dict = self.models["encoder"].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models["encoder"].load_state_dict(model_dict)
        device = torch.device(f'cuda:{self.opt.depth_encoder_gpu_id}')
        self.models["encoder"].to(device)
        self.parameters_to_train += list(self.models["encoder"].parameters())             

        self.models["depth"] = networks.DepthDecoder(
            self.models["encoder"].num_ch_enc, self.opt.scales)
        if self.opt.mono_pretrained:
            path = 'models/mono_640x192/depth.pth'
            model_dict = self.models["depth"].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models["depth"].load_state_dict(model_dict)
        device = torch.device(f'cuda:{self.opt.depth_decoder_gpu_id}')
        self.models["depth"].to(device)
        self.parameters_to_train += list(self.models["depth"].parameters())     
        
        print('Initialize GRU blocks')
        device = torch.device(f'cuda:{self.opt.gru_gpu_id}')
        
       
        if self.opt.gru_version == 'v2':
            self.models["gru"] = networks.ConvGRUBlocks_v2(kernel_size=(3, 3), bias=True, 
                                                           device=torch.device(f'cuda:{self.opt.gru_gpu_id}'),
                                                           attention=True)  
            print('ConvGRUBlocks_v2 with attention')
        elif self.opt.gru_version == 'v2_wo_att':
            self.models["gru"] = networks.ConvGRUBlocks_v2(kernel_size=(3, 3), bias=True, 
                                                           device=torch.device(f'cuda:{self.opt.gru_gpu_id}'),
                                                           attention=False)  
            print('ConvGRUBlocks_v2 w/o attention')
        elif self.opt.gru_version == 'v3':
            self.models["gru"] = networks.ConvGRUBlocks_v3(kernel_size=(3, 3), bias=True, 
                                                           device=torch.device(f'cuda:{self.opt.gru_gpu_id}'))
        elif self.opt.gru_version == 'v4':
            self.models["gru"] = networks.ConvGRUBlocks_v4(kernel_size=(3, 3), bias=True, 
                                                           device=torch.device(f'cuda:{self.opt.gru_gpu_id}'))
            self.models["head"] = networks.Head_v4()
            self.models["head"].to(device)
            self.parameters_to_train += list(self.models["head"].parameters()) 
        elif self.opt.gru_version == 'v5':
            self.models["gru"] = networks.ConvGRUBlocks_v5(kernel_size=(3, 3), bias=True, 
                                                           device=torch.device(f'cuda:{self.opt.gru_gpu_id}'))
        elif self.opt.gru_version == 'v6':
            self.models["gru"] = networks.ConvGRUBlocks_v4_Attention(kernel_size=(3, 3), bias=True, 
                                                           device=torch.device(f'cuda:{self.opt.gru_gpu_id}'))
            self.models["head"] = networks.Head_v4()
            self.models["head"].to(device)
            self.parameters_to_train += list(self.models["head"].parameters()) 
        elif self.opt.gru_version == 'v7':
            self.models["gru"] = networks.ConvGRUBlocks_v7(kernel_size=(3, 3), bias=True, 
                                                           device=torch.device(f'cuda:{self.opt.gru_gpu_id}'))
            self.models["head"] = networks.Head_v7()
            self.models["head"].to(device)
        elif self.opt.gru_version == 'v8':
            self.models["gru"] = networks.ConvGRUBlocks_v8(kernel_size=(3, 3), bias=True, 
                                                           device=torch.device(f'cuda:{self.opt.gru_gpu_id}'), 
                                                           attention=False) 
        elif self.opt.gru_version == 'v9':
            self.models["gru"] = networks.ConvGRUBlocks_v9(kernel_size=(3, 3), bias=True, 
                                                           device=torch.device(f'cuda:{self.opt.gru_gpu_id}'), 
                                                           attention=True) 
        elif self.opt.gru_version == 'v10':
            self.models["gru"] = networks.ConvGRUBlocks_v9(kernel_size=(3, 3), bias=True, 
                                                           device=torch.device(f'cuda:{self.opt.gru_gpu_id}'), 
                                                           attention=False) 
        else:
            self.models["gru"] = networks.ConvGRUBlocks(kernel_size=(3, 3), bias=True, 
                                                        device=torch.device(f'cuda:{self.opt.gru_gpu_id}'))
            
        self.models["gru"].to(device)
        self.parameters_to_train += list(self.models["gru"].parameters())

        print('Initialize pose net')

        self.models["pose_encoder"] = networks.ResnetEncoder(
            self.opt.num_layers,
            self.opt.weights_init == "pretrained",
            num_input_images=self.num_pose_frames)
        if self.opt.mono_pretrained:
            path = 'models/mono_640x192/pose_encoder.pth'
            model_dict = self.models["pose_encoder"].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models["pose_encoder"].load_state_dict(model_dict)
        device = torch.device(f'cuda:{self.opt.pose_encoder_gpu_id}')
        self.models["pose_encoder"].to(device)
        self.parameters_to_train += list(self.models["pose_encoder"].parameters()) 

        
        self.models["pose"] = networks.PoseDecoder(
            self.models["pose_encoder"].num_ch_enc,
            num_input_features=1,
            num_frames_to_predict_for=2)
        if self.opt.mono_pretrained:
            path = 'models/mono_640x192/pose.pth'
            model_dict = self.models["pose"].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models["pose"].load_state_dict(model_dict)
        device = torch.device(f'cuda:{self.opt.pose_encoder_gpu_id}')
        self.models["pose"].to(device)
        self.parameters_to_train += list(self.models["pose"].parameters())
        
        
        print('Initialize optimizer')
        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)

        if self.opt.load_weights_folder is not None:
            self.load_model()

        print("Training model named:", self.opt.model_name)
        print("Models and tensorboard events files are saved to:", self.opt.log_dir)
        print("Training is using:", self.device)

        # initialize datasets
    
        train_seq_filepath = "splits/eigen_zhou/train_sequences.txt"
        val_seq_filepath = "splits/eigen_zhou/val_sequences.txt"
        data_path = self.opt.data_path

        with open(train_seq_filepath, 'r') as f:
            train_seq = f.read().splitlines()

        with open(val_seq_filepath, 'r') as f:
            val_seq = f.read().splitlines()
    
        train_n_frames_dict = count_scene_frames_func(train_seq, data_path)
        val_n_frames_dict = count_scene_frames_func(val_seq, data_path)

        n = self.opt.len_sequence
        k = 2 # pose mask

        train_tuples = generate_frame_seq_func(train_n_frames_dict, train_seq, n, k, self.opt.train_n_tuples)
        val_tuples = generate_frame_seq_func(val_n_frames_dict, val_seq, n, k, self.opt.test_n_tuples)

        print('Generated train tuples: {}, val tuples: {}'.format(len(train_tuples), len(val_tuples)))
        
        train_dataset = datasets.KITTIDataset_v1(data_path, 192, 640, n, train_tuples, True)
        val_dataset = datasets.KITTIDataset_v1(data_path, 192, 640, n, val_tuples, False)
        
        self.train_loader = DataLoader(train_dataset, self.opt.batch_size, True, num_workers=self.opt.num_workers,
                                  pin_memory=True, drop_last=True)
        self.val_loader = DataLoader(val_dataset, self.opt.batch_size, False, 
                                num_workers=self.opt.num_workers, pin_memory=True, drop_last=False)
#         self.val_iter = iter(self.val_loader)
        print('Dataloaders created')
        
        # initialize writers
        num_train_samples = len(train_tuples) # * (n + k)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs
        
        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        
#         print('Current GPU mem utilization\n', GPUtil.showUtilization())
        
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size * self.opt.len_sequence, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size * self.opt.len_sequence, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        self.save_opts()

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            
            if (self.epoch + 1) == self.opt.h_s_epoch:
                print("Removed hidden states from optimizer")
                self.models["gru"].cgru_0.h0_layer1.requires_grad = False
                self.models["gru"].cgru_1.h0_layer1.requires_grad = False
                self.models["gru"].cgru_2.h0_layer1.requires_grad = False
                self.models["gru"].cgru_3.h0_layer1.requires_grad = False
                if self.opt.gru_version == 'v5':
                    self.models["gru"].cgru_4.h0_layer1.requires_grad = False
                if self.opt.gru_version == 'v8':
                    self.models["gru"].cgru_0.c0_layer1.requires_grad = False
                    self.models["gru"].cgru_1.c0_layer1.requires_grad = False
                    self.models["gru"].cgru_2.c0_layer1.requires_grad = False
                    self.models["gru"].cgru_3.c0_layer1.requires_grad = False

            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
#         self.model_lr_scheduler.step()

        print("Training")
        self.set_train()

        for batch_idx, inputs in enumerate(self.train_loader):

            before_op_time = time.time()

            outputs, losses = self.process_batch(inputs)

            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()
#             self.model_lr_scheduler.step() 

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 # and self.step < 2000
            late_phase = self.step % 2000 == 0

            if early_phase or late_phase:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)

                # if "depth_gt" in inputs:
                    # self.compute_depth_losses(inputs, outputs, losses)

                # self.log("train", inputs, outputs, losses)
                # self.val()

            self.step += 1

            
    def run_gru_v2(self, inputs):    
        # depth encoder-decoder
        n = self.opt.len_sequence
        
        # we dont take left and right images because theu are for pose estimation
        device = torch.device(f'cuda:{self.opt.depth_encoder_gpu_id}')
        enc_input = torch.cat([inputs[("color", 0, 0, i)] for i in range(n)]).to(device)
        features = self.models["encoder"](enc_input)
        depth_outputs = self.models["depth"](features, False)

        device = torch.device(f'cuda:{self.opt.gru_gpu_id}')
        for i in range(4):
            x = [t.unsqueeze(0) for t in torch.split(
                depth_outputs[('disp', i)], n, dim=0)]
            depth_outputs[('disp', i)] = torch.cat(x, dim=0).to(device)

        # initialize h0
        bs = self.opt.batch_size
        if bs > 1: 
            hidden_states = [self.models["gru"].cgru_0.h0_layer1.repeat(bs, 1, 1, 1),
                             self.models["gru"].cgru_1.h0_layer1.repeat(bs, 1, 1, 1),
                             self.models["gru"].cgru_2.h0_layer1.repeat(bs, 1, 1, 1),
                             self.models["gru"].cgru_3.h0_layer1.repeat(bs, 1, 1, 1),
                            ]
        else:
            hidden_states = [self.models["gru"].cgru_0.h0_layer1,
                             self.models["gru"].cgru_1.h0_layer1,
                             self.models["gru"].cgru_2.h0_layer1,
                             self.models["gru"].cgru_3.h0_layer1,
                            ]
        
        outputs = defaultdict(list)
        # gru
        for i in range(n):
            gru_inputs = {}
            gru_inputs[('disp', 0)] = depth_outputs[('disp', 0)][:, i]
            gru_inputs[('disp', 1)] = depth_outputs[('disp', 1)][:, i]
            gru_inputs[('disp', 2)] = depth_outputs[('disp', 2)][:, i]
            gru_inputs[('disp', 3)] = depth_outputs[('disp', 3)][:, i]

            hidden_states, disp = self.models["gru"](gru_inputs, hidden_states)
            
            for i in range(4):
                outputs[('disp', i)].append(disp[('disp', i)])

        for i in range(4):
            bs, _, h, w = outputs[('disp', i)][0].shape
            outputs[('disp', i)] = torch.stack(outputs[('disp', i)], 1).view(-1, 1, h, w)
        return outputs
  


    def run_gru_v8(self, inputs):    
        # depth encoder-decoder
        n = self.opt.len_sequence
        
        # we dont take left and right images because theu are for pose estimation
        device = torch.device(f'cuda:{self.opt.depth_encoder_gpu_id}')
        enc_input = torch.cat([inputs[("color", 0, 0, i)] for i in range(n)]).to(device)
        features = self.models["encoder"](enc_input)
        depth_outputs = self.models["depth"](features, True)

        device = torch.device(f'cuda:{self.opt.gru_gpu_id}')
        for i in range(4):
            x = [t.unsqueeze(0) for t in torch.split(
                depth_outputs[('disp', i)], n, dim=0)]
            depth_outputs[('disp', i)] = torch.cat(x, dim=0).to(device)

        # initialize h0
        bs = self.opt.batch_size
        if bs > 1: 
            hidden_states = [(self.models["gru"].cgru_0.h0_layer1.repeat(bs, 1, 1, 1),
                              self.models["gru"].cgru_0.c0_layer1.repeat(bs, 1, 1, 1)),
                             (self.models["gru"].cgru_1.h0_layer1.repeat(bs, 1, 1, 1),
                              self.models["gru"].cgru_1.c0_layer1.repeat(bs, 1, 1, 1)),
                             (self.models["gru"].cgru_2.h0_layer1.repeat(bs, 1, 1, 1),
                              self.models["gru"].cgru_2.c0_layer1.repeat(bs, 1, 1, 1)),
                             (self.models["gru"].cgru_3.h0_layer1.repeat(bs, 1, 1, 1),
                              self.models["gru"].cgru_3.c0_layer1.repeat(bs, 1, 1, 1)),
                            ]
        else:
            hidden_states = [(self.models["gru"].cgru_0.h0_layer1, self.models["gru"].cgru_0.c0_layer1),
                             (self.models["gru"].cgru_1.h0_layer1, self.models["gru"].cgru_1.c0_layer1),
                             (self.models["gru"].cgru_2.h0_layer1, self.models["gru"].cgru_2.c0_layer1),
                             (self.models["gru"].cgru_3.h0_layer1, self.models["gru"].cgru_3.c0_layer1),
                            ]
        
        outputs = defaultdict(list)
        # gru
        for i in range(n):
#             print(i)
            gru_inputs = {}
#             for k, v in depth_outputs.items():
#                 print(k, v.shape)
            gru_inputs[('disp', 0)] = depth_outputs[('disp', 0)][:, i]
            gru_inputs[('disp', 1)] = depth_outputs[('disp', 1)][:, i]
            gru_inputs[('disp', 2)] = depth_outputs[('disp', 2)][:, i]
            gru_inputs[('disp', 3)] = depth_outputs[('disp', 3)][:, i]
            
#             for k, v in gru_inputs.items():
#                 print(k, v.shape)

            hidden_states, disp = self.models["gru"](gru_inputs, hidden_states)
            
            for i in range(4):
                outputs[('disp', i)].append(disp[('disp', i)])

        for i in range(4):
            bs, _, h, w = outputs[('disp', i)][0].shape
            outputs[('disp', i)] = torch.stack(outputs[('disp', i)], 1).view(-1, 1, h, w)
        return outputs
    
    
    
       
            
    def run_gru_v3(self, inputs):    
        # depth encoder-decoder
        n = self.opt.len_sequence
        
        # we dont take left and right images because theu are for pose estimation
        device = torch.device(f'cuda:{self.opt.depth_encoder_gpu_id}')
        enc_input = torch.cat([inputs[("color", 0, 0, i)] for i in range(n)]).to(device)
        features = self.models["encoder"](enc_input)
        depth_outputs = self.models["depth"](features, self.opt.gru_pre_disp)

        device = torch.device(f'cuda:{self.opt.gru_gpu_id}')
        for i in range(4):
            x = [t.unsqueeze(0) for t in torch.split(
                depth_outputs[('disp', i)], n, dim=0)]
            depth_outputs[('disp', i)] = torch.cat(x, dim=0).to(device)

        # initialize h0
        bs = self.opt.batch_size
        if bs > 1: 
            hidden_states = [self.models["gru"].cgru_0.h0_layer1.repeat(bs, 1, 1, 1),
                             self.models["gru"].cgru_1.h0_layer1.repeat(bs, 1, 1, 1),
                             self.models["gru"].cgru_2.h0_layer1.repeat(bs, 1, 1, 1),
                             self.models["gru"].cgru_3.h0_layer1.repeat(bs, 1, 1, 1),
                            ]
        else:
            hidden_states = [self.models["gru"].cgru_0.h0_layer1,
                             self.models["gru"].cgru_1.h0_layer1,
                             self.models["gru"].cgru_2.h0_layer1,
                             self.models["gru"].cgru_3.h0_layer1,
                            ]
            
        outputs = defaultdict(list)
            
        # gru
        for i in range(n):
            gru_inputs = {}
            gru_inputs[('disp', 0)] = depth_outputs[('disp', 0)][:, i]
            gru_inputs[('disp', 1)] = depth_outputs[('disp', 1)][:, i]
            gru_inputs[('disp', 2)] = depth_outputs[('disp', 2)][:, i]
            gru_inputs[('disp', 3)] = depth_outputs[('disp', 3)][:, i]

            hidden_states, disp = self.models["gru"](gru_inputs, hidden_states)

            outputs[('disp', 0)].append(disp[('disp', 0)])
            outputs[('disp', 1)].append(disp[('disp', 1)])
            outputs[('disp', 2)].append(disp[('disp', 2)])
            outputs[('disp', 3)].append(disp[('disp', 3)])

        for i in range(4):
            bs, _, h, w = outputs[('disp', i)][0].shape
            outputs[('disp', i)] = torch.stack(outputs[('disp', i)], 1).view(-1, 1, h, w)
        return outputs
    
            
    def run_gru_v4(self, inputs):    
        # depth encoder-decoder
        n = self.opt.len_sequence
        
        # we dont take left and right images because theu are for pose estimation
        device = torch.device(f'cuda:{self.opt.depth_encoder_gpu_id}')
        enc_input = torch.cat([inputs[("color", 0, 0, i)] for i in range(n)]).to(device)
        features = self.models["encoder"](enc_input)
        depth_outputs = self.models["depth"](features, self.opt.gru_pre_disp)

        device = torch.device(f'cuda:{self.opt.gru_gpu_id}')
        for i in range(4):
            x = [t.unsqueeze(0) for t in torch.split(
                depth_outputs[('disp', i)], n, dim=0)]
            depth_outputs[('disp', i)] = torch.cat(x, dim=0).to(device)

        # initialize h0
        bs = self.opt.batch_size
        if bs > 1: 
            hidden_states = [self.models["gru"].cgru_0.h0_layer1.repeat(bs, 1, 1, 1),
                             self.models["gru"].cgru_1.h0_layer1.repeat(bs, 1, 1, 1),
                             self.models["gru"].cgru_2.h0_layer1.repeat(bs, 1, 1, 1),
                             self.models["gru"].cgru_3.h0_layer1.repeat(bs, 1, 1, 1),
                            ]
        else:
            hidden_states = [self.models["gru"].cgru_0.h0_layer1,
                             self.models["gru"].cgru_1.h0_layer1,
                             self.models["gru"].cgru_2.h0_layer1,
                             self.models["gru"].cgru_3.h0_layer1,
                            ]
            
        outputs = defaultdict(list)
            
        # gru + fusion / concat
        hidden_outputs = {0: [], 1:[], 2:[], 3:[]}
        hidden_outputs_aggr = {}

        # append init hidden (history for 0 frame)
        for i in range(4):
            hidden_outputs[i].append(hidden_states[i])

        for i in range(n):
            gru_inputs = {}
            gru_inputs[('disp', 0)] = depth_outputs[('disp', 0)][:, i]
            gru_inputs[('disp', 1)] = depth_outputs[('disp', 1)][:, i]
            gru_inputs[('disp', 2)] = depth_outputs[('disp', 2)][:, i]
            gru_inputs[('disp', 3)] = depth_outputs[('disp', 3)][:, i]

            hidden_states = self.models["gru"](gru_inputs, hidden_states)
            for i in range(4):
                hidden_outputs[i].append(hidden_states[i])

        for i in range(4):       
            hidden_outputs[i] = torch.stack(hidden_outputs[i], 1)
            if self.opt.fuse:
                hidden_outputs_aggr[i] = (hidden_outputs[i][:,1:] + hidden_outputs[i][:,:-1]) / 2
            else:
                hidden_outputs_aggr[i] = torch.cat((hidden_outputs[i][:,1:,:], hidden_outputs[i][:,:-1,:]), 2)

        for i in range(n):
            for j in range(4):
                outputs[('disp', j)].append(
                    depth_outputs[('disp', j)][:, i] + hidden_outputs_aggr[j][:,i])

        for i in range(4):
            outputs[('disp', i)] = torch.cat(outputs[('disp', i)], 0)

        outputs = self.models["head"](outputs, zero_scale_only=False)
        
#         for k, v in outputs.items():
#             print(k, v.shape)
            
        return outputs

        
    def run_gru_v5(self, inputs):    
        # WORKS for batch_size = 1 ONLY !!!
        # depth encoder-decoder
        n = self.opt.len_sequence
        
        # we dont take left and right images because theu are for pose estimation
        device = torch.device(f'cuda:{self.opt.depth_encoder_gpu_id}')
        enc_input = torch.cat([inputs[("color", 0, 0, i)] for i in range(n)]).to(device)
        features = self.models["encoder"](enc_input)
        
        # model with GRU fusion between inside skip connections

        # init hidden states
        bs = self.opt.batch_size
        if bs > 1: 
            hidden_states = [self.models["gru"].cgru_0.h0_layer1.repeat(bs, 1, 1, 1),
                             self.models["gru"].cgru_1.h0_layer1.repeat(bs, 1, 1, 1),
                             self.models["gru"].cgru_2.h0_layer1.repeat(bs, 1, 1, 1),
                             self.models["gru"].cgru_3.h0_layer1.repeat(bs, 1, 1, 1),
                             self.models["gru"].cgru_4.h0_layer1.repeat(bs, 1, 1, 1)]
        else:
            hidden_states = [self.models["gru"].cgru_0.h0_layer1,
                             self.models["gru"].cgru_1.h0_layer1,
                             self.models["gru"].cgru_2.h0_layer1,
                             self.models["gru"].cgru_3.h0_layer1, 
                             self.models["gru"].cgru_4.h0_layer1]

        hidden_outputs = defaultdict(list)

        # remember init hidden states 
        for i in range(5):
            hidden_outputs[i].append(hidden_states[i])

        # sequence loop
        for i in range(n):
            single_image_features = [features[j][i].unsqueeze(0) for j in range(5)]
            hidden_states = self.models["gru"](single_image_features, hidden_states)

            # remember all hidden states
            for i in range(5):
                hidden_outputs[i].append(hidden_states[i])
                
        # aggregation of hidden states (fusion / cat)
        for scale in range(5):    
            hidden_outputs[scale] = torch.cat(hidden_outputs[scale], 0)
            features[scale] = features[scale] + (hidden_outputs[scale][1:] + hidden_outputs[scale][:-1]) / 2

        # decoder
        outputs = self.models["depth"](features, False)  
        return outputs

    

    def run_gru_v7(self, inputs): 
        # gru + fusion
        # for batch size = 1 only
        n = self.opt.len_sequence
        
        # depth encoder-decoder
        device = torch.device(f'cuda:{self.opt.depth_encoder_gpu_id}')
        enc_input = torch.cat([inputs[("color", 0, 0, i)] for i in range(n)]).to(device)
        features = self.models["encoder"](enc_input)
        depth_outputs = self.models["depth"](features, True)
        
        device = torch.device(f'cuda:{self.opt.gru_gpu_id}')
        for i in range(4):
            x = [t.unsqueeze(0) for t in torch.split(
                depth_outputs[('disp', i)], n, dim=0)]
            depth_outputs[('disp', i)] = torch.cat(x, dim=0).to(device)
        
                
        # initialize h0
        bs = self.opt.batch_size
        if bs > 1: 
            hidden_curr = [self.models["gru"].cgru_0.h0_layer1.repeat(bs, 1, 1, 1),
                             self.models["gru"].cgru_1.h0_layer1.repeat(bs, 1, 1, 1),
                             self.models["gru"].cgru_2.h0_layer1.repeat(bs, 1, 1, 1),
                             self.models["gru"].cgru_3.h0_layer1.repeat(bs, 1, 1, 1),
                            ]
        else:
            hidden_curr = [self.models["gru"].cgru_0.h0_layer1,
                             self.models["gru"].cgru_1.h0_layer1,
                             self.models["gru"].cgru_2.h0_layer1,
                             self.models["gru"].cgru_3.h0_layer1,
                            ]
        hidden_prev = None
            
        outputs = defaultdict(list)
            
        # gru + fusion
        hidden_outputs = {0: [], 1:[], 2:[], 3:[]}

        for i in range(n):
            gru_inputs = {}
            gru_inputs[('disp', 0)] = depth_outputs[('disp', 0)][:, i]
            gru_inputs[('disp', 1)] = depth_outputs[('disp', 1)][:, i]
            gru_inputs[('disp', 2)] = depth_outputs[('disp', 2)][:, i]
            gru_inputs[('disp', 3)] = depth_outputs[('disp', 3)][:, i]
            
            # fusion before gru
            if hidden_prev is not None:
                for i in range(4):
                    hidden_states.append((hidden_curr[i] + hidden_prev[i]) / 2)
            else:
                hidden_states = hidden_curr
            
            hidden_prev = hidden_curr
            hidden_curr = self.models["gru"](gru_inputs, hidden_states)
            
            # save hidden outputs
            for i in range(4):
                hidden_outputs[i].append(hidden_curr[i])
        
        for i in range(4):
            hidden_outputs[i] = torch.cat(hidden_outputs[i], 0)

        hidden_outputs = self.models["head"](hidden_outputs, zero_scale_only=False)
            
        return hidden_outputs

 
    def run_gru_v9_v10(self, inputs):    
        # depth encoder-decoder
        n = self.opt.len_sequence
        
        # we dont take left and right images because theu are for pose estimation
        device = torch.device(f'cuda:{self.opt.depth_encoder_gpu_id}')
        enc_input = torch.cat([inputs[("color", 0, 0, i)] for i in range(n)]).to(device)
        features = self.models["encoder"](enc_input)
        depth_outputs = self.models["depth"](features, True)

        device = torch.device(f'cuda:{self.opt.gru_gpu_id}')
        for i in range(4):
            x = [t.unsqueeze(0) for t in torch.split(
                depth_outputs[('disp', i)], n, dim=0)]
            depth_outputs[('disp', i)] = torch.cat(x, dim=0).to(device)

        # initialize h0
        bs = self.opt.batch_size
        if bs > 1: 
            hidden_states = [self.models["gru"].cgru_0.h0_layer1.repeat(bs, 1, 1, 1),
                             self.models["gru"].cgru_1.h0_layer1.repeat(bs, 1, 1, 1),
                             self.models["gru"].cgru_2.h0_layer1.repeat(bs, 1, 1, 1),
                             self.models["gru"].cgru_3.h0_layer1.repeat(bs, 1, 1, 1),
                            ]
        else:
            hidden_states = [self.models["gru"].cgru_0.h0_layer1,
                             self.models["gru"].cgru_1.h0_layer1,
                             self.models["gru"].cgru_2.h0_layer1,
                             self.models["gru"].cgru_3.h0_layer1,
                            ]
        
        outputs = defaultdict(list)
        # gru
        for i in range(n):
            gru_inputs = {}
            gru_inputs[('disp', 0)] = depth_outputs[('disp', 0)][:, i]
            gru_inputs[('disp', 1)] = depth_outputs[('disp', 1)][:, i]
            gru_inputs[('disp', 2)] = depth_outputs[('disp', 2)][:, i]
            gru_inputs[('disp', 3)] = depth_outputs[('disp', 3)][:, i]

            hidden_states, disp = self.models["gru"](gru_inputs, hidden_states)
            
            for i in range(4):
                outputs[('disp', i)].append(disp[('disp', i)])

        for i in range(4):
            bs, _, h, w = outputs[('disp', i)][0].shape
            outputs[('disp', i)] = torch.stack(outputs[('disp', i)], 1).view(-1, 1, h, w)
        return outputs
    
    
            
    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        
          
        if self.opt.gru_version in ('v2', 'v2_wo_att'):
            outputs = self.run_gru_v2(inputs)
        elif self.opt.gru_version == 'v3':
            outputs = self.run_gru_v3(inputs)
        elif self.opt.gru_version in ['v4', 'v6']:
            outputs = self.run_gru_v4(inputs)
        elif self.opt.gru_version == 'v5':
            outputs = self.run_gru_v5(inputs)
        elif self.opt.gru_version == 'v7':
            outputs = self.run_gru_v7(inputs)
        elif self.opt.gru_version == 'v8':
            outputs = self.run_gru_v8(inputs)
        elif self.opt.gru_version in ('v9', 'v10'):
            outputs = self.run_gru_v9_v10(inputs)
        
        if self.use_pose_net:
            poses = self.predict_poses(inputs, None)

            # everything to default device 
            for key, ipt in poses.items():
                poses[key] = ipt.to(self.device)

            for key, ipt in outputs.items():
                outputs[key] = ipt.to(self.device)
            outputs.update(poses)

        # inputs to default device
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)
                
        # view synthesis
        self.generate_images_pred(inputs, outputs)

        # loss
        losses = self.compute_losses(inputs, outputs)

        return outputs, losses

    def predict_poses(self, inputs, features=None):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        n = self.opt.len_sequence
        
        device = torch.device(f'cuda:{self.opt.pose_encoder_gpu_id}')
        pose_feats = {}
        pose_feats[-1] = torch.cat([inputs[("color", -1, 0, i)] for i in range(n)], 0).to(device)
        pose_feats[1] = torch.cat([inputs[("color", 1, 0, i)] for i in range(n)], 0).to(device)
        pose_feats[0] = torch.cat([inputs[("color", 0, 0, i)] for i in range(n)], 0).to(device)

        for f_i in self.opt.frame_ids[1:]: # [-1, 1]
            # To maintain ordering we always pass frames in temporal order
            if f_i < 0:
                pose_inputs = [pose_feats[f_i], pose_feats[0]]
            else:
                pose_inputs = [pose_feats[0], pose_feats[f_i]]

            pose_inputs = torch.cat(pose_inputs, 1)
            pose_inputs = [self.models["pose_encoder"](pose_inputs)]

            axisangle, translation = self.models["pose"](pose_inputs)
            outputs[("axisangle", 0, f_i)] = axisangle
            outputs[("translation", 0, f_i)] = translation

            # Invert the matrix if the frame id is negative
            outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

        return outputs

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)

            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)

            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
#         for k,v in inputs.items():
#             print(k, v.shape)
            
        n = self.opt.len_sequence
        
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                source_scale = 0

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate([-1, 1]): #self.opt.frame_ids[1:]):

                T = outputs[("cam_T_cam", 0, frame_id)]

                inv_K = torch.cat([inputs[("inv_K", source_scale, i)] for i in range(n)], 0)
                cam_points = self.backproject_depth[source_scale](
                    depth, inv_K)
                K = torch.cat([inputs[("K", source_scale, i)] for i in range(n)], 0)
                pix_coords = self.project_3d[source_scale](
                    cam_points, K, T)

                outputs[("sample", frame_id, scale)] = pix_coords
                
                input_colors = torch.cat([inputs[("color", frame_id, source_scale, i)] for i in range(n)], 0) 

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    input_colors,
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border")

                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        input_colors


    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    
    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0
        n = self.opt.len_sequence

        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            disp = outputs[("disp", scale)]
            color =  torch.cat([inputs[("color", 0, scale, i)] for i in range(n)], 0)
            target = torch.cat([inputs[("color", 0, source_scale, i)] for i in range(n)], 0)
#             color = inputs[("color", 0, scale)]
#             target = inputs[("color", 0, source_scale)]

            for frame_id in [-1, 1]: #self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)

            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in [-1, 1]: #self.opt.frame_ids[1:]:
                    pred = torch.cat([inputs[("color", frame_id, source_scale, i)] for i in range(n)], 0)
                    # pred = inputs[("color", frame_id, source_scale)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    # save both images, and do min all at once below
                    identity_reprojection_loss = identity_reprojection_losses

            
            elif self.opt.predictive_mask:
                # use the predicted mask
                mask = outputs["predictive_mask"]["disp", scale]
                if not self.opt.v1_multiscale:
                    mask = F.interpolate(
                        mask, [self.opt.height, self.opt.width],
                        mode="bilinear", align_corners=False)

                reprojection_losses *= mask

                # add a loss pushing mask to 1 (using nn.BCELoss for stability)
                weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).to(self.device))
                loss += weighting_loss.mean()

            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses

#             print(GPUtil.showUtilization())
            
            if not self.opt.disable_automasking:
                # add random numbers to break ties
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape).to(self.device) * 0.00001

                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss
                

            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)

            if not self.opt.disable_automasking:
                outputs["identity_selection/{}".format(scale)] = (
                    idxs > identity_reprojection_loss.shape[1] - 1).float()

            loss += to_optimise.mean()

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss
        return losses
    
    
#     def compute_losses(self, inputs, outputs):
#         """Compute the reprojection and smoothness losses for a minibatch
#         """
#         losses = {}
#         total_loss = 0
#         n = self.opt.len_sequence

#         for scale in self.opt.scales:
#             loss = 0
#             reprojection_losses = []

#             if self.opt.v1_multiscale:
#                 source_scale = scale
#             else:
#                 source_scale = 0

#             h = self.opt.height // (2 ** scale)
#             w = self.opt.width // (2 ** scale)
            
#             color = inputs[("color", 0, scale)][0]#.reshape(-1, 3, h, w)
#             target = inputs[("color", 0, 0)][0]#.reshape(-1, 3, 192, 640)
#             disp = outputs[("disp", scale)]

#             for frame_id in [-1, 1]: #self.opt.frame_ids[1:]:
#                 pred = outputs[("color", frame_id, scale)]
#                 reprojection_losses.append(self.compute_reprojection_loss(pred, target))

#             reprojection_losses = torch.cat(reprojection_losses, 1)
            
#             if not self.opt.disable_automasking:
#                 identity_reprojection_losses = []
#                 for frame_id in [-1, 1]: #self.opt.frame_ids[1:]:
#                     target_inputs = inputs[("color", frame_id, 0)][0]#.reshape(-1, 3, 192, 640)
#                     pred = target_inputs
#                     identity_reprojection_losses.append(
#                         self.compute_reprojection_loss(pred, target))

#                 identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

#                 if self.opt.avg_reprojection:
#                     identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
#                 else:
#                     # save both images, and do min all at once below
#                     identity_reprojection_loss = identity_reprojection_losses
                    
#             if self.opt.avg_reprojection:
#                 reprojection_loss = reprojection_losses.mean(1, keepdim=True)
#             else:
#                 reprojection_loss = reprojection_losses

# #             print(GPUtil.showUtilization())
            
#             if not self.opt.disable_automasking:
#                 # add random numbers to break ties
#                 identity_reprojection_loss += torch.randn(
#                     identity_reprojection_loss.shape).to(self.device) * 0.00001

#                 combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
#             else:
#                 combined = reprojection_loss

#             if combined.shape[1] == 1:
#                 to_optimise = combined
#             else:
#                 to_optimise, idxs = torch.min(combined, dim=1)

#             if not self.opt.disable_automasking:
#                 outputs["identity_selection/{}".format(scale)] = (
#                     idxs > identity_reprojection_loss.shape[1] - 1).float()

#             loss += to_optimise.mean()

#             mean_disp = disp.mean(2, True).mean(3, True)
#             norm_disp = disp / (mean_disp + 1e-7)
#             smooth_loss = get_smooth_loss(norm_disp, color)

#             loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
#             total_loss += loss
#             losses["loss/{}".format(scale)] = loss

#         total_loss /= self.num_scales
#         losses["loss"] = total_loss
#         return losses

    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"].contiguous().view(-1, 1, 375, 1242)
        mask = depth_gt > 0

        # garg/eigen crop
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            for s in self.opt.scales:
                for frame_id in [0, -1, 1]: #self.opt.frame_ids:
                    writer.add_image(
                        "color_{}_{}/{}".format(frame_id, s, j),
                        inputs[("color", frame_id, s)][j].data, self.step)
                    if s == 0 and frame_id != 0:
                        writer.add_image(
                            "color_pred_{}_{}/{}".format(frame_id, s, j),
                            outputs[("color", frame_id, s)][j].data, self.step)

                writer.add_image(
                    "disp_{}/{}".format(s, j),
                    normalize_image(outputs[("disp", s)][j]), self.step)

                if self.opt.predictive_mask:
                    for f_idx, frame_id in enumerate([-1, 1]):#self.opt.frame_ids[1:]):
                        writer.add_image(
                            "predictive_mask_{}_{}/{}".format(frame_id, s, j),
                            outputs["predictive_mask"][("disp", s)][j, f_idx][None, ...],
                            self.step)

                elif not self.opt.disable_automasking:
                    writer.add_image(
                        "automask_{}/{}".format(s, j),
                        outputs["identity_selection/{}".format(s)][j][None, ...], self.step)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load[::-1]:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()

            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)
            del model_dict

        # loading adam state
#         torch.cuda.set_device(1)
        print(GPUtil.showUtilization())

        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")
