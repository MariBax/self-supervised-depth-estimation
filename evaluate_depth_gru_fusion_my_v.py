from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader
from PIL import Image

from collections import defaultdict

from layers import disp_to_depth
from utils import readlines
from options import MonodepthOptions
import datasets
import networks

import torch.nn.functional as F

import skimage
import skimage.transform
from torchvision import transforms

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)


splits_dir = os.path.join(os.path.dirname(__file__), "splits")

# Models which were trained with stereo supervision were trained with a nominal
# baseline of 0.1 units. The KITTI rig has a baseline of 54cm. Therefore,
# to convert our stereo predictions to real-world scale we multiply our depths by 5.4.
STEREO_SCALE_FACTOR = 5.4


def transform_depth(depth_pred, depth_gt):

    mask = depth_gt > 0
    crop_mask = np.zeros_like(mask)
    crop_mask[153:371, 44:1197] = 1
    mask = mask * crop_mask

    depth_gt = depth_gt[mask]
    depth_pred = depth_pred[mask]

    depth_pred *= np.median(depth_gt) / np.median(depth_pred)

    depth_pred = np.clip(depth_pred, a_min=1e-3, a_max=80)

    return depth_pred, depth_gt


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')
        
        
def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def get_image_path(frame_index, sequence_name):
        
        scene_date, scene_name = sequence_name.split('/')
        f_str = "{:010d}.jpg".format(frame_index)
        
        image_path = os.path.join(
            'kitti_data', 
            scene_date, 
            scene_name, 
            "image_02", 
            "data", 
            f_str)
        
        return image_path

    
def evaluate_v3_single_image(opt, dataloader, encoder, depth_decoder, gru, pre_disp):
    pred_disps = []
    len_dataloader = len(dataloader)
    
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            print(i, 'from', len_dataloader, 'batches')
            input_color = data[("color", 0, 0)].cuda()   

            # encoder-decoder outputs
            depth_outputs = depth_decoder(encoder(input_color), pre_disp)

            # gru
            bs = input_color.shape[0]

            hidden_states = [gru.cgru_0.h0_layer1.repeat(bs, 1, 1, 1),
                             gru.cgru_1.h0_layer1.repeat(bs, 1, 1, 1),
                             gru.cgru_2.h0_layer1.repeat(bs, 1, 1, 1),
                             gru.cgru_3.h0_layer1.repeat(bs, 1, 1, 1),
                            ]

            outputs = {}
            for i in range(4):
                outputs[('disp', i)] = []

            gru_inputs = {}
            gru_inputs[('disp', 0)] = depth_outputs[('disp', 0)]
            gru_inputs[('disp', 1)] = depth_outputs[('disp', 1)]
            gru_inputs[('disp', 2)] = depth_outputs[('disp', 2)]
            gru_inputs[('disp', 3)] = depth_outputs[('disp', 3)]

            hidden_states, disp = gru(gru_inputs, hidden_states)

            outputs[('disp', 0)] = disp[('disp', 0)]
            outputs[('disp', 1)] = disp[('disp', 1)]
            outputs[('disp', 2)] = disp[('disp', 2)]
            outputs[('disp', 3)] = disp[('disp', 3)]

            pred_disp, _ = disp_to_depth(outputs[("disp", 0)], opt.min_depth, opt.max_depth)
            pred_disp = pred_disp.cpu()[:, 0].numpy()

            if opt.post_process:
                N = pred_disp.shape[0] // 2
                pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])

            pred_disps.append(pred_disp)
        pred_disps = np.concatenate(pred_disps) 

    return pred_disps   



def evaluate_v3_seq_prev_images(opt, test_files, encoder, depth_decoder, gru, pre_disp=True):
    n_prev_images = 12
    test_scenes = sorted(list(test_files.keys()))
    pred_disps = []
    interp = Image.ANTIALIAS
    resize = transforms.Resize((192, 640), interpolation=interp)            
        
    with torch.no_grad():
        
        # loop over scenes
        for scene in test_scenes:
            scene_files = sorted(test_files[scene])
            print('Get predictions for scene', scene)
            print('Number of test files in scene', len(scene_files))
            print('Starting sequence predictions')
            print()

            # for each target image we get n_prev_images to run gru on them
            # if we cant get n_prev_images, we get the maximum possible number of previous frames
            # target image is the last in the sequence

            for test_file in scene_files:
                curr_paths = []
                folder, frame_id, _ = test_file.split()
                frame_id = int(frame_id)
                left_id = max(0, frame_id - n_prev_images)

                for f_id in range(left_id, frame_id + 1):
                    image_path = get_image_path(f_id, folder)
                    curr_paths.append(image_path)

                # start prediction
                hidden_states = [gru.cgru_0.h0_layer1,
                                 gru.cgru_1.h0_layer1,
                                 gru.cgru_2.h0_layer1,
                                 gru.cgru_3.h0_layer1]

                for image_seq_id, image_path in enumerate(curr_paths):
                    # extract image
                    im = resize(pil_loader(image_path))
                    im = transforms.ToTensor()(im).float()
                    im = im.unsqueeze(0).cuda()

                    # encoder-decoder
                    features = encoder(im)
                    depth_outputs = depth_decoder(features, pre_disp)
                    
                    # gru
                    hidden_states, disp = gru(depth_outputs, hidden_states)

                outputs = {}
                outputs[('disp', 0)] = disp[('disp', 0)]

                pred_disp, _ = disp_to_depth(outputs[("disp", 0)], opt.min_depth, opt.max_depth)
                pred_disp = pred_disp.cpu()[:, 0].numpy()
                pred_disps.append(pred_disp)
                
        pred_disps = np.concatenate(pred_disps) 
                
    return pred_disps



    
def evaluate_v4_single_image(opt, dataloader, encoder, depth_decoder, gru, head): 
    pred_disps = []
    len_dataloader = len(dataloader)

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            print(i, 'from', len_dataloader, 'batches')
            input_color = data[("color", 0, 0)].cuda()

            # encoder-decoder outputs
            depth_outputs = depth_decoder(encoder(input_color), True)

            # gru
            bs = input_color.shape[0]

            # init hidden states
            hidden_states = [gru.cgru_0.h0_layer1.repeat(bs, 1, 1, 1),
                             gru.cgru_1.h0_layer1.repeat(bs, 1, 1, 1),
                             gru.cgru_2.h0_layer1.repeat(bs, 1, 1, 1),
                             gru.cgru_3.h0_layer1.repeat(bs, 1, 1, 1)]  

            # gru + fusion / concat
            hidden_states_new = gru(depth_outputs, hidden_states)

            outputs = {}
            outputs[('disp', 0)] = depth_outputs[('disp', 0)] + hidden_states_new[0]
            outputs[('disp', 1)] = depth_outputs[('disp', 1)] + hidden_states_new[1]
            outputs[('disp', 2)] = depth_outputs[('disp', 2)] + hidden_states_new[2]
            outputs[('disp', 3)] = depth_outputs[('disp', 3)] + hidden_states_new[3]

            outputs = head(outputs)

            pred_disp, _ = disp_to_depth(outputs[("disp", 0)], opt.min_depth, opt.max_depth)
            pred_disp = pred_disp.cpu()[:, 0].numpy()
            pred_disps.append(pred_disp)
            
        pred_disps = np.concatenate(pred_disps) 
        
    return pred_disp



def evaluate_v7_seq(opt, test_files, encoder, depth_decoder, gru, head):
    test_scenes = sorted(list(test_files.keys()))
    pred_disps = []
    interp = Image.ANTIALIAS
    resize = transforms.Resize((192, 640), interpolation=interp)            
        
    with torch.no_grad():
        
        # loop over scenes
        for scene in test_scenes:
            scene_files = sorted(test_files[scene])
            print('Get predictions for scene', scene)
            print('Number of test files in scene', len(scene_files))
            print()

            hidden_curr = [gru.cgru_0.h0_layer1,
                             gru.cgru_1.h0_layer1,
                             gru.cgru_2.h0_layer1,
                             gru.cgru_3.h0_layer1]

            # loop over images in scene
            for img in scene_files:
                folder, frame_id, _ = img.split()
                frame_id = int(frame_id)

                image_path = get_image_path(frame_id, folder)
                rgb = resize(pil_loader(image_path))
                im = transforms.ToTensor()(rgb).float()
                im = im.unsqueeze(0).cuda()
                
                depth_outputs = depth_decoder(encoder(im), True)

                hidden_prev = hidden_curr
                hidden_states = []
                for i in range(4):
                    hidden_states.append((hidden_curr[i] + hidden_prev[i]) / 2)
                
                hidden_curr = gru(depth_outputs, hidden_states)
            
                outputs = head(hidden_curr, zero_scale_only=True)
                pred_disp, _ = disp_to_depth(outputs[("disp", 0)], opt.min_depth, opt.max_depth)
                pred_disp = pred_disp.cpu()[:, 0].numpy()
                pred_disps.append(pred_disp)
                
        pred_disps = np.concatenate(pred_disps) 
                
    return pred_disps



    
def evaluate_v4_seq(opt, test_files, encoder, depth_decoder, gru, head):
    test_scenes = sorted(list(test_files.keys()))
    pred_disps = []
    interp = Image.ANTIALIAS
    resize = transforms.Resize((192, 640), interpolation=interp)            
        
    with torch.no_grad():
        
        # loop over scenes
        for scene in test_scenes:
            scene_files = sorted(test_files[scene])
            print('Get predictions for scene', scene)
            print('Number of test files in scene', len(scene_files))
            print()

            hidden_states = [gru.cgru_0.h0_layer1,
                             gru.cgru_1.h0_layer1,
                             gru.cgru_2.h0_layer1,
                             gru.cgru_3.h0_layer1]
            hidden_outputs = [] # for zero scale only
            hidden_outputs_aggr = [] # for zero scale only
            depth_outputs_list = [] # for zero scale only

            # append init hidden (history for 0 frame)
            hidden_outputs.append(hidden_states[0])
            
            # loop over images in scene
            for img in scene_files:
                folder, frame_id, _ = img.split()
                frame_id = int(frame_id)

                image_path = get_image_path(frame_id, folder)
                rgb = resize(pil_loader(image_path))
                im = transforms.ToTensor()(rgb).float()
                im = im.unsqueeze(0).cuda()
                
                depth_outputs = depth_decoder(encoder(im), True)
                depth_outputs_list.append(depth_outputs[('disp', 0)])

                hidden_states = gru(depth_outputs, hidden_states)
                hidden_outputs.append(hidden_states[0])
            
            # aggregate hidden states (get h t-1 and h t)
            hidden_outputs = torch.stack(hidden_outputs, 1)
            hidden_outputs_aggr = (hidden_outputs[:,1:] + hidden_outputs[:,:-1]) / 2
            
            # sum depth outputs and gru outputs, run head, get depth
            for i in range(len(depth_outputs_list)): # for scene length
                head_input = {}
                head_input[('disp', 0)] = depth_outputs_list[i] + hidden_outputs_aggr[:, i]
                outputs = head(head_input, zero_scale_only=True)
                pred_disp, _ = disp_to_depth(outputs[("disp", 0)], opt.min_depth, opt.max_depth)
                pred_disp = pred_disp.cpu()[:, 0].numpy()
                pred_disps.append(pred_disp)
                
        pred_disps = np.concatenate(pred_disps) 
                
    return pred_disps


def evaluate_v4_seq_prev_images(opt, test_files, encoder, depth_decoder, gru, head, skip_frames=False):
    n_prev_images = 10
    test_scenes = sorted(list(test_files.keys()))
    pred_disps = []
    interp = Image.ANTIALIAS
    resize = transforms.Resize((192, 640), interpolation=interp)       
    
    total_num_of_test_files = len(test_scenes) * 25
    global_frame_count = 0
    skipped_ids = [] # global id of image in test files, needed to remove gt_depths from evaluation
    
    print('N prev images', n_prev_images)
    print('Evaluation with skipping files without prev frames:', skip_frames)
        
    with torch.no_grad():
        
        # loop over scenes
        for scene_id, scene in enumerate(test_scenes):
            scene_files = sorted(test_files[scene])
            print()
            print('Get predictions for scene', scene)
            print('Number of test files in scene', len(scene_files))
            print('Starting sequence predictions')
            print('Last file id', int(scene_files[-1].split()[1]))

            # for each target image we get n_prev_images to run gru on them
            # if we cant get n_prev_images, we get the maximum possible number of previous frames
            # target image is the last in the sequence

            for test_file_id, test_file in enumerate(scene_files):
                curr_paths = []
                folder, frame_id, _ = test_file.split()
                frame_id = int(frame_id)
                left_id = max(0, frame_id + 1 - n_prev_images)
                
                if skip_frames and frame_id + 1 - n_prev_images < 0:
                    print('Skipping frame_id', frame_id)
                    skipped_ids.append(scene_id * 25 + test_file_id)
                else:
                    global_frame_count += 1
                    for f_id in range(left_id, frame_id + 1):
                        image_path = get_image_path(f_id, folder)
                        curr_paths.append(image_path)

                    # start prediction
                    hidden_curr = [gru.cgru_0.h0_layer1,
                                   gru.cgru_1.h0_layer1,
                                   gru.cgru_2.h0_layer1,
                                   gru.cgru_3.h0_layer1]

                    for image_seq_id, image_path in enumerate(curr_paths):
                        # extract image
                        im = resize(pil_loader(image_path))
                        im = transforms.ToTensor()(im).float()
                        im = im.unsqueeze(0).cuda()

                        # encoder-decoder
                        features = encoder(im)
                        depth_outputs = depth_decoder(features, True)

                        # gru (update hidden states)
                        hidden_prev = hidden_curr[0]
                        hidden_curr = gru(depth_outputs, hidden_curr)

                    # aggregate hidden states
                    hidden_states_aggr = (hidden_prev + hidden_curr[0]) / 2

                    # add to depth outputs (last depth outputs = from target image)
                    outputs = {}
                    outputs[('disp', 0)] = depth_outputs[('disp', 0)] + hidden_states_aggr

                    outputs = head(outputs, zero_scale_only=True)

                    pred_disp, _ = disp_to_depth(outputs[("disp", 0)], opt.min_depth, opt.max_depth)
                    pred_disp = pred_disp.cpu()[:, 0].numpy()
                    pred_disps.append(pred_disp)

        pred_disps = np.concatenate(pred_disps) 
        
        print('Processed', global_frame_count, 'frames out of', total_num_of_test_files)
                
    return pred_disps, skipped_ids
 
    
def evaluate_mdp_seq(opt, test_files, encoder, depth_decoder, skip_frames=True):
    n_prev_images = 10
    test_scenes = sorted(list(test_files.keys()))
    pred_disps = []
    interp = Image.ANTIALIAS
    resize = transforms.Resize((192, 640), interpolation=interp)       
    
    total_num_of_test_files = len(test_scenes) * 25
    global_frame_count = 0
    skipped_ids = [] # global id of image in test files, needed to remove gt_depths from evaluation
    
    print('N prev images', n_prev_images)
    print('Evaluation with skipping files without prev frames:', skip_frames)
        
    with torch.no_grad():
        
        # loop over scenes
        for scene_id, scene in enumerate(test_scenes):
            scene_files = sorted(test_files[scene])
            print()
            print('Get predictions for scene', scene)
            print('Number of test files in scene', len(scene_files))
            print('Starting sequence predictions')
            print('Last file id', int(scene_files[-1].split()[1]))

            # for each target image we get n_prev_images to run gru on them
            # if we cant get n_prev_images, we get the maximum possible number of previous frames
            # target image is the last in the sequence

            for test_file_id, test_file in enumerate(scene_files):
                curr_paths = []
                folder, frame_id, _ = test_file.split()
                frame_id = int(frame_id)
                left_id = max(0, frame_id + 1 - n_prev_images)
                
                if skip_frames and frame_id + 1 - n_prev_images < 0:
                    print('Skipping frame_id', frame_id)
                    skipped_ids.append(scene_id * 25 + test_file_id)
                else:
                    global_frame_count += 1

                    # extract image
                    image_path = get_image_path(frame_id, folder)
                    im = resize(pil_loader(image_path))
                    im = transforms.ToTensor()(im).float()
                    im = im.unsqueeze(0).cuda()

                    # encoder-decoder
                    features = encoder(im)
                    depth_outputs = depth_decoder(features, False)

                    pred_disp, _ = disp_to_depth(depth_outputs[("disp", 0)], opt.min_depth, opt.max_depth)
                    pred_disp = pred_disp.cpu()[:, 0].numpy()
                    pred_disps.append(pred_disp)

        pred_disps = np.concatenate(pred_disps) 
        
        print('Processed', global_frame_count, 'frames out of', total_num_of_test_files)
                
    return pred_disps, skipped_ids



def evaluate_v5_seq(opt, test_files, encoder, depth_decoder, gru):
    test_scenes = sorted(list(test_files.keys()))
    pred_disps = []
    interp = Image.ANTIALIAS
    resize = transforms.Resize((192, 640), interpolation=interp)            
        
    with torch.no_grad():
        
        # loop over scenes
        for scene in test_scenes:
            scene_files = sorted(test_files[scene])
            print('Get predictions for scene', scene)
            print('Number of test files in scene', len(scene_files))
            print()

            hidden_curr = [gru.cgru_0.h0_layer1,
                           gru.cgru_1.h0_layer1,
                           gru.cgru_2.h0_layer1,
                           gru.cgru_3.h0_layer1,
                           gru.cgru_4.h0_layer1]
            
            # loop over images in scene
            for img in scene_files:
                folder, frame_id, _ = img.split()
                frame_id = int(frame_id)

                image_path = get_image_path(frame_id, folder)
                rgb = resize(pil_loader(image_path))
                im = transforms.ToTensor()(rgb).float()
                im = im.unsqueeze(0).cuda()
                
                # encoder
                features = encoder(im)
                
                # gru
                hidden_prev = hidden_curr
                hidden_curr = gru(features, hidden_curr)
            
                # aggregate hidden states (get h t-1 and h t)
                for scale in range(5):
                    features[scale] = features[scale] + (hidden_curr[scale] + hidden_prev[scale]) / 2
                    
                outputs = depth_decoder(features)
                
                pred_disp, _ = disp_to_depth(outputs[("disp", 0)], opt.min_depth, opt.max_depth)
                pred_disp = pred_disp.cpu()[:, 0].numpy()
                pred_disps.append(pred_disp)
                
        pred_disps = np.concatenate(pred_disps) 
                
    return pred_disps


def evaluate_v8_seq(opt, test_files, encoder, depth_decoder, gru):
    n_prev_images = 6
    test_scenes = sorted(list(test_files.keys()))
    pred_disps = []
    interp = Image.ANTIALIAS
    resize = transforms.Resize((192, 640), interpolation=interp)            
        
    with torch.no_grad():
        
        # loop over scenes
        for scene in test_scenes:
            scene_files = sorted(test_files[scene])
            print('Get predictions for scene', scene)
            print('Number of test files in scene', len(scene_files))
            print('Starting sequence predictions')
            print()

            # for each target image we get n_prev_images to run gru on them
            # if we cant get n_prev_images, we get the maximum possible number of previous frames
            # target image is the last in the sequence

            for test_file in scene_files:
                curr_paths = []
                folder, frame_id, _ = test_file.split()
                frame_id = int(frame_id)
                left_id = max(0, frame_id - n_prev_images)

                for f_id in range(left_id, frame_id + 1):
                    image_path = get_image_path(f_id, folder)
                    curr_paths.append(image_path)

                # start prediction
                hidden_states = [(gru.cgru_0.h0_layer1, gru.cgru_0.c0_layer1),
                                 (gru.cgru_1.h0_layer1, gru.cgru_1.c0_layer1),
                                 (gru.cgru_2.h0_layer1, gru.cgru_2.c0_layer1),
                                 (gru.cgru_3.h0_layer1, gru.cgru_3.c0_layer1),
                                ]

                for image_seq_id, image_path in enumerate(curr_paths):
                    # extract image
                    im = resize(pil_loader(image_path))
                    im = transforms.ToTensor()(im).float()
                    im = im.unsqueeze(0).cuda()

                    # encoder-decoder
                    features = encoder(im)
                    depth_outputs = depth_decoder(features, True)

                    
                    # gru
                    hidden_states, disp = gru(depth_outputs, hidden_states)

                outputs = {}
                outputs[('disp', 0)] = disp[('disp', 0)]

                pred_disp, _ = disp_to_depth(outputs[("disp", 0)], opt.min_depth, opt.max_depth)
                pred_disp = pred_disp.cpu()[:, 0].numpy()
                pred_disps.append(pred_disp)
                
        pred_disps = np.concatenate(pred_disps) 
                
    return pred_disps



def evaluate_v9_v10_seq(opt, test_files, encoder, depth_decoder, gru):
    n_prev_images = 10
    test_scenes = sorted(list(test_files.keys()))
    pred_disps = []
    interp = Image.ANTIALIAS
    resize = transforms.Resize((192, 640), interpolation=interp)            
        
    with torch.no_grad():
        
        # loop over scenes
        for scene in test_scenes:
            scene_files = sorted(test_files[scene])
            print('Get predictions for scene', scene)
            print('Number of test files in scene', len(scene_files))
            print('Starting sequence predictions')
            print()

            # for each target image we get n_prev_images to run gru on them
            # if we cant get n_prev_images, we get the maximum possible number of previous frames
            # target image is the last in the sequence

            for test_file in scene_files:
                curr_paths = []
                folder, frame_id, _ = test_file.split()
                frame_id = int(frame_id)
                left_id = max(0, frame_id - n_prev_images)

                for f_id in range(left_id, frame_id + 1):
                    image_path = get_image_path(f_id, folder)
                    curr_paths.append(image_path)

                # start prediction
                hidden_states = [gru.cgru_0.h0_layer1,
                                 gru.cgru_1.h0_layer1,
                                 gru.cgru_2.h0_layer1,
                                 gru.cgru_3.h0_layer1,
                                ]

                for image_seq_id, image_path in enumerate(curr_paths):
                    # extract image
                    im = resize(pil_loader(image_path))
                    im = transforms.ToTensor()(im).float()
                    im = im.unsqueeze(0).cuda()

                    # encoder-decoder
                    features = encoder(im)
                    depth_outputs = depth_decoder(features, True)

                    # gru
                    hidden_states, disp = gru(depth_outputs, hidden_states)

                outputs = {}
                outputs[('disp', 0)] = disp[('disp', 0)]

                pred_disp, _ = disp_to_depth(outputs[("disp", 0)], opt.min_depth, opt.max_depth)
                pred_disp = pred_disp.cpu()[:, 0].numpy()
                pred_disps.append(pred_disp)
                
        pred_disps = np.concatenate(pred_disps) 
                
    return pred_disps

def evaluate_v10_single_image(opt, dataloader, encoder, depth_decoder, gru):
    pred_disps = []
    len_dataloader = len(dataloader)
    
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            print(i, 'from', len_dataloader, 'batches')
            input_color = data[("color", 0, 0)].cuda()   

            # encoder-decoder outputs
            depth_outputs = depth_decoder(encoder(input_color), True)

            # gru
            bs = input_color.shape[0]

            hidden_states = [gru.cgru_0.h0_layer1.repeat(bs, 1, 1, 1),
                             gru.cgru_1.h0_layer1.repeat(bs, 1, 1, 1),
                             gru.cgru_2.h0_layer1.repeat(bs, 1, 1, 1),
                             gru.cgru_3.h0_layer1.repeat(bs, 1, 1, 1),
                            ]

            outputs = {}
            for i in range(4):
                outputs[('disp', i)] = []

            gru_inputs = {}
            gru_inputs[('disp', 0)] = depth_outputs[('disp', 0)]
            gru_inputs[('disp', 1)] = depth_outputs[('disp', 1)]
            gru_inputs[('disp', 2)] = depth_outputs[('disp', 2)]
            gru_inputs[('disp', 3)] = depth_outputs[('disp', 3)]

            hidden_states, disp = gru(gru_inputs, hidden_states)

            outputs[('disp', 0)] = disp[('disp', 0)]
            outputs[('disp', 1)] = disp[('disp', 1)]
            outputs[('disp', 2)] = disp[('disp', 2)]
            outputs[('disp', 3)] = disp[('disp', 3)]

            pred_disp, _ = disp_to_depth(outputs[("disp", 0)], opt.min_depth, opt.max_depth)
            pred_disp = pred_disp.cpu()[:, 0].numpy()

            if opt.post_process:
                N = pred_disp.shape[0] // 2
                pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])

            pred_disps.append(pred_disp)
        pred_disps = np.concatenate(pred_disps) 

    return pred_disps 

    
def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80
    
    single_image_eval = False
    pre_disp = False
    gru_version = opt.gru_version
    skipped_ids = []
    
    assert sum((opt.eval_mono, opt.eval_stereo)) == 1, \
        "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"

    if single_image_eval:
        gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
    else:
        gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths_seq.npz")
        
    gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]

    if opt.ext_disp_to_eval is None:

        opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

        assert os.path.isdir(opt.load_weights_folder), \
            "Cannot find a folder at {}".format(opt.load_weights_folder)

        print("-> Loading weights from {}".format(opt.load_weights_folder))

        filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
        encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
        decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")
        gru_path = os.path.join(opt.load_weights_folder, "gru.pth")
        head_path = os.path.join(opt.load_weights_folder, "head.pth")
        
        print('Initialize models')
        encoder = networks.ResnetEncoder(18, True)
        depth_decoder = networks.DepthDecoder(encoder.num_ch_enc)
        
        if gru_version == 'v2':
            gru = networks.ConvGRUBlocks_v2(kernel_size=(3, 3), bias=True, device=torch.device('cuda:0'),
                                           attention=True)
        elif gru_version in ('v2', 'v2_wo_att'):
            gru = networks.ConvGRUBlocks_v2(kernel_size=(3, 3), bias=True, device=torch.device('cuda:0'), 
                                            attention=False)
        elif gru_version == 'v3':
            gru = networks.ConvGRUBlocks_v3(kernel_size=(3, 3), bias=True, device=torch.device('cuda:0'))
        elif gru_version == 'v4':
            gru = networks.ConvGRUBlocks_v4(kernel_size=(3, 3), bias=True, device=torch.device('cuda:0'))
            head = networks.Head_v4()
        elif gru_version == 'v5':
            gru = networks.ConvGRUBlocks_v5(kernel_size=(3, 3), bias=True, device=torch.device('cuda:0'))
        elif gru_version == 'v7':
            gru = networks.ConvGRUBlocks_v7(kernel_size=(3, 3), bias=True, device=torch.device('cuda:0'))
            head = networks.Head_v7()
        elif gru_version == 'v8':
            gru = networks.ConvGRUBlocks_v8(kernel_size=(3, 3), bias=True, 
                                            device=torch.device('cuda:0'), attention=False)
        elif gru_version == 'v9':
            gru = networks.ConvGRUBlocks_v9(kernel_size=(3, 3), bias=True, 
                                            device=torch.device('cuda:0'), attention=True)
        elif gru_version == 'v10':
            gru = networks.ConvGRUBlocks_v9(kernel_size=(3, 3), bias=True, 
                                            device=torch.device('cuda:0'), attention=False)
        else:
            gru = networks.ConvGRUBlocks(kernel_size=(3, 3), bias=True, device=torch.device('cuda:0'))

        print('Load state dicts')
        # load encoder
        encoder_dict = torch.load(encoder_path)
        model_dict = encoder.state_dict()
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        # load decoder
        depth_decoder.load_state_dict(torch.load(decoder_path))
        # load gru
        if gru_version in ['v2', 'v2_wo_att', 'v3', 'v4', 'v5', 'v7', 'v8', 'v9', 'v10']:
            gru.load_state_dict(torch.load(gru_path))
            gru.cuda()
            gru.eval()
            if gru_version in ['v4', 'v7']:
                head.load_state_dict(torch.load(head_path))
                head.cuda()
                head.eval()

        encoder.cuda()
        encoder.eval()
        depth_decoder.cuda()
        depth_decoder.eval()

            
        print('--> Single_image_eval', single_image_eval)
        print('--> GRU pre disp', pre_disp)
        print('--> GRU version', gru_version)
        
        if single_image_eval: # predict always initializing hidden states
            dataset = datasets.KITTIRAWDataset(opt.data_path, filenames,
                                               encoder_dict['height'], encoder_dict['width'],
                                               [0], 4, is_train=False)
            dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers,
                                    pin_memory=True, drop_last=False)

            if gru_version in ('v2', 'v2_wo_att'):
                pred_disps = evaluate_v3_single_image(opt, dataloader, 
                                                      encoder, depth_decoder, gru, pre_disp=False)
            elif gru_version == 'v4': 
                pred_disps = evaluate_v4_single_image(opt, dataloader, encoder, depth_decoder, gru, head)
            elif gru_version == 'v10': 
                pred_disps = evaluate_v10_single_image(opt, dataloader, encoder, depth_decoder, gru)
            else:
                pred_disps = evaluate_v3_single_image(opt, dataloader, encoder, depth_decoder, gru, pre_disp)

        else: # predict for sequences
            
            with open("splits/eigen/test_files.txt", 'r') as f:
                test_seq = f.read().splitlines()
            
            test_files = defaultdict(list)
            for img in test_seq:
                scene, _, _ = img.split()
                test_files[scene].append(img) 

            if gru_version in ('v2', 'v2_wo_att'):
                pred_disps = evaluate_v3_seq_prev_images(opt, test_files, encoder, depth_decoder, gru, pre_disp=False)
            elif gru_version == 'v3':
                pred_disps = evaluate_v3_seq_prev_images(opt, test_files, encoder, depth_decoder, gru, pre_disp=True)
            elif gru_version == 'v4': 
                # pred_disps = evaluate_v4_seq(opt, test_files, encoder, depth_decoder, gru, head)
                # pred_disps = evaluate_v4_seq_prev_images(opt, test_files, 
                # encoder, depth_decoder, gru, head)
                pred_disps, skipped_ids = evaluate_v4_seq_prev_images(opt, test_files, 
                                                         encoder, depth_decoder, gru, head, skip_frames=True)
            elif gru_version == 'v5':
                pred_disps = evaluate_v5_seq(opt, test_files, encoder, depth_decoder, gru)
            elif gru_version == 'v7':
                pred_disps = evaluate_v7_seq(opt, test_files, encoder, depth_decoder, gru, head)
            elif gru_version == 'v8':
                pred_disps = evaluate_v8_seq(opt, test_files, encoder, depth_decoder, gru)
            elif gru_version in ('v9', 'v10'):
                pred_disps = evaluate_v9_v10_seq(opt, test_files, encoder, depth_decoder, gru)
            elif gru_version == 'mdp':
                pred_disps, skipped_ids = evaluate_mdp_seq(opt, test_files, encoder, depth_decoder, skip_frames=True)
                
    else:
        # Load predictions from file
        print("-> Loading predictions from {}".format(opt.ext_disp_to_eval))
        pred_disps = np.load(opt.ext_disp_to_eval)

        if opt.eval_eigen_to_benchmark:
            eigen_to_benchmark_ids = np.load(
                os.path.join(splits_dir, "benchmark", "eigen_to_benchmark_ids.npy"))

            pred_disps = pred_disps[eigen_to_benchmark_ids]

    if opt.save_pred_disps:
        output_path = os.path.join(
            opt.load_weights_folder, "disps_{}_split.npy".format(opt.eval_split))
        print("-> Saving predicted disparities to ", output_path)
        np.save(output_path, pred_disps)

    if opt.no_eval:
        print("-> Evaluation disabled. Done.")
        quit()

    elif opt.eval_split == 'benchmark':
        save_dir = os.path.join(opt.load_weights_folder, "benchmark_predictions")
        print("-> Saving out benchmark predictions to {}".format(save_dir))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for idx in range(len(pred_disps)):
            disp_resized = cv2.resize(pred_disps[idx], (1216, 352))
            depth = STEREO_SCALE_FACTOR / disp_resized
            depth = np.clip(depth, 0, 80)
            depth = np.uint16(depth * 256)
            save_path = os.path.join(save_dir, "{:010d}.png".format(idx))
            cv2.imwrite(save_path, depth)

        print("-> No ground truth is available for the KITTI benchmark, so not evaluating. Done.")
        quit()

    # gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
    # gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]

    print("-> Evaluating")

    if opt.eval_stereo:
        print("   Stereo evaluation - "
              "disabling median scaling, scaling by {}".format(STEREO_SCALE_FACTOR))
        opt.disable_median_scaling = True
        opt.pred_depth_scale_factor = STEREO_SCALE_FACTOR
    else:
        print("   Mono evaluation - using median scaling")

    errors = []
    ratios = []
    gt_depths = np.delete(gt_depths, skipped_ids, axis=0)
    
    print('-->Preds and gt_depths lengths are equal', pred_disps.shape[0] == gt_depths.shape[0])

    for i in range(pred_disps.shape[0]):

        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 1 / pred_disp

        pred_depth, gt_depth = transform_depth(pred_depth, gt_depth)

#         if opt.eval_split == "eigen":
#             mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

#             crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
#                              0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
#             crop_mask = np.zeros(mask.shape)
#             crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
#             mask = np.logical_and(mask, crop_mask)

#         else:
#             mask = gt_depth > 0

#         pred_depth = pred_depth[mask]
#         gt_depth = gt_depth[mask]

#         pred_depth *= opt.pred_depth_scale_factor
#         if not opt.disable_median_scaling:
#             ratio = np.median(gt_depth) / np.median(pred_depth)
#             ratios.append(ratio)
#             pred_depth *= ratio

#         pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
#         pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

        errors.append(compute_errors(gt_depth, pred_depth))

    if not opt.disable_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

    mean_errors = np.array(errors).mean(0)

    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")


if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())
