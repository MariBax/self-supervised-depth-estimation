import torch
import math
import os
import numpy as np
import random
import torch.nn.functional as F
from tqdm import tqdm

def log10(x):
    return torch.log(x) / math.log(10)

class MetricAggregator(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0.0

        self.sum_rmse, self.sum_mae = 0, 0
        self.sum_absrel, self.sum_lg10 = 0, 0
        self.sum_delta1, self.sum_delta2, self.sum_delta3 = 0, 0, 0
        self.sum_rmse_log, self.sum_sq_rel = 0, 0

    def evaluate(self, output, target):
        valid_mask = target>0
        output = output[valid_mask]
        target = target[valid_mask]
        output = output.detach().cpu().numpy()
        target = target.detach().cpu().numpy()

        abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = self.compute_errors(output, target)

        self.sum_rmse += rmse
        self.sum_absrel += abs_rel
        self.sum_sq_rel += sq_rel
        self.sum_rmse_log += rmse_log

        self.sum_delta1 += a1
        self.sum_delta2 += a2
        self.sum_delta3 += a3

        self.count += 1

    def compute_errors(self, output, target):

        thresh = np.maximum((target / output), (output / target))
        a1 = (thresh < 1.25     ).mean()
        a2 = (thresh < 1.25 ** 2).mean()
        a3 = (thresh < 1.25 ** 3).mean()

        rmse = (target - output) ** 2
        rmse = np.sqrt(rmse.mean())

        rmse_log = (np.log(target) - np.log(output)) ** 2
        rmse_log = np.sqrt(rmse_log.mean())

        abs_rel = np.mean(np.abs(target - output) / target)

        sq_rel = np.mean(((target - output) ** 2) / target)

        return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

    def average(self):
        return {
            "rmse": self.sum_rmse / self.count,
            "absrel": self.sum_absrel / self.count,
            "sq_rel": self.sum_sq_rel / self.count,
            "rmse_log": self.sum_rmse_log / self.count,
            "delta1": self.sum_delta1 / self.count,
            "delta2": self.sum_delta2 / self.count,
            "delta3": self.sum_delta3 / self.count
        }

def sort_test_seq_func(test_filenames):
    output = {}
    
    for fname in test_filenames:
        scene_name, frame, side = fname.split()
        
        if scene_name not in output:
            output[scene_name] = [frame]
        else:
            output[scene_name].append(frame)
            
    
    return output

def count_scene_frames_func(frame_seq, dataset_path):
    
    seq2n_frames_dict = {}
    
    for seq_name in frame_seq:
        scene_date, scene_name = seq_name.split('/')
        imgs_path = os.path.join(dataset_path, scene_date, scene_name, 'image_02','data')
        fnames = os.listdir(imgs_path)
        n_files = len(fnames)
        seq2n_frames_dict[seq_name] = n_files
    
    return seq2n_frames_dict

def generate_frame_seq_func(n_frames_dict, frame_seq, n, k, n_tuples):
    dataset = []
    t = tqdm(range(len(frame_seq)))
    for i in t:
        seq_name = frame_seq[i] 
        n_frames = n_frames_dict[seq_name]
        
        n_tuples_avg = n_frames // n # dunno how many sequences to generate
        #print(n_frames, n_tuples_avg, n_tuples)
        if n_tuples > n_tuples_avg: # case, when we can't generate the right amount of sub-sequences
            n_tuples_avg = n_tuples 
        
        # for debug purposes
        #n_tuples_avg = 10

        right_boundary = n_frames - 1 - n - k
        left_seq_start = random.sample(range(right_boundary), n_tuples_avg)
        tuples = [(seq_name, range(x, x + n + k)) for x in left_seq_start]
        
        dataset.extend(tuples)
        
    return dataset

def get_mask_func(n_seq, k):
    masks = []
    n_mask = n_seq
    
    for i in range(-1, -(k+1), -1):
        masks.append(np.eye(n_mask,n_mask,i))
        
    return np.sum(masks, 0)

def get_context_vectors(layer_1_vec, layer_2_vec, n_seq, k, device):
    bs, n_seq, _, _, _ = layer_1_vec.shape
    layer_1_v = layer_1_vec.view(bs, n_seq, -1)

    layer_2_v = layer_2_vec.view(bs, n_seq, -1)
    scores = torch.bmm(layer_1_v, layer_1_v.transpose(1, 2)).double()

    #scores = torch.exp(scores.double())

    mask = get_mask_func(n_seq, k)
    mask = torch.tensor(mask).unsqueeze(0).double().to(device)
    scores = mask * scores
    
    row_scores_sum = torch.sum(scores, 2).unsqueeze(2)
    
    scores = scores / (row_scores_sum + 1e-6)
    scores = scores.float()
    
    #context_vectors = torch.bmm(scores, layer_1_v)[:,k:,:]
    context_vectors = torch.bmm(scores, layer_2_v)[:,k:,:]
    
    return context_vectors.view(layer_2_vec[:,k:,:].shape)

def get_context_vector(x, prev_states_1, prev_states_2, l2_shape):
    x = x.view(1, -1)
    scores = torch.mm(x, torch.cat(prev_states_1).T)
    scores = scores / (torch.sum(scores, 1) + 1e-6)

    context_vector = torch.mm(scores, torch.cat(prev_states_2)).view(l2_shape)
    
    return context_vector

def disp_to_depth(disp, min_depth=0.1, max_depth=100.0):

    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth

def transform_depth(depth_pred, depth_gt):

    mask = depth_gt > 0
    crop_mask = torch.zeros_like(mask)
    crop_mask[:, :, 153:371, 44:1197] = 1
    mask = mask * crop_mask

    depth_gt = depth_gt[mask]
    depth_pred = depth_pred[mask]

    depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

    depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

    return depth_pred, depth_gt