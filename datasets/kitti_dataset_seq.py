import torch
import math
import numpy as np

import os 
import cv2
import random 
from PIL import Image

from torch.utils.data import Dataset
import skimage
import skimage.transform
from torchvision import transforms
import torch.nn.functional as F

from kitti_utils import generate_depth_map
from torchvision import transforms as T

def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

class KITTIDataset_v1(Dataset):
    """Superclass for different types of KITTI dataset loaders
    """
    def __init__(self, data_path, height, width, n, sequences, is_train):
        super(KITTIDataset_v1, self).__init__()

        self.sequences = sequences

        self.loader = pil_loader
        self.n = n
        self.full_res_shape = (1242, 375)
        self.height = height
        self.width = width
        self.data_path = data_path
        self.is_train = is_train
        self.to_tensor = transforms.ToTensor()


        self.brightness = 0.2
        self.contrast = 0.2
        self.saturation = 0.2
        self.hue = 0.1

#         self.color_aug = transforms.ColorJitter.get_params(self.brightness, self.contrast, self.saturation, self.hue)


        self.color_aug = transforms.Compose([
                transforms.ColorJitter(
                    self.brightness, self.contrast, self.saturation, self.hue),
            ]) 

        
        self.K = np.array([[0.58, 0, 0.5, 0],
                   [0, 1.92, 0.5, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]], dtype=np.float32)
        
        self.K_scales = {}
        
        for scale in range(4):
            K = self.K.copy()

            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            self.K_scales[("K", scale)] = torch.from_numpy(K)
            self.K_scales[("inv_K", scale)] = torch.from_numpy(inv_K)
            
        self.resize = {}
        self.interp = Image.ANTIALIAS
        for i in range(4):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)


    def check_depth(self):
        line = self.filenames[0].split()
        scene_name = line[0]
        frame_index = int(line[1])

        velo_filename = os.path.join(
            self.data_path,
            scene_name,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        return os.path.isfile(velo_filename)


    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        sequence_name, frame_range = self.sequences[index]
        frame_fnames = [self.get_image_path(x, sequence_name) for x in frame_range]

        do_flip = self.is_train and random.random() > 0.5
        do_color_aug = self.is_train and random.random() > 0.5
        
        inputs = {}
        
        # center images
        colors = [self.get_color(x, do_flip, do_color_aug) for x in frame_fnames[1:self.n + 1]]
        for i in range(4):
            for j, color in enumerate(colors):
                rgb = np.array(color[i])
#                 rgb = np.transpose(rgb, (2, 0, 1))
                inputs[('color', 0, i, j)] = self.to_tensor(rgb).float()
        
        # left images
        colors = [self.get_color(x, do_flip, do_color_aug) for x in frame_fnames[:self.n]]
        for i in range(4):
            for j, color in enumerate(colors):
                rgb = np.array(color[i])
#                 rgb = np.transpose(rgb, (2, 0, 1))
                inputs[('color', -1, i, j)] = self.to_tensor(rgb).float()

        # right images
        colors = [self.get_color(x, do_flip, do_color_aug) for x in frame_fnames[2:]]
        for i in range(4):
            for j, color in enumerate(colors):
                rgb = np.array(color[i])
#                 rgb = np.transpose(rgb, (2, 0, 1))
                inputs[('color', 1, i, j)] = self.to_tensor(rgb).float()
        
        depth = np.array([self.get_depth(x, sequence_name, do_flip) \
                          for x in frame_range[1:len(frame_range) - 1]]) # pose
        inputs[('depth_gt')] = torch.tensor(depth).unsqueeze(1).float()
        
        for scale in range(4):
            for j in range(self.n):
                inputs[("K", scale, j)] = self.K_scales[("K", scale)].detach().clone()
                inputs[("inv_K", scale, j)] = self.K_scales[("inv_K", scale)].detach().clone()
                
        return inputs

    def get_image_path(self, frame_index, sequence_name):
        
        scene_date, scene_name = sequence_name.split('/')
        f_str = "{:010d}.jpg".format(frame_index)
        
        image_path = os.path.join(
            self.data_path, 
            scene_date, 
            scene_name, 
            "image_02", 
            "data", 
            f_str)
        return image_path


    def get_color(self, image_path, do_flip, do_color_aug):
        colors = {}
        
        color = self.loader(image_path)

        for i in range(4):
            color = self.resize[i](color)

            if do_flip:
                color = color.transpose(Image.FLIP_LEFT_RIGHT)

            if do_color_aug:
                color = self.color_aug(color)
            
            colors[i] = np.asarray(color)

        return colors

    def get_depth(self, frame_index, sequence_name, do_flip):
        
        scene_date, scene_name = sequence_name.split('/')

        calib_path = os.path.join(self.data_path, scene_date)

        velo_filename = os.path.join(
            self.data_path,
            scene_date,
            scene_name,
            "velodyne_points/data/{:010d}.bin".format(frame_index))

        depth_gt = generate_depth_map(calib_path, velo_filename, 2)

        depth_gt = skimage.transform.resize(
            depth_gt, self.full_res_shape[::-1], order=0, preserve_range=True, mode='constant')

        if do_flip:
            depth_gt = np.fliplr(depth_gt).copy()

        return depth_gt

    
    
    
class KITTIDataset_v2(Dataset):
    """Superclass for different types of KITTI dataset loaders
    """
    def __init__(self, config, filenames, is_train):
        super(KITTIDataset_v2, self).__init__()

        self.config = config
        self.filenames = filenames

        self.loader = pil_loader
        self.full_res_shape = (1242, 375)
        self.height = config['height']
        self.width = config['width']
        self.is_train = is_train

        self.side_dict = {'l':2,'r':3}

        try:
            self.brightness, self.contrast = (0.8, 1.2), (0.8, 1.2)
            self.saturation, self.hue = (0.8, 1.2), (-0.1, 0.1)
            transforms.ColorJitter.get_params(self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.color_aug = transforms.ColorJitter.get_params(self.brightness, self.contrast, self.saturation, self.hue)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):

        filename = self.filenames[index]
        sequence_name, frame_index, side = filename.split()
        frame_index = int(frame_index)

        frame_fname = self.get_image_path(frame_index, sequence_name, self.side_dict[side])

        do_flip = self.is_train and random.random() > 0.5
        do_color_aug = self.config['use_color_aug'] and self.is_train and random.random() > 0.5

        rgb = self.get_color(frame_fname, do_flip, do_color_aug)

        rgb = np.transpose(rgb, (2, 0, 1))
        
        rgb = torch.tensor(rgb).float()

        depth = self.get_depth(frame_index, sequence_name, do_flip)

        depth = torch.tensor(depth).unsqueeze(0).float()

        return (rgb, depth)

    def get_image_path(self, frame_index, sequence_name, side):

        scene_date, scene_name = sequence_name.split('/')
        f_str = "{:010d}.jpg".format(frame_index)
        
        image_path = os.path.join(
            self.config['data_path'], 
            scene_date, 
            scene_name, 
            "image_0{}".format(side), 
            "data", 
            f_str)
        
        return image_path


    def get_color(self, image_path, do_flip, do_color_aug):
        color = self.loader(image_path)
        color = color.resize((self.width, self.height))

        if do_flip:
            color = color.transpose(Image.FLIP_LEFT_RIGHT)

        if do_color_aug:
            color = self.color_aug(color)

        return np.asarray(color)

    def get_depth(self, frame_index, sequence_name, do_flip):
        
        scene_date, scene_name = sequence_name.split('/')

        calib_path = os.path.join(self.config['data_path'], scene_date)
        filename = "{:010d}.bin".format(frame_index)
        
        velo_filename = os.path.join(
            self.config['data_path'],
            scene_date,
            scene_name,
            "velodyne_points/data",
            filename)

        depth_gt = generate_depth_map(calib_path, velo_filename)
        depth_gt = skimage.transform.resize(
            depth_gt, self.full_res_shape[::-1], order=0, preserve_range=True, mode='constant')

        if do_flip:
            depth_gt = np.fliplr(depth_gt).copy()

        return depth_gt

class KITTIDataset_v3(Dataset):
    """Superclass for different types of KITTI dataset loaders
    """
    def __init__(self, config, test_frame_idxs, sequence_name):
        super(KITTIDataset_v3, self).__init__()

        self.config = config
        self.test_frame_idxs = test_frame_idxs
        self.sequence_name = sequence_name

        self.frame_idxs = os.listdir(os.path.join(config['data_path'], sequence_name, 'image_02', 'data'))
        self.frame_filenames  = [os.path.join(config['data_path'], sequence_name, 'image_02', 'data', x) for x in self.frame_idxs]

    
        self.loader = pil_loader
        self.full_res_shape = (1242, 375)
        self.height = config['height']
        self.width = config['width']

    def __len__(self):
        return len(self.frame_filenames)

    def __getitem__(self, index):

        frame_fname = self.frame_filenames[index]
        frame_idx = self.frame_idxs[index].split('.')[0]

        rgb = self.get_color(frame_fname)
        rgb = np.transpose(rgb, (2, 0, 1))
        rgb = torch.tensor(rgb).float().unsqueeze(0)

        if frame_idx in self.test_frame_idxs:

            depth = self.get_depth(frame_idx, self.sequence_name)

            depth = torch.tensor(depth).unsqueeze(0).float()
            depth = depth.unsqueeze(0)
        else:
            depth = 0

        return (frame_idx, rgb, depth)

    def get_color(self, image_path):
        color = self.loader(image_path)
        color = color.resize((self.width, self.height))

        return np.asarray(color)

    def get_depth(self, frame_index, sequence_name):
        
        scene_date, scene_name = sequence_name.split('/')

        calib_path = os.path.join(self.config['data_path'], scene_date)
        filename = "{}.bin".format(frame_index)
        
        velo_filename = os.path.join(
            self.config['data_path'],
            scene_date,
            scene_name,
            "velodyne_points/data",
            filename)

        depth_gt = generate_depth_map(calib_path, velo_filename)
        depth_gt = skimage.transform.resize(
            depth_gt, self.full_res_shape[::-1], order=0, preserve_range=True, mode='constant')

        return depth_gt