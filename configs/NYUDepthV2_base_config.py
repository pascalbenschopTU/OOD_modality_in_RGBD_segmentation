import os
import os.path as osp
import sys
import time
import numpy as np
from easydict import EasyDict as edict
import argparse

C = edict()
config = C
cfg = C

C.seed = 12345

remoteip = os.popen('pwd').read()
C.root_dir = 'datasets'
C.abs_dir = osp.realpath(".")

# Dataset config
"""Dataset Path"""
C.dataset_name = 'NYUDepthv2'
C.dataset_path = osp.join(C.root_dir, C.dataset_name)
C.rgb_root_folder = osp.join(C.dataset_path, 'RGB')
C.rgb_format = '.jpg'
C.gt_root_folder = osp.join(C.dataset_path, 'Label')
C.gt_format = '.png'
C.gt_transform = True
C.x_root_folder = osp.join(C.dataset_path, 'Depth')
C.x_format = '.png'
C.x_is_single_channel = True # True for raw depth, thermal and aolp/dolp(not aolp/dolp tri) input
C.x_channels = 3
C.x_e_channels = 1
C.train_source = osp.join(C.dataset_path, "train.txt")
C.eval_source = osp.join(C.dataset_path, "test.txt")
C.is_test = True


C.num_train_imgs = 795
C.num_eval_imgs = 654


# Set classes to "groceries", "all" or "gems"
C.classes = 'NYUDepthv2'
C.num_classes = 41
C.class_names =  ['background', 'wall','floor','cabinet','bed','chair','sofa','table','door','window','bookshelf','picture','counter','blinds',
    'desk','shelves','curtain','dresser','pillow','mirror','floor mat','clothes','ceiling','books','refridgerator',
    'television','paper','towel','shower curtain','box','whiteboard','person','night stand','toilet',
    'sink','lamp','bathtub','bag','otherstructure','otherfurniture','otherprop']


"""Image Config"""
# Background class is 0, set to -1 to ignore
C.background = 0
C.image_height = 400
C.image_width = 400
C.norm_mean = np.array([0.485, 0.456, 0.406])
C.norm_std = np.array([0.229, 0.224, 0.225])

""" Settings for network, this would be different for each kind of model"""
C.model = 'DFormer'
C.backbone = 'DFormer-Base'
C.pretrained_model = 'checkpoints/pretrained/DFormer_Base.pth.tar'
C.decoder = 'ham'
C.decoder_embed_dim = 512
C.optimizer = 'AdamW'

"""Train Config"""
C.start_epoch = 1
C.lr = 8e-5
C.lr_power = 0.9
C.momentum = 0.9
C.weight_decay = 0.01
C.lamda = 1e-6
C.batch_size = 4
C.val_batch_size = 16
C.nepochs = 40
C.niters_per_epoch = C.num_train_imgs // C.batch_size  + 1
C.num_workers = 1
C.train_scale_array = [0.5, 0.75, 1, 1.25, 1.5, 1.75]
C.warm_up_epoch = 10

C.fix_bias = True
C.bn_eps = 1e-3
C.bn_momentum = 0.1
C.drop_path_rate=0.1
C.aux_rate =0.0

"""Eval Config"""
C.eval_iter = 25
C.eval_stride_rate = 2 / 3
C.eval_scale_array = [0.5,0.75,1,1.25,1.5] # [0.75, 1, 1.25] # 0.5,0.75,1,1.25,1.5
C.eval_flip =  True # False #
C.eval_crop_size = [480, 480] # [height weight]

"""Store Config"""
C.checkpoint_start_epoch = 9 # 200
C.checkpoint_step = 10

"""Path Config"""
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
add_path(osp.join(C.root_dir))

C.checkpoint_dir = 'checkpoints\\NYUDepthv2'
C.log_dir = osp.abspath(osp.join(C.checkpoint_dir, C.dataset_name + '_' + C.backbone))
C.tb_dir = osp.abspath(osp.join(C.log_dir, "tb"))
C.log_dir_link = C.log_dir

if __name__ == '__main__':
    print(config.nepochs)

C.date_time = '20240612-121117'
