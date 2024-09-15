# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 13:37:37 2019 by Attila Lengyel - attila@lengyel.nl
"""

# from utils2.helpers import gen_train_dirs, plot_confusion_matrix, get_train_trans, get_test_trans
from dataloader.transforms import train_trans, test_trans
# from utils.routines import train_epoch, evaluate
from datasets.cityscapes_ext import CityscapesExt
from datasets.nighttime_driving import NighttimeDrivingDataset
from datasets.dark_zurich import DarkZurichDataset
# from models.refinenet import RefineNet
from model_wrapper import ModelWrapper
from train import train_model_from_config
from evaluate_models import evaluate_with_loader
from adapt_dataset_and_test import test_property_shift
from dataloader.RGBXDataset import RGBXDataset
from dataloader.dataloader import get_val_loader, get_train_loader

import importlib
import json
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import Subset
import shutil, time, random
import matplotlib.pyplot as plt
import numpy as np
import os

def load_model_weights(model, dataset='cityscapes'):
    if dataset == 'cityscapes':
        depth_model_weights = r"checkpoints\Cityscapes_DFormer-Base\run_20240907-102001_depth\epoch_60_miou_58.575.pth"
        rgb_model_weights = r"checkpoints\Cityscapes_DFormer-Base\run_20240906-153744_rgb\epoch_60_miou_69.303.pth"
        rgbd_model_weights = r"checkpoints\Cityscapes_DFormer-Base\run_20240907-135905_rgbd\epoch_60_miou_73.002.pth"
    elif dataset == 'nyu':
        depth_model_weights = r"checkpoints\NYUDepthv2_DFormer-Base\run_20240610-182309_depth\epoch_100_miou_35.249.pth"
        rgb_model_weights = r"checkpoints\NYUDepthv2_DFormer-Base\run_20240610-203359_rgb\epoch_100_miou_43.123.pth"
        rgbd_model_weights = r"checkpoints\NYUDepthv2_DFormer-Base\run_20240607-111847_rgbd\epoch_100_miou_46.619.pth"
    else:
        print("No weights found for this dataset")
        return

    print("Loading model weights")
    try:
        def remove_prefix_from_state_dict(state_dict, prefix):
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith(prefix):
                    new_key = k[len(prefix):]
                    new_state_dict[new_key] = v
                else:
                    new_state_dict[k] = v
            return new_state_dict

        model.model.load_state_dict(remove_prefix_from_state_dict(torch.load(rgbd_model_weights)['model'], 'model.'), strict=True)
        model.depth_model.load_state_dict(remove_prefix_from_state_dict(torch.load(depth_model_weights)['model'], 'model.'), strict=True)
        model.rgb_model.load_state_dict(remove_prefix_from_state_dict(torch.load(rgb_model_weights)['model'], 'model.'), strict=True)
        print("Model weights loaded")
    except Exception as e:
        print(e)
        print("Could not load model")
        # exit()

def main(args):
    # Configure dataset paths here
    cs_path = os.path.abspath('datasets/Cityscapes/')
    nd_path = os.path.abspath('datasets/NighttimeDrivingTest/')
    dz_path = os.path.abspath('datasets/Dark_Zurich_val_anon/')

    print('--- Training args ---')
    print(args)

    # Fix seed
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    # Load dataset
    trainset = CityscapesExt(cs_path, split='train', target_type='semantic', transforms=train_trans)
    valset = CityscapesExt(cs_path, split='val', target_type='semantic', transforms=test_trans)
    testset_day = CityscapesExt(cs_path, split='test', target_type='semantic', transforms=test_trans)
    testset_nd = NighttimeDrivingDataset(nd_path, transforms=test_trans)
    testset_dz = DarkZurichDataset(dz_path, transforms=test_trans)

    # Use mini-dataset for debugging purposes
    if args.xs:
        trainset = Subset(trainset, list(range(5)))
        valset = Subset(valset, list(range(5)))
        testset_nd = Subset(testset_nd, list(range(5)))
        testset_dz = Subset(testset_dz, list(range(5)))
        print('WARNING: XS_DATASET SET TRUE')

    if args.dataset == 'cityscapes':

        dataloaders = {}
        dataloaders['train'] = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.workers)
        dataloaders['val'] = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
        dataloaders['test_day'] = torch.utils.data.DataLoader(testset_day, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
        dataloaders['test_nd'] = torch.utils.data.DataLoader(testset_nd, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
        dataloaders['test_dz'] = torch.utils.data.DataLoader(testset_dz, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)

        num_classes = len(CityscapesExt.validClasses)

        # Define model, loss, optimizer and scheduler
        criterion = nn.CrossEntropyLoss(ignore_index=CityscapesExt.voidClass)
        # model = RefineNet(num_classes, pretrained=args.pretrained, invariant=args.invariant)

        MODEL_CONFIGURATION_DICT_FILE = "configs/model_configurations.json"
        with open(MODEL_CONFIGURATION_DICT_FILE, 'r') as f:
            model_configurations = json.load(f)

        
        config_location = 'configs/Cityscapes_config.py'
        config = importlib.import_module(config_location.replace('.py', '').replace('/', '.')).config

        model_config = model_configurations["DFormer_base"]
        # model_config = model_configurations["DFormer_small"]
        # model_config = model_configurations["SegFormer_mit_b2"]

        config.update(model_config)
        config.update({
            "num_classes": num_classes,
            "nepochs": 60,
            "x_channels": 3,
            "x_e_channels": 1,
            "num_train_imgs": len(trainset),
            "num_eval_imgs": len(valset),
            "background": CityscapesExt.voidClass,
            "batch_size": 1,
        })

        model = ModelWrapper(
            config=config,
            criterion=criterion,
            norm_layer=nn.BatchNorm2d,
            pretrained=True,
        )

        model.set_ood_scores("cityscapes_training_ood_scores.json")
    
    elif args.dataset == 'nyu':
        MODEL_CONFIGURATION_DICT_FILE = "configs/model_configurations.json"
        with open(MODEL_CONFIGURATION_DICT_FILE, 'r') as f:
            model_configurations = json.load(f)

        
        config_location = 'configs/NYUDepthV2_base_config.py'
        config = importlib.import_module(config_location.replace('.py', '').replace('/', '.')).config

        criterion = nn.CrossEntropyLoss(ignore_index=config.background)

        model_config = model_configurations["DFormer_base"]

        config.update(model_config)

        dataloaders = {}
        dataloaders['val'], _ = get_val_loader(None, RGBXDataset, config, 1)
        dataloaders['train'], _ = get_train_loader(None, RGBXDataset, config)

        model = ModelWrapper(
            config=config,
            criterion=criterion,
            norm_layer=nn.BatchNorm2d,
            pretrained=True,
        )

        model.set_ood_scores("nyud_training_ood_scores.json")

    if args.mode == 'train':
        train_model_from_config(config=config, train_loader=dataloaders['train'], val_loader=dataloaders['val'])
    elif args.mode == 'test':
        with open('results.txt', 'a') as f:
            f.write(str(config) + '\n')
            f.write(f"Time: {time.strftime('%H:%M:%S', time.gmtime())}\n")

        model.to('cuda')
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        
        # Load model weights
        load_model_weights(model, args.dataset)

        torch.cuda.empty_cache()
        model.eval()

        with torch.no_grad():
            evaluator = evaluate_with_loader(
                model=model,
                dataloader=dataloaders['val'],
                config=config,
                device='cuda',
                bin_size=float('inf'),
            )

            print("val miou: ", evaluator.miou)

            with open('results.txt', 'a') as f:
                f.write(f"validation results: {str(evaluator.miou)} \n")

        torch.manual_seed(0)
        torch.cuda.manual_seed(0)

        model2 = ModelWrapper(
            config=config,
            criterion=criterion,
            norm_layer=nn.BatchNorm2d,
            pretrained=True,
        )
        model2.to('cuda')
        load_model_weights(model2, args.dataset)
        torch.cuda.empty_cache()
        model2.eval()

        with torch.no_grad():

            evaluator2 = evaluate_with_loader(
                model=model2,
                dataloader=dataloaders['test_nd'],
                config=config,
                device='cuda',
                bin_size=float('inf'),
            )

            print("test night miou: ", evaluator2.miou)

            with open('results.txt', 'a') as f:
                f.write(f"night results: {str(evaluator2.miou)} \n")

        torch.manual_seed(0)
        torch.cuda.manual_seed(0)

        model3 = ModelWrapper(
            config=config,
            criterion=criterion,
            norm_layer=nn.BatchNorm2d,
            pretrained=True,
        )
        model3.to('cuda')
        load_model_weights(model3, args.dataset)
        torch.cuda.empty_cache()
        model3.eval() 

        with torch.no_grad():
            evaluator3 = evaluate_with_loader(
                model=model3,
                dataloader=dataloaders['test_dz'],
                config=config,
                device='cuda',
                bin_size=float('inf'),
            )

            print("test dark zurich miou: ", evaluator3.miou)

            with open('results.txt', 'a') as f:
                f.write(f"dark zurich results: {str(evaluator3.miou)} \n")

    elif args.mode == 'adapt':
        model.to('cuda')
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        
        # Load model weights
        load_model_weights(model, args.dataset)

        torch.cuda.empty_cache()
        model.eval()

        # test_property_shift(
        #     config=config, 
        #     property_values=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 
        #     model=model, 
        #     dataloader=dataloaders['val'],
        #     property_name='brightness', 
        #     origin_directory_path=r"datasets\Cityscapes\leftImg8bit\val_original",
        #     destination_directory_path=r"datasets\Cityscapes\leftImg8bit\val", 
        #     split="test", 
        #     device="cuda",
        # )

        # test_property_shift(
        #     config=config,
        #     property_values=[0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
        #     model=model,
        #     dataloader=dataloaders['val'],
        #     property_name='noise',
        #     origin_directory_path=r"datasets\Cityscapes\leftImg8bit\val_original",
        #     destination_directory_path=r"datasets\Cityscapes\leftImg8bit\val",
        #     split="test",
        #     device="cuda",
        # )

        # test_property_shift(
        #     config=config,
        #     property_values=[0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
        #     model=model,
        #     dataloader=dataloaders['val'],
        #     property_name='noise',
        #     origin_directory_path=r"datasets\Cityscapes\depth\val_original",
        #     destination_directory_path=r"datasets\Cityscapes\depth\val",
        #     split="test",
        #     device="cuda",
        # )

        # test_property_shift(
        #     config=config,
        #     property_values=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        #     model=model,
        #     dataloader=dataloaders['val'],
        #     property_name='brightness',
        #     origin_directory_path=r"datasets\NYUDepthv2\RGB_original",
        #     destination_directory_path=r"datasets\NYUDepthv2\RGB",
        #     split="test",
        #     device="cuda",
        # )

        test_property_shift(
            config=config,
            property_values=[0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
            model=model,
            dataloader=dataloaders['val'],
            property_name='noise',
            origin_directory_path=r"datasets\NYUDepthv2\RGB_original",
            destination_directory_path=r"datasets\NYUDepthv2\RGB",
            split="test",
            device="cuda",
        )

        # test_property_shift(
        #     config=config,
        #     property_values=[0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
        #     model=model,
        #     dataloader=dataloaders['val'],
        #     property_name='noise',
        #     origin_directory_path=r"datasets\NYUDepthv2\Depth_original",
        #     destination_directory_path=r"datasets\NYUDepthv2\Depth",
        #     split="test",
        #     device="cuda",
        # )


        # with torch.no_grad():
        #     evaluator = evaluate_with_loader(
        #         model=model,
        #         dataloader=dataloaders['val'],
        #         config=config,
        #         device='cuda',
        #         bin_size=float('inf'),
        #     )

        #     print("val miou: ", evaluator.miou)

        #     with open('results.txt', 'a') as f:
        #         f.write(f"validation results: {evaluator.to_string()} \n")

        
        # with torch.no_grad():
        #     evaluator = evaluate_ood_scores_with_loader(
        #         model=model,
        #         dataloader=dataloaders['val'],
        #         config=config,
        #         device='cuda',
        #         bin_size=float('inf'),
        #     )

        #     print("val miou: ", evaluator.miou)

        #     with open('results.txt', 'a') as f:
        #         f.write(f"validation results: {evaluator.to_string()} \n")

        
        # with torch.no_grad():
        #     evaluator = evaluate_with_loader(
        #         model=model,
        #         dataloader=dataloaders['train'],
        #         config=config,
        #         device='cuda',
        #         bin_size=float('inf'),
        #     )

        #     print("val miou: ", evaluator.miou)

        #     with open('results_train.txt', 'a') as f:
        #         f.write(f"validation results: {evaluator.to_string()} \n")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Segmentation training and evaluation')
    parser.add_argument('--invariant', type=str, default=None,
                        help='invariant (E,W,C,N,H)')
    parser.add_argument('--init-scale', metavar='1.0', default=[1.0], type=float,
                        help='initial value for scale')
    parser.add_argument('--resume', type=str, default=None,
                        help='resume training from checkpoint')
    parser.add_argument('--batch-size', type=int, default=3, metavar='N',
                        help='input batch size for training (default: 3)')
    parser.add_argument('--epochs', type=int, default=120, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--wd', type=float, default=1e-4, metavar='WD',
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--hflip', action='store_true', default=False,
                        help='perform random horizontal flipping')
    parser.add_argument('--rc', action='store_true', default=False,
                        help='perform random cropping')
    parser.add_argument('--jitter', type=float, default=0.0, metavar='J',
                        help='color jitter augmentation (default: 0.0)')
    parser.add_argument('--scale', type=float, default=0.0, metavar='J',
                        help='random scale augmentation (default: 0.0)')
    parser.add_argument('--pretrained', action='store_true', default=False,
                        help='initialize feature extractor from imagenet pretrained weights')
    parser.add_argument('--xs', action='store_true', default=False,
                        help='use small dataset subset for debugging')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--workers', type=int, default=4, metavar='W',
                        help='number of data workers (default: 4)')
    parser.add_argument('-m', '--mode', type=str, default='test', 
                        help='Mode to run the script in. Either "train", "test" or "adapt"')
    parser.add_argument('-d', '--dataset', type=str, default='cityscapes',
                        help='Dataset to use. Either "cityscapes" or "nyu"')
    args = parser.parse_args()

    main(args)
