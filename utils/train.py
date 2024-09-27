import os
import sys
import time
import json
import random
import argparse
from tqdm import tqdm
import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid
import numpy as np

sys.path.append("../UsefullnessOfDepth")

# Utils
from utils.dataloader.dataloader import get_train_loader,get_val_loader
from utils.dataloader.RGBXDataset import RGBXDataset
from utils.lr_policy import WarmUpPolyLR
from utils.evaluate_models import save_results, evaluate_with_loader
# from utils.hyperparameter_tuning import tune_hyperparameters
from utils.update_config import update_config
from utils.model_wrapper import ModelWrapper

MODEL_CONFIGURATION_DICT_FILE = "configs/model_configurations.json"

class DummyTB:
    def add_scalar(self, *args, **kwargs):
        pass

class DummySummary:
    def visualize_image(self, *args, **kwargs):
        pass


def train_model_from_config(config, **kwargs):
    print(f"Using kwargs: {kwargs}")

    is_tuning = kwargs.get('is_tuning', False)
    use_tb = config.get('use_tb', False)
    use_aux = config.get('use_aux', False)

    tb, tb_summary = DummyTB(), DummySummary()
    if not is_tuning:
        config.log_dir = config.log_dir+'/run_'+time.strftime('%Y%m%d-%H%M%S', time.localtime()).replace(' ','_')
        os.makedirs(config.log_dir, exist_ok=True)

    # Get data loader from kwargs if it exists
    if kwargs.get('train_loader', None) is not None:
        train_loader = kwargs.get('train_loader')
    else:
        train_loader, _ = get_train_loader(None, RGBXDataset, config)
    if kwargs.get('val_loader', None) is not None:
        val_loader = kwargs.get('val_loader')
    else:
        val_loader, _ = get_val_loader(None, RGBXDataset, config, 1)

    print('train_loader: ', len(train_loader), 'val_loader: ', len(val_loader), "config num_train_imgs: ", config.num_train_imgs)

    # Dont ignore the background class
    criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=config.background)
    BatchNorm2d = nn.BatchNorm2d
    
    base_lr = config.lr

    model = ModelWrapper(
        config=config,
        criterion=criterion,
        norm_layer=BatchNorm2d,
        pretrained=True,
    )

    ############################################ Optimizer ###########################################
    
    params_list = model.params_list
    
    if config.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params_list, lr=base_lr, betas=(0.9, 0.999), weight_decay=config.weight_decay)
    elif config.optimizer == 'SGDM':
        optimizer = torch.optim.SGD(params_list, lr=base_lr, momentum=config.momentum, weight_decay=config.weight_decay)
    else:
        raise NotImplementedError
    
    ############################################ Scheduler ###########################################

    config.niters_per_epoch = config.num_train_imgs // config.batch_size + 1
    if is_tuning:
        config.niters_per_epoch = config.num_train_imgs // config.batch_size
    total_iteration = config.nepochs * config.niters_per_epoch
    lr_policy = WarmUpPolyLR(base_lr, config.lr_power, total_iteration, config.niters_per_epoch * config.warm_up_epoch)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device: ',device, 'model name: ',config.backbone, 'batch size: ',config.batch_size)
    model.to(device)

    # Load pretrained model
    start_epoch = 1

    if config.pretrained_model is not None:
        checkpoint = torch.load(config.pretrained_model, map_location=device)
        if 'model' in checkpoint:
            print('loading pretrained model from %s' % config.pretrained_model)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch'] + 1
            current_idx = checkpoint['iteration']
            print('starting from epoch: ', start_epoch, 'iteration: ', current_idx)

    optimizer.zero_grad()
    best_miou=0.0

    for epoch in range(start_epoch, config.nepochs+1):
        model.train()
        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
        pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout,
                        bar_format=bar_format)
        dataloader = iter(train_loader)

        sum_loss = 0
        i=0
        for idx in pbar:
            minibatch = next(dataloader)
            imgs = minibatch['data']
            gts = minibatch['label']
            modal_xs = minibatch['modal_x']

            imgs = imgs.to(device)
            gts = gts.to(device)
            modal_xs = modal_xs.to(device)
    
            output = model(imgs, modal_xs)

            loss = model.get_loss(output, gts, criterion)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            current_idx = (epoch- 1) * config.niters_per_epoch + idx 
            lr = lr_policy.get_lr(current_idx)

            for i in range(len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = lr

            sum_loss += loss.item()

            if (epoch % config.checkpoint_step == 0 and epoch > int(config.checkpoint_start_epoch) or epoch == 1) and idx == 0:
                tb_summary.visualize_image(
                    writer=tb,
                    images=imgs,
                    depth=modal_xs,
                    target=gts,
                    output=output,
                    global_step=epoch,
                    config=config
                )

            print_str = 'Epoch {}/{}'.format(epoch, config.nepochs) \
                    + ' Iter {}/{}:'.format(idx + 1, config.niters_per_epoch) \
                    + ' lr=%.4e' % lr \
                    + ' loss=%.4f total_loss=%.4f' % (loss, (sum_loss / (idx + 1)))
            pbar.set_description(print_str)

            del imgs, gts, modal_xs, loss
        
        tb.add_scalar('train/loss', sum_loss / len(pbar), epoch)
        tb.add_scalar('train/lr', lr, epoch)

        if (epoch % config.checkpoint_step == 0 and epoch > int(config.checkpoint_start_epoch)) or epoch == config.nepochs or epoch == 1:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            with torch.no_grad():
                model.eval()
                metrics = evaluate_with_loader(model, val_loader, config, device, bin_size=float('inf'))
                miou = metrics.miou
                mious = [miou]
                print('macc: ', metrics.macc, 'mf1: ', metrics.mf1, 'miou: ', metrics.miou)

                if not is_tuning:
                    if miou > best_miou:
                        best_miou = miou
                        print('saving model...')
                        save_checkpoint(model, optimizer, epoch, current_idx, os.path.join(config.log_dir, f"epoch_{epoch}_miou_{miou}.pth"))
                    
                    save_results(config.log_dir + '/results.txt', metrics, 0)
                
                tb.add_scalar('val/macc', metrics.macc, epoch)
                tb.add_scalar('val/mf1', metrics.mf1, epoch)
                tb.add_scalar('val/miou', miou, epoch)

                del metrics

            ray_callback = kwargs.get('ray_callback', None)
            if ray_callback is not None:
                # Call the ray callback with miou and loss
                ray_callback(miou, sum_loss / len(pbar), epoch)

    return best_miou

def save_checkpoint(model, optimizer, epoch, iteration, path):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in model.state_dict().items():
        key = k
        if k.split('.')[0] == 'module':
            key = k[7:]
        new_state_dict[key] = v

    state_dict = {}
    state_dict['model'] = new_state_dict
    state_dict['optimizer'] = optimizer.state_dict()
    state_dict['epoch'] = epoch
    state_dict['iteration'] = iteration

    torch.save(state_dict, path)

def run_hyperparameters(config, file_path, num_samples, num_hyperparameter_epochs):
    best_config_dict = tune_hyperparameters(
        config, 
        num_samples=num_samples, 
        max_num_epochs=num_hyperparameter_epochs, 
        cpus_per_trial=4, 
        gpus_per_trial=1,
        train_callback=train_model_from_config
    )
    
    # Write best hyperparameters to file
    try:
        if not os.path.exists(config.log_dir):
            os.makedirs(config.log_dir, exist_ok=True)
        with open(file_path, 'a') as file:
            # Append time to best hyperparameters config
            best_config_dict["date_time"] = time.strftime('%Y%m%d-%H%M%S', time.localtime()).replace(' ','_')
            file.write(str(best_config_dict) + "\n")
    except Exception as e:
        print(f"Error occurred while writing to file: {e}")

    return best_config_dict

def train_model(config_location: str, 
                checkpoint_dir: str="checkpoints", 
                model: str=None, 
                dataset_classes: str="random", 
                num_hyperparameter_samples: int=20, 
                num_hyperparameter_epochs: int=5, 
                num_epochs: int=60, 
                x_channels: int=3, 
                x_e_channels: int=1,
                dataset_name: str=None,
                max_train_images: int=1e6, 
                gpus: int=1,
    ):

    with open(MODEL_CONFIGURATION_DICT_FILE, 'r') as f:
        model_configurations = json.load(f)

    if model is None:
        raise ValueError("Model not provided")
    
    model_configuration = model_configurations.get(model, None)
    if model_configuration is None:
        raise ValueError(f"Model {model} not found in model configurations")

    config_dict_to_update = {
        "checkpoint_dir": checkpoint_dir,
        "x_channels": x_channels,
        "x_e_channels": x_e_channels,
        "nepochs": num_epochs,
    }

    # Add model_configuration to config_dict_to_update
    config_dict_to_update.update(model_configuration)

    if dataset_name is not None:
        config_dict_to_update["dataset_name"] = dataset_name

    config = update_config(
        config_location, 
        config_dict_to_update
    )

    # make sure the number of training images is within the range
    config.num_train_imgs = max(1, min(config.num_train_imgs, max_train_images))

    file_path = os.path.join(config.log_dir, f"x_{x_channels}_x_e_{x_e_channels}_best_hyperparameters.txt")

    if num_hyperparameter_epochs == -1:
        try:
            with open(file_path, 'r') as file:
                # Read the file as a string
                data = file.readlines()[-1]

            # Replace single quotes with double quotes
            data = data.replace("'", '"')

            # Load the string as JSON
            hyperparameters = json.loads(data)

            config = update_config(
                config_location,
                hyperparameters
            )
            print(f"Using hyperparameters from file: {hyperparameters}")
        except Exception as e:
            print(f"Error while processing hyperparameters: {e}")
            print("Getting new hyperparameters")
            best_hyperparameter_dict = run_hyperparameters(config, file_path, num_hyperparameter_samples, num_hyperparameter_epochs)
    if num_hyperparameter_epochs > 0:
        best_hyperparameter_dict = run_hyperparameters(config, file_path, num_hyperparameter_samples, num_hyperparameter_epochs)

    config = update_config(
        config_location,
        best_hyperparameter_dict
    )

    best_miou = train_model_from_config(config)

    return best_miou, config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="DFormer.local_configs.SynthDet.SynthDet_black_back_default_2_Dformer_Tiny",
        help="The config to use for training the model",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="The model to use for training the model, choose DFormer, CMX or DeepLab",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="The directory to save the model checkpoints",
    )
    parser.add_argument(
        "--dataset_classes",
        type=str,
        default="groceries",
        help="The type of dataset to use for training",
    )
    parser.add_argument(
        "-hs", "--num_hyperparameter_samples",
        type=int,
        default=10,
        help="The number of samples to use for hyperparameter tuning",
    )
    parser.add_argument(
        "-he", "--num_hyperparameter_epochs",
        type=int,
        default=5,
        help="The number of epochs to use for hyperparameter tuning",
    )
    parser.add_argument(
        "-e", "--num_epochs",
        type=int,
        default=60,
        help="The number of epochs to use for hyperparameter tuning",
    )
    parser.add_argument(
        "--x_channels",
        type=int,
        default=3,
        help="The number of channels in the x modalities",
    )
    parser.add_argument(
        "--x_e_channels",
        type=int,
        default=1,
        help="The number of channels in the x_e modalities",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="The number of GPUs to use for training",
    )
    args = parser.parse_args()

    config_location = args.config

    best_miou, config = train_model(
        config_location=config_location,
        checkpoint_dir=args.checkpoint_dir,
        model=args.model,
        dataset_classes=args.dataset_classes,
        num_hyperparameter_samples=args.num_hyperparameter_samples,
        num_hyperparameter_epochs=args.num_hyperparameter_epochs,
        num_epochs=args.num_epochs,
        x_channels=args.x_channels,
        x_e_channels=args.x_e_channels,
        gpus=args.gpus,
    )

    print(f"Best mIoU: {best_miou}")