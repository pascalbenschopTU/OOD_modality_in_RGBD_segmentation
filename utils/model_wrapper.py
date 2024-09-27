from inspect import signature
from typing import Optional
from torch import nn
import torch.nn.functional as F
import torch
import sys
import numpy as np
import copy
import os
import json

# import the directory with the models "../UsefullnessOfDepth"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

# Models
from models.DFormer.builder import EncoderDecoder as DFormer
from models.refinenet import RefineNet
# from models_CMX.builder import EncoderDecoder as CMXmodel
# from model_pytorch_deeplab_xception.deeplab import DeepLab
# from models_segformer import SegFormer
# from model_TokenFusion.segformer import WeTr as TokenFusion
# from models_Gemini.segformer import WeTr as Gemini
# from model_HIDANet.model import HiDANet as HIDANet

from utils.init_func import group_weight
from utils.pyt_utils import calculate_ood_score

class ModelWrapper(nn.Module):
    def __init__(
            self,
            config, 
            criterion=nn.CrossEntropyLoss(reduction='mean'),
            norm_layer=nn.BatchNorm2d, 
            pretrained=True,
        ):
        super(ModelWrapper, self).__init__()
        self.config = config
        self.backbone = config.backbone
        self.criterion = criterion
        self.norm_layer = norm_layer
        self.pretrained = pretrained
        self.pretrained_weights = config.pretrained_model
        self.is_token_fusion = False

        if hasattr(self.config, "model"):
            self.model_name = self.config.model
        else:
            self.model_name = "DFormer_Tiny"
        self.set_model()

    def set_ood_scores(self, filename):
        with open(filename, "r") as f:
            data = json.load(f)

        self.mean_training_ood_score_depth = np.mean(data["depth_output"])
        self.mean_training_ood_score_rgb = np.mean(data["rgb_output"])
        self.mean_training_ood_score_ensemble = np.mean(data["rgbd_output"])

        self.std_dev_training_ood_score_depth = np.std(data["depth_output"])
        self.std_dev_training_ood_score_rgb = np.std(data["rgb_output"])
        self.std_dev_training_ood_score_ensemble = np.std(data["rgbd_output"])

        self.save_predictions = True

    def set_model(self):
        if self.model_name == "DFormer":
            self.model = DFormer(cfg=self.config, criterion=self.criterion, norm_layer=self.norm_layer)
            self.params_list = group_weight([], self.model, self.norm_layer, self.config.lr)
        
            # rgb_config = copy.deepcopy(self.config)
            # rgb_config.x_channels = 3
            # rgb_config.x_e_channels = 3
            # self.rgb_model = DFormer(cfg=rgb_config, criterion=self.criterion, norm_layer=self.norm_layer)

            # self.rgb_model = RefineNet(self.config.num_classes, pretrained=self.pretrained, invariant="W")
            # # self.rgb_model = RefineNet(self.config.num_classes, pretrained=self.pretrained, invariant=None)

            # depth_config = copy.deepcopy(self.config)
            # depth_config.x_channels = 1
            # depth_config.x_e_channels = 1
            # self.depth_model = DFormer(cfg=depth_config, criterion=self.criterion, norm_layer=self.norm_layer)

        elif self.model_name == "DeepLab":
            self.model = DeepLab(cfg=self.config, criterion=self.criterion, norm_layer=self.norm_layer)
            self.params_list = group_weight([], self.model, self.norm_layer, self.config.lr)
        elif self.model_name == "TokenFusion":
            self.is_token_fusion = True
            self.model = TokenFusion(cfg=self.config, pretrained=self.pretrained)
            self.params_list = self.model.get_param_groups(self.config.lr, self.config.weight_decay)
        elif self.model_name == "Gemini":
            self.is_token_fusion = True
            self.model = Gemini(cfg=self.config, pretrained=self.pretrained)
            self.params_list = self.model.get_param_groups(self.config.lr, self.config.weight_decay)
        elif self.model_name == "SegFormer":
            self.model = SegFormer(backbone=self.backbone, cfg=self.config, criterion=self.criterion)
            self.params_list = group_weight([], self.model, self.norm_layer, self.config.lr)
            if self.pretrained:
                self.model.init_pretrained(self.config.pretrained_model)
        elif self.model_name == "CMX":
            self.model = CMXmodel(cfg=self.config, criterion=self.criterion, norm_layer=self.norm_layer)
            self.params_list = group_weight([], self.model, self.norm_layer, self.config.lr)
        elif self.model_name == "HIDANet":
            self.model = HIDANet()
            self.params_list = group_weight([], self.model, self.norm_layer, self.config.lr)
        else:
            raise ValueError("Model not found: ", self.model_name)
        
        self.is_rgb_model = len(signature(self.model.forward).parameters) == 1
        print("Model: ", self.model_name, " BackBone: ", self.backbone, " Pretrained: ", self.pretrained, " RGB Model: ", self.is_rgb_model)
    
    # Assumes the model takes in RGB in the shape (B, 3, H, W) and depth in the shape (B, 1, H, W)
    # Where B is the batch size, H is the height and W is the width of the input images
    def forward(self, x, x_e):
        if self.model is None:
            raise ValueError("Model not found")
        
        if len(x.size()) == 3:
            x = x.unsqueeze(0)
        if len(x_e.size()) == 3:
            x_e = x_e.unsqueeze(0)

        output = None
        
        # Check if self.model has a forward function that accepts only x or also x_e
        if self.is_rgb_model:
            output = self.model(x)
        else:
            output = self.model(x, x_e)
        
        return output

        # Check how much inputs the model has
        output_rgb = self.rgb_model(x)
        output_depth = self.depth_model(x_e, x_e)

        # return output_rgb
        # return output_depth

        ood_scores_rgb = calculate_ood_score(output_rgb)
        ood_scores_depth = calculate_ood_score(output_depth)
        ood_scores_ensemble = calculate_ood_score(output)

        # print(f"ood_scores_rgb shape: {ood_scores_rgb.shape} ood_scores_depth shape: {ood_scores_depth.shape} ood_scores_ensemble shape: {ood_scores_ensemble.shape}")

        # # Assume you already calculated these
        # mean_ood_score_rgbd = ood_scores_ensemble.mean()  # for rgbd output
        # mean_ood_score_rgb = ood_scores_rgb.mean()    # for rgb output
        # mean_ood_score_depth = ood_scores_depth.mean()  # for depth output

        # print(f"mean_ood_score_rgbd: {mean_ood_score_rgbd} mean_ood_score_rgb: {mean_ood_score_rgb} mean_ood_score_depth: {mean_ood_score_depth}")

        mean_ood_score_rgbd = ood_scores_ensemble.mean(dim=(1, 2))  # for rgbd output
        mean_ood_score_rgb = ood_scores_rgb.mean(dim=(1, 2))    # for rgb output
        mean_ood_score_depth = ood_scores_depth.mean(dim=(1, 2)) # for depth output

        

        # print(f"other mean_ood_score_rgbd: {mean_ood_score_rgbd} mean_ood_score_rgb: {mean_ood_score_rgb} mean_ood_score_depth: {mean_ood_score_depth}")

        if hasattr(self, "training_data_filename"):
            self.save_ood_scores_to_file(
                mean_ood_score_depth=mean_ood_score_depth,
                mean_ood_score_rgb=mean_ood_score_rgb,
                mean_ood_score_rgbd=mean_ood_score_rgbd,
                filename=self.training_data_filename
            )

            return output
         
        final_output = torch.zeros_like(output)

        distance_rgb = abs(mean_ood_score_rgb - self.mean_training_ood_score_rgb)
        distance_depth = abs(mean_ood_score_depth - self.mean_training_ood_score_depth)
        distance_ensemble = abs(mean_ood_score_rgbd - self.mean_training_ood_score_ensemble)

        final_output = torch.zeros_like(output)

        for batch_index in range(output.shape[0]):
            ood_scores_rgb = calculate_ood_score(output_rgb[batch_index])
            ood_scores_depth = calculate_ood_score(output_depth[batch_index])
            ood_scores_ensemble = calculate_ood_score(output[batch_index])
        

            mean_ood_score_depth = ood_scores_depth.mean()
            mean_ood_score_rgb = ood_scores_rgb.mean()
            mean_ood_score_rgbd = ood_scores_ensemble.mean()

            distance_rgb = abs(mean_ood_score_rgb - self.mean_training_ood_score_rgb)
            distance_depth = abs(mean_ood_score_depth - self.mean_training_ood_score_depth)
            distance_ensemble = abs(mean_ood_score_rgbd - self.mean_training_ood_score_ensemble)


            distances = torch.tensor([distance_rgb.item(), distance_depth.item(), distance_ensemble.item()])
            weights = F.softmax(-distances, dim=0)

            weight_rgb = weights[0]
            weight_depth = weights[1]
            weight_ensemble = weights[2]

            final_output[batch_index] = weight_rgb * output_rgb[batch_index] + weight_depth * output_depth[batch_index] + weight_ensemble * output[batch_index]

            # Without rgb

            # distances = torch.tensor([distance_depth.item(), distance_ensemble.item()])
            # weights = F.softmax(-distances, dim=0)

            # weight_depth = weights[0]
            # weight_ensemble = weights[1]

            # final_output[batch_index] = weight_depth * output_depth[batch_index] + weight_ensemble * output[batch_index]

        return final_output
    

    def get_loss(self, output, target, criterion):
        # If the model has an auxillary loss, use it
        if self.config.get('use_aux', False):
            foreground_mask = target != self.config.background
            label = (foreground_mask > 0).long()
            loss = criterion(output[0], target.long()) + self.config.aux_rate * criterion(output[1], label)
            output = output[0]
        else:
            # If the model is (similar to) TokenFusion, the output is a list of outputs and masks
            if self.is_token_fusion and isinstance(output, list):  # Output of TokenFusion
                output, masks = output
                loss = 0
                for out in output:
                    soft_output = F.log_softmax(out, dim=1)
                    loss += criterion(soft_output, target)

                if self.config.lamda > 0 and masks is not None:
                    L1_loss = 0
                    for mask in masks:
                        L1_loss += sum([torch.abs(m).sum().cuda() for m in mask])
                    loss += self.config.lamda * L1_loss
            # HIDANet model has a special loss function
            elif self.model_name == "HIDANet":
                loss = self.model.calculate_loss(output, target)
            # For models that have multiple outputs
            elif isinstance(output, list) or isinstance(output, tuple):
                loss = 0
                for out in output:
                    soft_output = F.log_softmax(out, dim=1)
                    loss += criterion(soft_output, target)
            # The default loss function
            else:
                loss = criterion(output, target.long())

        return loss
    

    def save_ood_scores_to_file(
        self,
        mean_ood_score_rgbd: torch.Tensor,
        mean_ood_score_rgb: torch.Tensor,
        mean_ood_score_depth: torch.Tensor,
        filename: str
    ):
         # Prepare a dictionary with the keys and values
        ood_data = {
            "rgbd_output": mean_ood_score_rgbd.mean().item(),
            "rgb_output": mean_ood_score_rgb.mean().item(),
            "depth_output": mean_ood_score_depth.mean().item()
        }

        
        # Ensure the file exists, or create it with empty lists for each key
        if not os.path.exists(filename):
            with open(filename, "w") as f:
                # Initialize the file with empty lists for each key
                json.dump({
                    "rgbd_output": [],
                    "rgb_output": [],
                    "depth_output": []
                }, f)

        # Read the existing data from the JSON file
        with open(filename, "r") as f:
            data = json.load(f)

        # Append the new OOD scores to the corresponding lists
        data["rgbd_output"].append(ood_data["rgbd_output"])
        data["rgb_output"].append(ood_data["rgb_output"])
        data["depth_output"].append(ood_data["depth_output"])

        # Write the updated data back to the JSON file
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)
