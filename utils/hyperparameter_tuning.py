from functools import partial
import os
import sys
from torch.utils.data import DataLoader
import torch
import numpy as np
from functools import partial

sys.path.append('../UsefullnessOfDepth')

# Dataset
from utils.dataloader.dataloader import ValPre
from utils.dataloader.RGBXDataset import RGBXDataset

# Ray imports for hyperparameter tuning
import ray
from ray import tune
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler
from ray import train
from ray.air.integrations.wandb import WandbLoggerCallback


# https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html
# https://docs.ray.io/en/latest/tune/examples/hpo-frameworks.html
# https://docs.ray.io/en/latest/tune/api_docs/suggestion.html#tune-suggest-optuna

def ray_callback(miou, loss, epoch):
    train.report({"miou": miou, "loss": loss, "epoch": epoch})

def update_config_with_hyperparameters(
        hyperparameters, 
        config, 
        num_epochs, 
        train_loader_length, 
        val_loader_length
    ):
    # Set hyperparameters
    config.lr = hyperparameters["lr"]
    config.lr_power = hyperparameters["lr_power"]
    config.momentum = hyperparameters["momentum"]
    config.weight_decay = hyperparameters["weight_decay"]
    config.batch_size = hyperparameters["batch_size"]
    config.nepochs = num_epochs
    config.warm_up_epoch = 1
    config.checkpoint_step = 1
    config.checkpoint_start_epoch = 0
    config.num_train_imgs = train_loader_length * config.batch_size
    config.num_val_imgs = val_loader_length * config.batch_size

    return config

def get_dataloaders(train_dataset, batch_size, dataset_split=0.1, train_split=0.8):
    # Create a subset of the dataset for faster training
    subset_length = int(dataset_split * len(train_dataset))
    train_dataset_subset, _ = torch.utils.data.random_split(
        train_dataset,
        [subset_length, len(train_dataset) - subset_length]
    )

    # Split the subset into training and validation sets
    train_length = int(train_split * len(train_dataset_subset))
    val_length = len(train_dataset_subset) - train_length
    train_dataset_split, val_dataset_split = torch.utils.data.random_split(
        train_dataset_subset, 
        [train_length, val_length]
    )

    # Create DataLoaders for the training and validation sets
    train_loader = DataLoader(train_dataset_split, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset_split, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

def train_setup(
        hyperparameters,
        config, 
        train_dataset,
        num_epochs=5, 
        train_callback=None
    ):
    train_loader, val_loader = get_dataloaders(train_dataset, config.batch_size)

    config = update_config_with_hyperparameters(hyperparameters, config, num_epochs, len(train_loader), len(val_loader))
    kwargs = {
        "is_tuning": True,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "ray_callback": ray_callback,
    }

    train_callback(config, **kwargs)


def shorten_trial_dirname_creator(trial, experiment_name="empty"):
    # Extract relevant trial information and create a shortened directory name
    short_trial_name = f"{experiment_name}_{trial.trial_id}"
    return short_trial_name

def update_config_paths(config):
    # Make all paths in config absolute
    config.dataset_path = os.path.abspath(config.dataset_path)
    config.rgb_root_folder = os.path.abspath(config.rgb_root_folder)
    config.gt_root_folder = os.path.abspath(config.gt_root_folder)
    config.x_root_folder = os.path.abspath(config.x_root_folder)
    config.log_dir = os.path.abspath(config.log_dir)
    config.tb_dir = os.path.abspath(config.tb_dir)
    config.checkpoint_dir = os.path.abspath(config.checkpoint_dir)
    config.train_source = os.path.abspath(config.train_source)
    config.eval_source = os.path.abspath(config.eval_source)
    if config.pretrained_model is not None:
        config.pretrained_model = os.path.abspath(config.pretrained_model)
    
    return config

def tune_hyperparameters(
        config, 
        train_dataset,
        num_samples=20, 
        max_num_epochs=5, 
        cpus_per_trial=4, 
        gpus_per_trial=1, 
        train_callback=None
    ):
    config = update_config_paths(config)

    experiment_name = f"{config.dataset_name}_{config.backbone}"

    param_space = {
        "lr": tune.loguniform(1e-5, 1e-3),
        "batch_size": tune.choice([16]),
        "lr_power": tune.uniform(0.8, 1.0),
        "momentum": tune.uniform(0.9, 0.99),
        "weight_decay": tune.loguniform(1e-4, 1e-2),
    }

    model = config.get("model", None)
    large_models = ["mit_b2", "xception", "mit_b3", "DFormer-Base", "TokenFusion", "Gemini", "CMX", "HIDANet"]
    if config.backbone in large_models or model in large_models:
        param_space["batch_size"] = tune.choice([4])
    extra_large_models = ["DFormer-Large"]
    if config.backbone in extra_large_models:
        param_space["batch_size"] = tune.choice([4])

    algorithm = OptunaSearch(
        metric="miou",
        mode="max",
    )

    scheduler = ASHAScheduler(
        time_attr="training_iteration",
        metric="miou",
        mode="max",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2,
    )

    ray.init()

   

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(
                train_setup, 
                config=config,
                train_dataset=train_dataset,
                num_epochs=max_num_epochs,
                train_callback=train_callback,                
            ),
            resources={"cpu": cpus_per_trial, "gpu": gpus_per_trial},
        ),
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            search_alg=algorithm,
            trial_dirname_creator=partial(shorten_trial_dirname_creator, experiment_name=experiment_name),
            num_samples=num_samples,
        ),
        run_config=train.RunConfig(
            stop={"training_iteration": max_num_epochs},
        ),
        param_space=param_space,
    )

    results = tuner.fit()
    best_config = results.get_best_result(
        metric="miou",
        mode="max",    
    ).config
    print("Best hyperparameters found were: ", best_config)

    ray.shutdown()

    return best_config
