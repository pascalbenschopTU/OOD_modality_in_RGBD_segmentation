import torchvision.transforms.functional as F
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import importlib
import sys
import torch
# Import partial
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed

from evaluate_models import evaluate_with_loader


def adapt_dataset(origin_directory_path, destination_directory_path, property_value, adaptation_method):
    # Walk through the source directory
    paths = []
    destination_paths = []

    for root, _, files in os.walk(origin_directory_path):
        for file in files:
            # Construct full file path
            file_path = os.path.join(root, file)
            paths.append(file_path)

            destination_path = os.path.join(destination_directory_path, os.path.relpath(file_path, origin_directory_path))
            destination_paths.append(destination_path)

    paths.sort()
    destination_paths.sort()

    # Initialize the progress bar
    progress_bar = tqdm(total=len(paths), desc=f'Processing files in {origin_directory_path}')
    print("Property value: ", property_value)

    # for path, destination_path in zip(paths, destination_paths):
    #     adaptation_method(path, destination_path, property_value)
    #     progress_bar.update(1)

    def process_file(path, destination_path):
        adaptation_method(path, destination_path, property_value)
        progress_bar.update(1)

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_file, path, destination_path) for path, destination_path in zip(paths, destination_paths)]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"An error occurred: {e}")

    # Close the progress bar
    progress_bar.close()


def adjust_saturation(image_path, destination_path, saturation_value):
    image = Image.open(image_path)
    saturated_image = F.adjust_saturation(image, saturation_value)

    # Save the image
    saturated_image.save(destination_path)

    return saturation_value

def adjust_hue(image_path, destination_path, hue):
    image = Image.open(image_path)
    color_image = F.adjust_hue(image, hue)

    # Save the image
    color_image.save(destination_path)

    return hue

def adjust_brightness(image_path, destination_path, brightness_factor):
    image = Image.open(image_path)
    brightened_image = F.adjust_brightness(image, brightness_factor)

    # Save the image
    brightened_image.save(destination_path)

    return brightness_factor

def adjust_noise(image_path, destination_path, noise_factor):
    image = Image.open(image_path)
    noisy_image = np.array(image)
    if np.max(noisy_image) > 1 and noise_factor <= 1:
        noise_factor *= 255
    noise = np.random.normal(0, noise_factor, noisy_image.shape)
    noisy_image = np.clip(noisy_image + noise, 0, 255)

    noisy_image = Image.fromarray(noisy_image.astype(np.uint8))

    # Save the image
    noisy_image.save(destination_path)

    return noise_factor

def adjust_depth_level(image_path, destination_path, depth_level, depth_range=0.1):
    depth_image = Image.open(image_path)
    depth_image = np.array(depth_image)
    min_depth = np.min(depth_image)
    max_depth = np.max(depth_image)

    # Normalize to [0, 1]
    normalized_depth_image = (depth_image - min_depth) / (max_depth - min_depth)

    # Scale to the desired depth level
    min_depth = 0
    max_depth = 1 - depth_range
    scaled_depth_image = normalized_depth_image * depth_range + min(max_depth, max(min_depth, depth_level))

    depth_image = np.clip(scaled_depth_image * 255, 0, 255)

    # Convert to uint8
    depth_image = depth_image.astype(np.uint8)

    depth_image = Image.fromarray(depth_image)
    depth_image = depth_image.convert("L")

    # Save the image
    depth_image.save(destination_path)

    return depth_level


def adapt_property(origin_directory_path, destination_directory_path, property_value, property_name, split, depth_range=0.1):
    if property_name == "saturation":
        adapt_dataset(origin_directory_path, destination_directory_path, property_value, adjust_saturation)
    if property_name == "hue":
        adapt_dataset(origin_directory_path, destination_directory_path, property_value, adjust_hue)
    if property_name == "brightness":
        adapt_dataset(origin_directory_path, destination_directory_path, property_value, adjust_brightness)
    if property_name == "noise":
        adapt_dataset(origin_directory_path, destination_directory_path, property_value, adjust_noise)
    if property_name == "depth_level":
        adjust_depth_level_func = partial(adjust_depth_level, depth_range=depth_range)
        adapt_dataset(origin_directory_path, destination_directory_path, property_value, adjust_depth_level_func)


def test_property_shift(
        config, 
        property_values, 
        model, 
        dataloader,
        property_name, 
        origin_directory_path,
        destination_directory_path, 
        split="test", 
        device="cuda",
    ):
    results_file_name = f"{property_name}_tests.txt"
    with open(results_file_name, "a") as result_file:
        result_file.write(f"Property: {property_name}\n")
        result_file.write(f"Property values: {property_values}\n")
        result_file.write(f"Config: {config}\n")
        result_file.write("\n")

    miou_values = []

    for property_value in property_values:
        adapt_property(origin_directory_path, destination_directory_path, property_value, property_name, split)

        model.to('cuda')
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        
        torch.cuda.empty_cache()
        model.eval()

        with torch.no_grad():
            evaluator = evaluate_with_loader(
                model=model,
                dataloader=dataloader,
                config=config,
                device='cuda',
                bin_size=float('inf'),
            )

            print("val miou: ", evaluator.miou)

            with open(results_file_name, "a") as f:
                f.write(f"validation results: {str(evaluator.miou)} \n")

            miou_values.append(evaluator.miou)

    return property_values, miou_values

    


# if __name__ == "__main__":
#     argparser = argparse.ArgumentParser()
#     argparser.add_argument("-op", "--origin_directory_path", type=str, default="datasets/SUNRGBD/RGB_original", help="The path to the SUNRGBD dataset")
#     argparser.add_argument("-dp", "--destination_directory_path", type=str, default="datasets/SUNRGBD/RGB", help="The path to save the adapted dataset")
#     argparser.add_argument("-s", "--split", type=str, default="test", help="The split to adapt")
#     argparser.add_argument("-cfg", '--config', help='train config file path')
#     argparser.add_argument("-mw", '--model_weights', help='File of model weights')
#     argparser.add_argument("-m", '--model', help='Model name', default='DFormer')
#     argparser.add_argument("-bs", '--bin_size', help='Bin size for testing', default=1, type=int)
#     argparser.add_argument("-pname", '--property_name', help='Property name', default='saturation')
#     argparser.add_argument("-pmin", '--min_property_value', help='Minimum property value', default=-1.0, type=float)
#     argparser.add_argument("-pmax", '--max_property_value', help='Maximum property value', default=-1.0, type=float)
#     argparser.add_argument("-pvr", '--property_value_range', help='Property value range', default=10, type=int)
#     argparser.add_argument("-dr", "--depth_range", help="Range of depth", default=0.1, type=float)
#     argparser.add_argument("-xch", "--x_channels", help="Number of input channels", default=3, type=int)
#     argparser.add_argument("-xech", "--x_e_channels", help="Number of input channels", default=1, type=int)

#     args = argparser.parse_args()

#     module_name = args.config
#     if ".py" in module_name:
#         module_name = module_name.replace(".py", "")
#         module_name = module_name.replace("\\", ".")
#         while module_name.startswith("."):
#             module_name = module_name[1:]

#     args.config = module_name

#     if args.config is not None:
#         property_values = np.linspace(args.min_property_value, args.max_property_value, args.property_value_range)
#         if args.property_name == "saturation" and args.min_property_value == -1.0 and args.max_property_value == -1.0:
#             property_values = np.linspace(0.001, 2.0, 11)
#         if args.property_name == "hue" and args.min_property_value == -1.0 and args.max_property_value == -1.0:
#             property_values = np.linspace(-0.5, 0.5, 11)
#         if args.property_name == "brightness" and args.min_property_value == -1.0 and args.max_property_value == -1.0:
#             property_values = np.linspace(0.001, 2.0, 11)
#         if args.property_name == "depth_level" and args.min_property_value == -1.0 and args.max_property_value == -1.0:
#             property_values = np.linspace(0.0, 0.9, 10)

#         if args.split == "empty":
#             args.split = ""

#         test_property_shift(
#             config=args.config, 
#             property_values=property_values, 
#             model=None, 
#             property_name=args.property_name, 
#             origin_directory_path=args.origin_directory_path, 
#             destination_directory_path=args.destination_directory_path, 
#             model=args.model, 
#             split=args.split, 
#             device="cuda",
#             args=args
#         )

        
    