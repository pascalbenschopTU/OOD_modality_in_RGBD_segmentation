import torch
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
import sys
import argparse

from transformers import pipeline

def predict_dataset(
        dataset_location,
        result_folder_name=None, 
        replace_file_name_string=[None, None],
        recursive=False
    ):

    if result_folder_name is not None:
        result_folder = os.path.join(dataset_location, result_folder_name)
        os.makedirs(result_folder, exist_ok=True)
    else:
        result_folder = dataset_location

    device = 0 if torch.cuda.is_available() else -1


    if recursive:
        for root, _, files in os.walk(dataset_location):
            for filename in tqdm(files):
                if filename.endswith(".png") and "depth" not in filename:
                    image = Image.open(os.path.join(root, filename))
                    rgb = np.array(image)
                    rgb_torch = torch.from_numpy(rgb).permute(2, 0, 1)[:3]

                    pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Large-hf", device=device)
                    depth_pred = pipe(image)["depth"]

                    if replace_file_name_string is not [None, None]:
                        output_root = root.replace(replace_file_name_string[0], replace_file_name_string[1])
                        filename = filename.replace(replace_file_name_string[0], replace_file_name_string[1])
                    else:
                        output_root = root

                    # Create directories recursively along the path up until the actual filename
                    output_path = os.path.dirname(os.path.join(output_root, filename))
                    os.makedirs(output_path, exist_ok=True)
                    
                    depth_pred.save(os.path.join(output_root, filename))
    else:
        for file in tqdm(os.listdir(dataset_location)):
            rgb = np.array(Image.open(os.path.join(dataset_location, file)))
            rgb_torch = torch.from_numpy(rgb).permute(2, 0, 1)[:3]

            # predict
            predictions = model.infer(rgb_torch)

            # get GT and pred
            depth_pred = predictions["depth"].squeeze().cpu().numpy()
            depth_pred = (depth_pred - depth_pred.min()) / (depth_pred.max() - depth_pred.min())
            depth_pred = (depth_pred * 255).astype(np.uint8)
            depth_pred = Image.fromarray(depth_pred)

            filename = os.path.join(result_folder, file)
            if replace_file_name_string is not [None, None]:
                filename = filename.replace(replace_file_name_string[0], replace_file_name_string[1])
            depth_pred.save(filename)

    print("Predictions saved in", result_folder)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-d", "--dataset_location", type=str, required=True)
    argparser.add_argument("-r", "--result_folder_name", type=str, default=None)
    argparser.add_argument("-rfs", "--replace_file_name_string", type=str, nargs=2, default=[None, None])
    argparser.add_argument("-rec", "--recursive", action="store_true", default=False)
    args = argparser.parse_args()

    predict_dataset(
        args.dataset_location, 
        args.result_folder_name, 
        args.replace_file_name_string,
        args.recursive
    )