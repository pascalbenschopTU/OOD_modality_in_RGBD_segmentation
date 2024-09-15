import torch
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
import sys
import argparse

def predict_dataset(
        dataset_location,
        result_folder_name=None, 
        replace_file_name_string=[None, None],
        recursive=False
    ):
    
    model = torch.hub.load(
        "lpiccinelli-eth/unidepth",
        "UniDepth",
        version="v1",
        # backbone="ViTL14",
        backbone="vitl14",
        pretrained=True,
        trust_repo=True,
        force_reload=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    model.eval()

    if result_folder_name is not None:
        result_folder = os.path.join(dataset_location, result_folder_name)
        os.makedirs(result_folder, exist_ok=True)
    else:
        result_folder = dataset_location

    if recursive:
        for root, _, files in os.walk(dataset_location):
            for filename in tqdm(files):
                if filename.endswith(".png") and "depth" not in filename:
                    rgb = np.array(Image.open(os.path.join(root, filename)))
                    rgb_torch = torch.from_numpy(rgb).permute(2, 0, 1)[:3]

                    # predict
                    predictions = model.infer(rgb_torch)

                    # get GT and pred
                    depth_pred = predictions["depth"].squeeze().cpu().numpy()
                    depth_pred = (depth_pred - depth_pred.min()) / (depth_pred.max() - depth_pred.min())
                    depth_pred = (depth_pred * 255).astype(np.uint8)
                    depth_pred = Image.fromarray(depth_pred)

                    if replace_file_name_string is not [None, None]:
                        filename = filename.replace(replace_file_name_string[0], replace_file_name_string[1])
                    
                    depth_pred.save(os.path.join(root, filename))
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