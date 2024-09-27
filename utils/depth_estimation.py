import torch
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
import sys
import argparse

from transformers import pipeline

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

def predict_dataset(
        dataset_location,
        replace_file_name_string=[None, None]
    ):

    # device = 0 if torch.cuda.is_available() else -1
    device = torch.cuda.current_device() if torch.cuda.is_available() else -1
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")


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

    print("Predictions saved with name: ", replace_file_name_string[1])


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-d", "--dataset_location", type=str, required=True)
    argparser.add_argument("-rfs", "--replace_file_name_string", type=str, nargs=2, default=[None, None], required=True)
    args = argparser.parse_args()

    predict_dataset(
        args.dataset_location, 
        args.replace_file_name_string,
    )