# Enhancing RGB-D Segmentation in the Presence of Out-of-Distribution Modalities


For macbook
conda env create -f environment_mac.yml

pip install -r requirements.txt


## Functionality

Show depth estimation
```
python .\utils\Show_depth_estimation.py
```

Test specific methods on NYUDepthV2
```
python utils/train_day_night.py --hflip --rc --jitter 0.3 --scale 0.3 --batch-size 6 --pretrained --invariant 'W' --mode adapt --dataset nyu
```


```
python utils/train_day_night.py --hflip --rc --jitter 0.3 --scale 0.3 --batch-size 6 --pretrained --invariant 'W' --mode train --dataset cityscapes

python utils/train_day_night.py --hflip --rc --jitter 0.3 --scale 0.3 --batch-size 6 --pretrained --invariant 'W' --mode test --dataset cityscapes
```

## Depth estimation

python .\utils\depth_estimation.py -d .\datasets\Cityscapes\leftImg8bit\ -rfs leftImg8bit depth
python .\utils\depth_estimation.py -d .\datasets\NighttimeDrivingTest\leftImg8bit\ -rfs leftImg8bit depth
python .\utils\depth_estimation.py -d .\datasets\Dark_Zurich_val_anon\rgb_anon\ -rfs rgb_anon depth


## MMSegmentation

pip install xformers==0.0.22.post4 --index-url https://download.pytorch.org/whl/cu118

Follow instructions from:
https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/get_started.md#installation
https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/4_train_test.md 

Follow instructions from https://github.com/LiheYoung/Depth-Anything/tree/main/semseg 

Download: https://github.com/LiheYoung/Depth-Anything/tree/main/torchhub/

For resuming:

```
~/anaconda3/envs/<env>/lib/python3.11/site-packages/mmengine/runner/loops.py
and comment out the following lines:
# if self._iter > 0:
# print_log(
# f'Advance dataloader {self._iter} steps to skip data '
# 'that has already been trained',
# logger='current',
# level=logging.WARNING)
# for _ in range(self._iter):
# next(self.dataloader_iterator)
```

To run:
```
python .\train.py .\depth_anything_small_mask2former_16xb1_80k_cityscapes_896x896_freeze.py --work-dir depth_anything_depth
```