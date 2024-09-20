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
python utils/train_day_night.py --hflip --rc --jitter 0.3 --scale 0.3 --batch-size 6 --pretrained --invariant 'W' --mode test --dataset cityscapes
```

## Depth estimation

python .\utils\depth_estimation.py -d .\datasets\Cityscapes\leftImg8bit\ -rfs leftImg8bit depth2 -rec
python .\utils\depth_estimation.py -d .\datasets\NighttimeDrivingTest\leftImg8bit\ -rfs leftImg8bit depth2 -rec
python .\utils\depth_estimation.py -d .\datasets\Dark_Zurich_val_anon\rgb_anon\ -rfs rgb_anon depth2 -rec