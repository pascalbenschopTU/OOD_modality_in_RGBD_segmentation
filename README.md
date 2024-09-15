# Enhancing RGB-D Segmentation in the Presence of Out-of-Distribution Modalities


## Functionality

Show depth estimation
```
python .\utils\Show_depth_estimation.py
```

Test specific methods on NYUDepthV2
```
python .\utils\train_day_night.py --hflip --rc --jitter 0.3 --scale 0.3 --batch-size 6 --pretrained --invariant 'W' --mode adapt --dataset nyu
```


```
python .\utils\train_day_night.py --hflip --rc --jitter 0.3 --scale 0.3 --batch-size 6 --pretrained --invariant 'W' --mode test --dataset cityscapes
```