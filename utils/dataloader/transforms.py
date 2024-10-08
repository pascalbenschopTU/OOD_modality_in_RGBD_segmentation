import warnings
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.transforms.functional as TF

import cv2
import numpy as np
import numbers
import random
import collections


def get_2dshape(shape, *, zero=True):
    if not isinstance(shape, collections.abc.Iterable):
        shape = int(shape)
        shape = (shape, shape)
    else:
        h, w = map(int, shape)
        shape = (h, w)
    if zero:
        minv = 0
    else:
        minv = 1

    assert min(shape) >= minv, 'invalid shape: {}'.format(shape)
    return shape

def random_crop_pad_to_shape(img, crop_pos, crop_size, pad_label_value):
    h, w = img.shape[:2]
    start_crop_h, start_crop_w = crop_pos
    assert ((start_crop_h < h) and (start_crop_h >= 0))
    assert ((start_crop_w < w) and (start_crop_w >= 0))

    crop_size = get_2dshape(crop_size)
    crop_h, crop_w = crop_size

    img_crop = img[start_crop_h:start_crop_h + crop_h,
               start_crop_w:start_crop_w + crop_w, ...]

    img_, margin = pad_image_to_shape(img_crop, crop_size, cv2.BORDER_CONSTANT,
                                      pad_label_value)

    return img_, margin

def generate_random_crop_pos(ori_size, crop_size):
    ori_size = get_2dshape(ori_size)
    h, w = ori_size

    crop_size = get_2dshape(crop_size)
    crop_h, crop_w = crop_size

    pos_h, pos_w = 0, 0

    if h > crop_h:
        pos_h = random.randint(0, h - crop_h + 1)

    if w > crop_w:
        pos_w = random.randint(0, w - crop_w + 1)

    return pos_h, pos_w

def pad_image_to_shape(img, shape, border_mode, value):
    margin = np.zeros(4, np.uint32)
    shape = get_2dshape(shape)
    pad_height = shape[0] - img.shape[0] if shape[0] - img.shape[0] > 0 else 0
    pad_width = shape[1] - img.shape[1] if shape[1] - img.shape[1] > 0 else 0

    margin[0] = pad_height // 2
    margin[1] = pad_height // 2 + pad_height % 2
    margin[2] = pad_width // 2
    margin[3] = pad_width // 2 + pad_width % 2

    img = cv2.copyMakeBorder(img, margin[0], margin[1], margin[2], margin[3],
                             border_mode, value=value)

    return img, margin

def pad_image_size_to_multiples_of(img, multiple, pad_value):
    h, w = img.shape[:2]
    d = multiple

    def canonicalize(s):
        v = s // d
        return (v + (v * d != s)) * d

    th, tw = map(canonicalize, (h, w))

    return pad_image_to_shape(img, (th, tw), cv2.BORDER_CONSTANT, pad_value)

def resize_ensure_shortest_edge(img, edge_length,
                                interpolation_mode=cv2.INTER_LINEAR):
    assert isinstance(edge_length, int) and edge_length > 0, edge_length
    h, w = img.shape[:2]
    if h < w:
        ratio = float(edge_length) / h
        th, tw = edge_length, max(1, int(ratio * w))
    else:
        ratio = float(edge_length) / w
        th, tw = max(1, int(ratio * h)), edge_length
    img = cv2.resize(img, (tw, th), interpolation_mode)

    return img

def random_scale(img, gt, scales):
    scale = random.choice(scales)
    sh = int(img.shape[0] * scale)
    sw = int(img.shape[1] * scale)
    img = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_LINEAR)
    gt = cv2.resize(gt, (sw, sh), interpolation=cv2.INTER_NEAREST)
    
    return img, gt, scale

def random_scale_rgbx(img, gt, modal_x, scales):
    scale = random.choice(scales)
    sh = int(img.shape[0] * scale)
    sw = int(img.shape[1] * scale)
    img = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_LINEAR)
    gt = cv2.resize(gt, (sw, sh), interpolation=cv2.INTER_NEAREST)
    modal_x = cv2.resize(modal_x, (sw, sh), interpolation=cv2.INTER_LINEAR)

    return img, gt, modal_x, scale

def random_scale_with_length(img, gt, length):
    size = random.choice(length)
    sh = size
    sw = size
    img = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_LINEAR)
    gt = cv2.resize(gt, (sw, sh), interpolation=cv2.INTER_NEAREST)

    return img, gt, size

def random_mirror(img, gt):
    if random.random() >= 0.5:
        img = cv2.flip(img, 1)
        gt = cv2.flip(gt, 1)

    return img, gt,

def random_rotation(img, gt):
    angle = random.random() * 20 - 10
    h, w = img.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    img = cv2.warpAffine(img, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR)
    gt = cv2.warpAffine(gt, rotation_matrix, (w, h), flags=cv2.INTER_NEAREST)

    return img, gt

def random_gaussian_blur(img):
    gauss_size = random.choice([1, 3, 5, 7])
    if gauss_size > 1:
        # do the gaussian blur
        img = cv2.GaussianBlur(img, (gauss_size, gauss_size), 0)

    return img

def center_crop(img, shape):
    h, w = shape[0], shape[1]
    y = (img.shape[0] - h) // 2
    x = (img.shape[1] - w) // 2
    return img[y:y + h, x:x + w]

def random_crop(img, gt, size):
    if isinstance(size, numbers.Number):
        size = (int(size), int(size))
    else:
        size = size

    h, w = img.shape[:2]
    crop_h, crop_w = size[0], size[1]

    if h > crop_h:
        x = random.randint(0, h - crop_h + 1)
        img = img[x:x + crop_h, :, :]
        gt = gt[x:x + crop_h, :]

    if w > crop_w:
        x = random.randint(0, w - crop_w + 1)
        img = img[:, x:x + crop_w, :]
        gt = gt[:, x:x + crop_w]

    return img, gt

def normalize(img, mean, std):
    # pytorch pretrained model need the input range: 0-1
    img = img.astype(np.float64) / 255.0
    img = img - mean
    img = img / std
    return img


################################
### Data helpers
################################

def test_trans(image, depth_image, mask=None, target_size=(512,1024), mean=None, std=None):
    # Basic image pre-processing
    image = TF.resize(image, target_size, interpolation=1) # Resize, 1 for LANCZOS, 2 for BILINEAR
    depth_image = TF.resize(depth_image, target_size, interpolation=1) # Resize, 1 for LANCZOS, 2 for BILINEAR

    # From PIL to Tensor
    image = TF.to_tensor(image)
    depth_image = TF.to_tensor(depth_image)

    if mean and std:
        image = TF.normalize(image, mean, std)

    if mask:
        mask = TF.resize(mask, target_size, interpolation=0) # 0 for Image.NEAREST
        mask = np.array(mask, np.uint8) # PIL Image to numpy array
        mask = torch.from_numpy(mask) # Numpy array to tensor
    return image, depth_image, mask

def train_trans(image, depth_image, mask, target_size=(512,1024), crop_size=(384,768), jitter=0.3, scale=0.3, hflip=True, depth_aug=True):
    # Generate random parameters for augmentation
    bf = random.uniform(1-jitter,1+jitter)
    cf = random.uniform(1-jitter,1+jitter)
    sf = random.uniform(1-jitter,1+jitter)
    hf = random.uniform(-jitter,+jitter)
    scale_factor = random.uniform(1-scale,1+scale)
    pflip = random.randint(0,1) > 0.5

    # Resize
    image = TF.resize(image, target_size, interpolation=1) # Resize, 2 for Image.BILINEAR
    depth_image = TF.resize(depth_image, target_size, interpolation=1) # Resize, 2 for Image.BILINEAR
    mask = TF.resize(mask, target_size, interpolation=0) # Resize, 0 for Image.NEAREST

    image = TF.affine(image, 0, [0,0], scale_factor, [0,0])
    depth_image = TF.affine(depth_image, 0, [0,0], scale_factor, [0,0])
    mask = TF.affine(mask, 0, [0,0], scale_factor, [0,0])

    # Random cropping
    if crop_size:
        # From PIL to Tensor
        image = TF.to_tensor(image)
        depth_image = TF.to_tensor(depth_image)
        mask = TF.to_tensor(mask)
        h, w = target_size
        th, tw = crop_size # target size
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        image = image[:,i:i+th,j:j+tw]
        depth_image = depth_image[:,i:i+th,j:j+tw]
        mask = mask[:,i:i+th,j:j+tw]
        image = TF.to_pil_image(image)
        depth_image = TF.to_pil_image(depth_image)
        mask = TF.to_pil_image(mask[0,:,:])

    # Apply noise, blur, depth range augmentations
    if depth_aug:
        min_depth = 0
        max_depth = 255

        # {'occlusion_prob': 0.03899435843568351, 'noise_std': 0.0984376379069491}
        noise_std = 0.0984376379069491
        occlusion_prob = 0.03899435843568351
        depth_image = TF.to_tensor(depth_image)

        if np.random.rand() < occlusion_prob:
            mask = torch.rand_like(depth_image) < 0.1
            depth_image[mask] = 0

        noise = torch.randn_like(depth_image) * noise_std * max_depth
        noise = noise.clamp(0, 255).byte()
        depth_image = depth_image + noise
        depth_image = depth_image.clamp(0, 255).byte()

        depth_image = TF.to_pil_image(depth_image)

        
    # H-flip
    if pflip == True and hflip == True:
        image = TF.hflip(image)
        depth_image = TF.hflip(depth_image)
        mask = TF.hflip(mask)
    
    # Color jitter
    if jitter != 0:
        image = TF.adjust_brightness(image, bf)
        image = TF.adjust_contrast(image, cf)
        image = TF.adjust_saturation(image, sf)
        image = TF.adjust_hue(image, hf)

    # From PIL to Tensor
    image = TF.to_tensor(image)
    depth_image = TF.to_tensor(depth_image)

    # Convert ids to train_ids
    mask = np.array(mask, np.uint8) # PIL Image to numpy array
    mask = torch.from_numpy(mask) # Numpy array to tensor
        
    return image, depth_image, mask