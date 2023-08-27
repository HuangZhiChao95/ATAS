from __future__ import print_function

import random

import torch
import torch.nn.functional as F


def aug(input_tensor):
    batch_size = input_tensor.shape[0]
    x = torch.zeros(batch_size)
    y = torch.zeros(batch_size)
    flip = [False] * batch_size
    rst = torch.zeros((len(input_tensor), 3, 32, 32), dtype=torch.float32, device=input_tensor.device)
    for i in range(batch_size):
        flip_t = bool(random.getrandbits(1))
        x_t = random.randint(0, 8)
        y_t = random.randint(0, 8)

        rst[i, :, :, :] = input_tensor[i, :, x_t:x_t + 32, y_t:y_t + 32]
        if flip_t:
            rst[i] = torch.flip(rst[i], [2])
        flip[i] = flip_t
        x[i] = x_t
        y[i] = y_t

    return rst, {"crop": {'x': x, 'y': y}, "flipped": flip}


def aug_trans(input_tensor, transform_info):
    batch_size = input_tensor.shape[0]
    x = transform_info['crop']['x']
    y = transform_info['crop']['y']
    flip = transform_info['flipped']
    rst = torch.zeros((len(input_tensor), 3, 32, 32), dtype=torch.float32, device=input_tensor.device)

    for i in range(batch_size):
        flip_t = int(flip[i])
        x_t = int(x[i])
        y_t = int(y[i])
        rst[i, :, :, :] = input_tensor[i, :, x_t:x_t + 32, y_t:y_t + 32]
        if flip_t:
            rst[i] = torch.flip(rst[i], [2])
    return rst


def inverse_aug(source_tensor, adv_tensor, transform_info):
    x = transform_info['crop']['x']
    y = transform_info['crop']['y']
    flipped = transform_info['flipped']
    batch_size = source_tensor.shape[0]

    for i in range(batch_size):
        flip_t = int(flipped[i])
        x_t = int(x[i])
        y_t = int(y[i])
        if flip_t:
            adv_tensor[i] = torch.flip(adv_tensor[i], [2])
        source_tensor[i, :, x_t:x_t + 32, y_t:y_t + 32] = adv_tensor[i]

    return source_tensor


def aug_imagenet(input_tensor):
    input_tensor = F.interpolate(input_tensor, (256, 256), mode='bilinear')
    batch_size = input_tensor.shape[0]
    x = torch.zeros(batch_size)
    y = torch.zeros(batch_size)
    flip = [False] * batch_size
    rst = torch.zeros((len(input_tensor), 3, 224, 224), dtype=torch.float32, device=input_tensor.device)

    for i in range(batch_size):
        flip_t = bool(random.getrandbits(1))
        x_t = random.randint(0, 32)
        y_t = random.randint(0, 32)

        rst[i, :, :, :] = input_tensor[i, :, x_t:x_t + 224, y_t:y_t + 224]
        if flip_t:
            rst[i] = torch.flip(rst[i], [2])
        flip[i] = flip_t
        x[i] = x_t
        y[i] = y_t

    return rst, {"crop": {'x': x, 'y': y}, "flipped": flip}


def aug_trans_imagenet(input_tensor, transform_info):
    batch_size = input_tensor.shape[0]
    x = transform_info['crop']['x']
    y = transform_info['crop']['y']
    flip = transform_info['flipped']
    rst = torch.zeros((len(input_tensor), 3, 224, 224), dtype=torch.float32, device=input_tensor.device)

    for i in range(batch_size):
        flip_t = int(flip[i])
        x_t = int(x[i])
        y_t = int(y[i])
        rst[i, :, :, :] = input_tensor[i, :, x_t:x_t + 224, y_t:y_t + 224]
        if flip_t:
            rst[i] = torch.flip(rst[i], [2])
    return rst


def inverse_aug_imagenet(source_tensor, adv_tensor, transform_info):
    interpolate_tensor = F.interpolate(source_tensor, (256, 256), mode='bilinear')
    x = transform_info['crop']['x']
    y = transform_info['crop']['y']
    flipped = transform_info['flipped']
    batch_size = source_tensor.shape[0]

    for i in range(batch_size):
        flip_t = int(flipped[i])
        x_t = int(x[i])
        y_t = int(y[i])
        if flip_t:
            adv_tensor[i] = torch.flip(adv_tensor[i], [2])
        interpolate_tensor[i, :, x_t:x_t + 224, y_t:y_t + 224] = adv_tensor[i]

    return F.interpolate(source_tensor, (32, 32), mode='bilinear')
