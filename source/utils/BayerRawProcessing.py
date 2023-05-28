# -*- encoding: utf-8 -*-
# @Author  :  Liang Hao
# @Time    :  2023/05/27 12:39:31
# @FileName:  BayerRawProcessing.py
# @Contact :  lianghao@whu.edu.cn

# generate tiles and reconstruction

import numpy as np

def isTypeInt(image):
    return image.dtype in [np.uint8, np.uint16, np.uint32, np.uint64, np.int8, np.int16, np.int32, np.int64, np.uint]

def downsample_bayer(image):
    R = image[0 :: 2, 0 ::2] # 红色通道，取偶数行和偶数列的像素
    G1 = image[0 :: 2, 1 :: 2] # 绿色通道1， 取偶数行和奇数列的像素
    G2 = image[1 :: 2, 0 :: 2] # 绿色通道2，取奇数行和偶数列的像素
    B = image[1 :: 2,  1 :: 2] # 蓝色通道，取奇数行和奇数列的像素
    if isTypeInt(image):
        return np.right_shift(R + G1 + G2 + B + 2, 2)
    else:
        return (R + G1 + G2 + B) * 0.25

def generate_tiles(image, tile_size, stride):
    if isinstance(tile_size, int):
        tile_size = (tile_size, tile_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    tiles = []
    h, w = image.shape
    tile_size_h, tile_size_w = tile_size
    stride_h, stride_w = stride
    for i in range(0, h - tile_size_h + 1, stride_h):
        row = []
        for j in range(0, w - tile_size_w + 1, stride_w):
            tile = image[i : i + tile_size_h, j : j + tile_size_w]
            row.append(tile)
        tiles.append(row)
    tiles = np.array(tiles)
    return tiles

def reconstruct_image(tiles, tile_size, stride, img_shape):
    if isinstance(tile_size, int):
        tile_size = (tile_size, tile_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(img_shape, int):
        img_shape = (img_shape, img_shape)        
    h, w = img_shape
    tile_size_h, tile_size_w = tile_size
    stride_h, stride_w = stride
    row = 0
    image = np.zeros(shape=img_shape, dtype=tiles.dtype)
    for i in range(0, h - tile_size_h + 1, stride_h):
        column = 0
        for j in range(0, w - tile_size_w + 1, stride_w):
            image[i : i + tile_size_h, j : j + tile_size_w] = tiles[row, column]
            column = column + 1
        row = row + 1
    return image

