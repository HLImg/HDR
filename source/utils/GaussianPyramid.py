# -*- encoding: utf-8 -*-
# @Author  :  Liang Hao
# @Time    :  2023/05/26 15:09:11
# @FileName:  GaussianPyramid.py
# @Contact :  lianghao@whu.edu.cn
# 高斯金字塔

import math
import cv2 as cv
import numpy as np

class Pyramid: 
    def gaussian(self, x, y, sigma):
        sigma2 = sigma ** 2
        return math.exp(- (x ** 2 + y ** 2) / (2 * sigma2)) / (2 * math.pi * sigma2)
    
    @staticmethod
    def gaussian_kernel(ksize, sigma, normalized=True):
        kernel = []
        center = ksize // 2
        for i in range(ksize):
            row = []
            for j in range(ksize):
                x = i - center
                y = j - center
                row.append(Pyramid.gaussian(x, y, sigma))
            kernel.append(row)
        kernel = np.array(kernel)
        if normalized:
            kernel = kernel / kernel.sum()
        return kernel
    
    @staticmethod
    def conv2d(image, kernel):
        """
        金字塔的卷积：将高斯滤波器的核与图像的每个像素相乘，然后求和
        灰度图像的卷积操作，且image已经归一化
        image.shape : (h, w)
        kerne : (window, window)，一般是奇数且核是归一化的
        """
        height, width = image.shape
        img_convolved = np.zeros_like(image)
        window = kernel.shape[0]
        radius  = window // 2
        img_pad = np.pad(image, pad_width=((radius, radius), (radius, radius)))
        for i in range(height):
            for j in range(width):
                x = i + radius
                y = j + radius
                s = img_pad[x - radius : x + radius + 1, y - radius : y + radius + 1]
                img_convolved[i, j] = np.sum(s * kernel)
        return img_convolved
    
    @staticmethod
    def downsample(image):
        """
        金字塔下采样，即去除图像的偶数行和列，得到缩小一半的图像
        """
        height, width = image.shape
        image_down = np.zeros(shape=(height // 2, width // 2), dtype=image.dtype)
        for x in range(height // 2):
            for y in range(width // 2):
                image_down[x, y] = image[2 * x + 1, 2 * y + 1]
        return image_down
    
    def gaussian_pyramid(self, image, levels=4, ksize=5, sigma=1, normalized=True):
        """构建高斯金字塔
        Args:
            image (_type_): 原始图像
            levels (int, optional): 金字塔的层数. Defaults to 4.
        """
        if normalized:
            image = image / 255.
        pyramid = []
        pyramid.append(image)
        kernel = self.gaussian_kernel(ksize, sigma, normalized)
        # 循环指定层数，或者直到图像尺寸小于高斯核的尺寸为止
        for _ in range(levels):
            if pyramid[-1].shape[0] < 5 or pyramid[-1].shape[1] < 5:
                break
            # 对当前层的图像进行高斯滤波和下采样操作，得到下一层的图像，并添加到列表中
            img_smoothed = self.conv2d(pyramid[-1], kernel)
            img_downsampled = self.downsample(img_smoothed)
            pyramid.append(img_downsampled)
        
        return pyramid