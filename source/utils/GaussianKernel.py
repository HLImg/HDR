# -*- encoding: utf-8 -*-
# @Author  :  Liang Hao
# @Time    :  2023/05/26 13:22:16
# @FileName:  GaussianKernel.py
# @Contact :  lianghao@whu.edu.cn

import math
import numpy as np

class GaussianKernel2D:
    def gaussian(self, x, y, sigma):
        sigma2 = sigma ** 2
        return math.exp( - (x ** 2 + y ** 2) / (2 * sigma2)) / (2 * math.pi * sigma2)
    
    @staticmethod
    def gaussian_kernel(ksize, sigma):
        kernel = []
        center = ksize // 2 # the center position of gaussian kernel
        for i in range(ksize):
            row = []
            for j in range(ksize):
                x = i - center
                y = j - center
                row.append(GaussianKernel2D.gaussian(x, y, sigma))
            kernel.append(row)
        return np.array(kernel)
        