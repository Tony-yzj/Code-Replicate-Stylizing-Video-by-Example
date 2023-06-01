import cv2
import numpy as np

#  使用LumaKeying对G通道抠图
def LumaKeying(src, threshold0, threshold1, C):
    # 使用G通道
    channels = cv2.split(src)
    # 初始化mask
    mask = np.zeros(src.shape[:2], dtype=np.uint8)
    for y in range(channels[1].shape[0]):
        for x in range(channels[1].shape[1]):
            L = channels[1][y, x]
            d = min(abs(L - threshold0), abs(L - threshold1))
            if L > threshold0 and L < threshold1:  # 在阈值之间，说明是背景
                mask[y, x] = 255
            elif d > C:  # 大于C说明一定是前景
                mask[y, x] = 0
            else:  # 软化边缘
                mask[y, x] = int(d * 255.0 / C)
    return mask

def color_difference_keying(src, threshold0, threshold1):
    channels = cv2.split(src)
    # Initialize mask
    mask = 255 * np.ones_like(channels[1], dtype=np.uint8)
    for y in range(channels[1].shape[0]):
        for x in range(channels[1].shape[1]):
            B = channels[0][y, x]
            G = channels[1][y, x]
            R = channels[2][y, x]
            d = G - max(R, B)
            if d > threshold0: # Greater than T0 means background
                mask[y, x] = 0
            elif threshold0 - d > threshold1: # Means foreground
                mask[y, x] = 255
            else: # Soften edges
                mask[y, x] = int((threshold0 - d) * 255.0 / threshold1)
    return mask


import cv2

def threeDKeying(src, threshold0, threshold1):
    # 定义原图像的四个样本点坐标
    samples = [(100, 100), (src.shape[1] - 100, src.shape[0] - 100), (100, src.shape[0] - 100), (src.shape[1] - 100, 100)]

    # 分离三个 RGB 通道
    channels = cv2.split(src)

    # 初始化 mask
    mask = np.zeros(src.shape[:2], dtype=np.uint8)

    for y in range(channels[1].shape[0]):
        for x in range(channels[1].shape[1]):
            B = channels[0][y, x]
            G = channels[1][y, x]
            R = channels[2][y, x]

            d = 1e9
            # 计算与四个样本点的最短空间距离
            for k in range(4):
                dis = 0
                # 计算距离
                for c in range(3):
                    delta = src[samples[k][1], samples[k][0]][c] - channels[c][y, x]
                    dis += delta * delta
                d = min(d, dis)

            if d < threshold0:  # 与样本点的距离小于阈值T0，说明是背景
                mask[y, x] = 0
            elif d > threshold1:  # 与样本点的距离大于阈值T0，说明是前景
                mask[y, x] = 255
            else:  # 软化边缘
                mask[y, x] = int((d - threshold0) * 255.0 / (threshold1 - threshold0))

    return mask


