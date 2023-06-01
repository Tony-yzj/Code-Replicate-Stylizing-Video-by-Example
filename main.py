import cv2
import Frame

# load the image
image = cv2.imread('src/demo.jpeg')
# mask = Frame.LumaKeying(image, threshold0=120, threshold1=200, C=20)
# cv2.imwrite('src/LumaKwying_1.jpeg', mask)
# mask = Frame.color_difference_keying(image, threshold0=70, threshold1=20)
# cv2.imwrite('src/LumaKwying_2.jpeg', mask)
# mask = Frame.threeDKeying(image, threshold0=300, threshold1=800)
# cv2.imwrite('src/LumaKwying_3.jpeg', mask)

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 计算图像绿值的范围，则除了图像人物以外，其他均为白色255，图像人物为黑色0
mask = cv2.inRange(hsv, (35, 43, 46), (77, 255, 255))



# 对mask进行形态学操作
k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3), (-1, -1))
# 返回指定形状和尺寸的核用于后面的形态学操作
mask = cv2.morphologyEx(mask,3, k)
# 通过闭操作 填充内部的小白点，去除干扰
mask = cv2.erode( mask, k)
# 腐蚀操作
mask = cv2.GaussianBlur(mask, (3, 3), 0, 0)
# 高斯模糊
# 将图像进行取反操作，则图像人物为白色255，其他为黑色0
cv2.bitwise_not(mask, mask)

cv2.imwrite('src/LumaKwying_4.jpeg', mask)

# convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# apply Gaussian blur
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
# use Canny edge detection to detect edges
canny_edges = cv2.Canny(blurred_image, 20, 80)
# apply binary threshold to create a mask
_, mask = cv2.threshold(canny_edges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# extract the foreground object using the mask
foreground = cv2.bitwise_and(image, image, mask=mask)

# display the result
cv2.imwrite('Foreground.jpeg', canny_edges)

import cv2
import numpy as np



# 转换为灰度图像
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = image

# 高斯滤波器
kernel_size = 3
gaussian = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 1)

gaussian_down = cv2.pyrDown(gaussian)
gaussian = cv2.pyrUp(gaussian_down)

# Gedge操作
gedge = cv2.subtract(gray, gaussian)

# # 显示结果
# cv2.imshow('Original', gray)
# cv2.imshow('Gaussian', gaussian)
cv2.imwrite('src/Gaussian.jpeg', gaussian)
cv2.imwrite('src/Gray.jpeg', gray)
cv2.imwrite('src/Gedge.jpeg', gedge)


