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

