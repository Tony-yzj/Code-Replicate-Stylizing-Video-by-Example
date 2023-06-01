import cv2
import numpy as np
import copy as cp
import multiprocessing as mp
channel = 1
path = "source/"

import cv2
import numpy as np

#use gaussian pyramid to process the src picture
def gaussian_pyramid(image, levels):
    pyramid = [image]
    for _ in range(0, levels-1):
        image = cv2.pyrDown(image)
        pyramid.append(image)
    return pyramid

#use laplacian pyramid to recover the guassian pyramid
def laplacian_pyramid(pyramid, levels):
    l_pyramid = []
    for i in range(0, levels-1):
        upsample = cv2.pyrUp(pyramid[i+1])
        height, width = pyramid[i].shape[:2]
        upsample = cv2.resize(upsample, (width, height))
        # diff = cv2.subtract(pyramid[i], upsample)
        diff = pyramid[i] - upsample
        l_pyramid.append(diff)
    return l_pyramid

def recover_img(img, level, pyramid):
    for i in range(level, 0, -1):
        upsample = cv2.pyrUp(img)
        height, width = pyramid[i-1].shape[:2]
        upsample = cv2.resize(upsample, (width, height))
        # img = cv2.add(upsample, pyramid[i-1])
        img = upsample + pyramid[i-1]
    return img



def main():
    print("Readimg begins...")
    image = cv2.imread(path+"Mee.jpg")
    imgA0 = cv2.imread(path+"source_fullgi.png")
    imgB0 = cv2.imread(path+"target_fullgi.png")

    # 生成高斯金字塔
    gaussian_levels = 6
    g_pyramid = gaussian_pyramid(image, gaussian_levels)



    # 生成拉普拉斯金字塔
    l_pyramid = laplacian_pyramid(g_pyramid, gaussian_levels)

    # 计算高斯分布的权重值列表
    for i in range(gaussian_levels-1, 0, -1):
        g_pyramid[i] = recover_img(g_pyramid[i], i, l_pyramid)

    # 将结果图像转换为整数形式（0-255范围）
    # result = np.clip(result, 0, 255).astype(np.uint8)
    for i in range(gaussian_levels):
        cv2.imshow("laplacian{}".format(i), g_pyramid[i])
        cv2.waitKey(0)

    # 显示结果图像
    # cv2.imshow("Result Image", result)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()

    
if __name__ == '__main__':
	main()
# q = mp.Queue()
    # p1 = mp.Process(target=Stylit.stylit, args=(imgA, imgB, imgAprime, imgBprime, q))
    # p2 = mp.Process(target=Stylit.stylit, args=(imgA1, imgB1, imgAprime, imgBprime1, q))
    # p3 = mp.Process(target=Stylit.stylit, args=(imgA2, imgB2, imgAprime, imgBprime2, q))
    # p4 = mp.Process(target=Stylit.stylit, args=(imgA3, imgB3, imgAprime, imgBprime3, q))
    # p1.start()
    # p2.start()
    # p3.start()
    # p4.start()
    # imgBprime = q.get()
    # imgBprime1 = q.get()
    # imgBprime2 = q.get()
    # imgBprime3 = q.get()
    # p1.join()
    # p2.join()
    # p3.join()
    # p4.join()