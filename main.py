import cv2
import numpy as np
import Stylit
import multiprocessing as mp
channel = 4
path = "source/"

def main():
    print("Readimg begins...")
    imgA0 = cv2.imread(path+"source_fullgi.png")
    imgB0 = cv2.imread(path+"target_fullgi.jpg")
    
    imgA1 = cv2.imread(path+"source_dirdif.png")
    imgB1 = cv2.imread(path+"target_dirdif.png")

    imgA2 = cv2.imread(path+"source_dirspc.png")
    imgB2 = cv2.imread(path+"target_dirspc.png")

    imgA3 = cv2.imread(path+"source_indirb.png")
    imgB3 = cv2.imread(path+"target_indirb.png")
    print("Read images finished...")
    imgA = np.zeros((channel, imgA0.shape[0], imgA0.shape[1], 3), np.uint8)
    imgA[0] = imgA0
    imgA[1] = imgA1
    imgA[2] = imgA2
    imgA[3] = imgA3
    imgAprime = cv2.imread(path+"source_style.png")
    imgB = np.zeros((channel, imgB0.shape[0], imgB0.shape[1], 3), np.uint8)
    imgB[0] = imgB0
    imgB[1] = imgB1
    imgB[2] = imgB2
    imgB[3] = imgB3
    imgBprime = np.zeros((imgB0.shape[0], imgB0.shape[1], 3), np.uint8)
    imgBprime[:] = [255,255,255]
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
    print("Stylit begins...")
    Stylit.stylit(imgA, imgB, imgAprime, imgBprime)
    print("Stylit finished...")
    # Stylit.stylit(imgA1, imgB1, imgAprime, imgBprime1)
    # Stylit.stylit(imgA2, imgB2, imgAprime, imgBprime2)
    # Stylit.stylit(imgA3, imgB3, imgAprime, imgBprime3)
    # imgBprime = ((imgBprime.astype("float32")+imgBprime1.astype("float32")+imgBprime2.astype("float32")+imgBprime3.astype("float32"))/4).astype("uint8")
    cv2.imwrite(path+"result.png",imgBprime)
    cv2.imshow('img', imgBprime)
    cv2.waitKey(0)

if __name__ == '__main__':
	main()
