import cv2
import math
import numpy as np
import random
import multiprocessing as mp
w = 5
iteration = 6
Channel = 4

class NNF:
    def __init__(self, A, B, Aprime, Bprime):
        self.A = A
        self.B = B
        self.Aprime = Aprime
        self.Bprime = Bprime
        self.Arows = A.shape[1]
        self.Acols = A.shape[2]
        self.Brows = B.shape[1]
        self.Bcols = B.shape[2]
        self.nnf = np.zeros([self.Arows, self.Acols, 3], dtype="int32")
        self.bmap = np.zeros([self.Brows, self.Bcols, 3], dtype="int32")
        self.cp = np.zeros([self.Arows, self.Acols], dtype="int32")

    #NNF: nearest neighbor field
    #Data Structure: 3-dimension array [row_x, col_y, value_diff]

    #nnf_init: Initiate the nnf, for every patch in A, randomly choose a patch in B, this patch_A is to be assigned to B
    def nnf_init(self, K, R, u):
        print("NNF init begins...")
        # nnf = np.zeros([Arows, Acols, 3])
        
        for i in range(self.Arows):
            for j in range(self.Acols):
                if self.cp[i,j] <= K and R != 0 or self.cp[i,j] < K:
                    available = np.argwhere(self.bmap[:, :, 2] == 0)[:, 0:2]
                    l = random.randint(0,len(available)-1)
                    x_b = available[l][0]
                    y_b = available[l][1]

                    # while self.bmap[x_b, y_b, 2] != 0:
                    #     x_b = random.randint(0,self.Brows-1)
                    #     y_b = random.randint(0,self.Bcols-1)
                    self.nnf[i,j,:2] = x_b, y_b
                    self.nnf[i,j,2] = calculate(self.A, self.B, self.Aprime, self.Bprime, i, j, x_b, y_b, u)
        # print(self.nnf)
        print("NNF init finished...")
        # return self.nnf

    #nnf_iter: iterate several times of the patch match algorithm, to reach the convergence
    def nnf_iter(self, R, K, u, iter=6):
        print("Interation begins...")
        R_local = R
        for i in range(0, iter):
            print("Iteration Num. "+str(i+1)+" begins...")
            #odd time: order from up down, left to right
            if i % 2 == 1:
                for x in range(self.Arows):
                    for y in range(self.Acols):
                        if self.cp[x,y] <= K and R != 0 or self.cp[x,y] < K:
                            self.propagation(x, y, i%2, u)
                            self.random_search(x, y, u)
                            #at the last iter time, we increase cp, and assign the patch of B to A
                            if i == iter-1 :
                                if self.cp[x, y] == K:
                                    R_local -= 1
                                #address conflict of two patches in src correspond to the same patch in target, assign the final nnf
                                self.address_conflict(x, y)
            #even time: reversed scanning order
            else:
                for x in range(self.Arows-1, -1, -1):
                    for y in range(self.Acols-1, -1, -1):
                        if self.cp[x,y] <= K and R != 0 or self.cp[x,y] < K:
                            self.propagation(x, y, i%2, u)
                            self.random_search(x, y, u)
                            if i == iter-1 :
                                if self.cp[x, y] == K:
                                    R_local -= 1
                                self.address_conflict(x, y)

            print("Iteration Num. "+str(i+1)+" finished...")
        print("Interation finished...")
        return R_local

    def address_conflict(self, x, y):
        if(self.nnf[x,y,2] < self.bmap[int(self.nnf[x,y,0]), int(self.nnf[x,y,1]), 2]):
            #quite complex as I need to 
            self.cp[int(self.bmap[int(self.nnf[x,y,0]), int(self.nnf[x,y,1]),0]),int(self.bmap[int(self.nnf[x,y,0]), int(self.nnf[x,y,1]),1])] -= 1
            self.bmap[int(self.nnf[x,y,0]), int(self.nnf[x,y,1]),:3] = x,y,self.nnf[x,y,2]
            self.cp[x,y] += 1
        elif self.bmap[int(self.nnf[x,y,0]), int(self.nnf[x,y,1]), 2] == 0:
            self.bmap[int(self.nnf[x,y,0]), int(self.nnf[x,y,1]),:3] = x,y,self.nnf[x,y,2]
            self.cp[x,y] += 1
    #propagation: find the best match patch according to its corresponding patch 
    # along with their neighbors' best match patches
    def propagation(self, A_x, A_y, odd_flag, u):
        B_x = int(self.nnf[A_x, A_y, 0])
        B_y = int(self.nnf[A_x, A_y, 1])
        B_diff = self.nnf[A_x, A_y, 2]
        if not odd_flag:
            #left neighbor
            if A_x-1 >= 0:
                temp_bx = int(self.nnf[A_x-1, A_y, 0])
                temp_by = int(self.nnf[A_x-1, A_y, 1])
                if temp_bx+1 < self.Brows and self.bmap[temp_bx+1, temp_by, 2] == 0:
                    temp_bdiff = calculate(self.A, self.B, self.Aprime, self.Bprime, A_x, A_y, temp_bx+1, temp_by, u)
                    #if the patch near the lest neighbor's corresponding patch is better
                    if temp_bdiff < B_diff:
                        B_x = temp_bx+1
                        B_y = temp_by
                        B_diff = temp_bdiff
            #up neighbor
            if A_y-1 >= 0:
                temp_bx = int(self.nnf[A_x, A_y-1, 0])
                temp_by = int(self.nnf[A_x, A_y-1, 1])
                if temp_by+1 < self.Brows and self.bmap[temp_bx, temp_by+1, 2] == 0:
                    temp_bdiff = calculate(self.A, self.B, self.Aprime, self.Bprime, A_x, A_y, temp_bx, temp_by+1, u)
                    if temp_bdiff < B_diff:
                        B_x = temp_bx
                        B_y = temp_by+1
                        B_diff = temp_bdiff
        else:
            #left neighbor
            if A_x+1 < self.Arows:
                temp_bx = int(self.nnf[A_x+1, A_y, 0])
                temp_by = int(self.nnf[A_x+1, A_y, 1])
                if temp_bx-1 >= 0 and self.bmap[temp_bx-1, temp_by, 2] == 0:
                    temp_bdiff = calculate(self.A, self.B, self.Aprime, self.Bprime, A_x, A_y, temp_bx-1, temp_by, u)
                    if temp_bdiff < B_diff:
                        B_x = temp_bx-1
                        B_y = temp_by
                        B_diff = temp_bdiff
            #up neighbor
            if A_y+1 < self.Acols:
                temp_bx = int(self.nnf[A_x, A_y+1, 0])
                temp_by = int(self.nnf[A_x, A_y+1, 1])
                if temp_by-1 >= 0 and self.bmap[temp_bx, temp_by-1, 2] == 0:
                    temp_bdiff = calculate(self.A, self.B, self.Aprime, self.Bprime, A_x, A_y, temp_bx, temp_by-1, u)
                    if temp_bdiff < B_diff:
                        B_x = temp_bx
                        B_y = temp_by-1
                        B_diff = temp_bdiff
        self.nnf[A_x, A_y, 0] = B_x
        self.nnf[A_x, A_y, 1] = B_y
        self.nnf[A_x, A_y, 2] = B_diff
        # return nnf

    #random_search: let 
    def random_search(self, A_x, A_y, u):
        #random search radius, and alpha is to reduce the radius each seach
        radius = 8
        alpha = 0.5
        B_x = int(self.nnf[A_x, A_y, 0] + radius*random.uniform(-1,1))
        B_y = int(self.nnf[A_x, A_y, 1] + radius*random.uniform(-1,1))
        while radius >= 1:
            while (B_x < 0 or B_x >= self.Brows) or (B_y < 0 or B_y >= self.Bcols):
                B_x = int(self.nnf[A_x, A_y, 0] + radius*random.uniform(-1,1))
                B_y = int(self.nnf[A_x, A_y, 1] + radius*random.uniform(-1,1))
            diff = calculate(self.A, self.B, self.Aprime, self.Bprime, A_x, A_y, B_x, B_y, u, self.nnf[A_x, A_y, 2])
            if diff < self.nnf[A_x, A_y, 2] and self.bmap[B_x, B_y, 2] == 0:
                self.nnf[A_x, A_y, 0] = B_x
                self.nnf[A_x, A_y, 1] = B_y
                self.nnf[A_x, A_y, 2] = diff
            radius *= alpha
        return self.nnf
    
    '''
    calculate the parameter for reversed NNF retrieval
    A is the src; B is the target
    K = floor(|B| / |A|)
    R = |B| % |A|
    '''
    def reverse_NNF_para(self, A, B):
        size_A = A.shape[1]*A.shape[2]
        size_B = B.shape[1]*B.shape[2]
        K = int(math.floor(size_B/size_A))
        R = size_B % size_A
        return K, R

    #dealer_reverse_NNF: 
    def dealer_reverse_NNF(self, u, iter=6):
        K, R = self.reverse_NNF_para(self.A, self.B)
        # nnf = nnf_init(A, B, Bprime, Aprime, u)
        cover = 0
        total = self.Bcols * self.Brows
        while cover < total:
            print("cover pixels in B are " + str(cover) + "/"+str(total))
            self.nnf_init(K, R, u)
            R = self.nnf_iter(R, K, u, iter)
            cover = np.sum(self.cp)

def stylit(A, B, Aprime, Bprime, channel):
    global Channel
    Channel = channel
    avg = np.zeros([3])
    avg.fill(2)
    for iter_time in range(0,6):
        print("Round "+str(iter_time+1)+" begins...")
        u = 2
        deal_nnf = NNF(A, B, Aprime, Bprime)
        deal_nnf.dealer_reverse_NNF(u, iteration)
        nnf = deal_nnf.bmap
        # nnf.nnf_init(u)
        # nnf.nnf_iter(nnf, A, B, Bprime, Aprime, u)
        for i in range(Bprime.shape[0]):
            print("Row "+str(i+1)+" generate in result...")
            for j in range(Bprime.shape[1]):
                if iter_time == 0:
                    Bprime[i,j] = average(A,Aprime,nnf,i,j).astype("uint8")
                else:
                    Bprime[i,j] = ((average(A,Aprime,nnf,i,j)+Bprime[i,j].astype("float32"))/avg).astype("uint8")

def average(A, Aprime, nnf, B_x, B_y):
    A_x = int(nnf[B_x, B_y, 0])
    A_y = int(nnf[B_x, B_y, 1])
    Arows = A.shape[1]
    Acols = A.shape[2]
    cost = np.zeros([3])
    cnt = 0
    xmin = int(min(w//2, A_x))
    xmax = int(min(w//2, Arows-A_x-1)+1)
    ymin = int(min(w//2, A_y))
    ymax = int(min(w//2, Acols-A_y-1)+1)
    for i in range(A_x-w//2, A_x+w//2+1):
        for j in range(A_y-w//2, A_y+w//2+1):
            if i < 0 or j < 0 or i > Arows-1 or j > Acols-1:
                continue
            cnt += 1
            cost = cost+Aprime[i,j].astype(np.float32)
    num = np.zeros([3])
    num.fill(cnt)
    cost = (cost / num)
    return cost

def calculate(A, B, Aprime, Bprime, A_x, A_y, B_x, B_y, u, best=float("inf")):
    Arows = A.shape[1]
    Acols = A.shape[2]
    Brows = B.shape[1]
    Bcols = B.shape[2]
    cost = 0

    xmin = int(min(w//2, A_x, B_x))
    xmax = int(min(w//2, Arows-A_x-1, Brows-B_x-1)+1)
    ymin = int(min(w//2, A_y, B_y))
    ymax = int(min(w//2, Acols-A_y-1, Bcols-B_y-1)+1)
    cost = np.sum(u*(A[0:Channel, int(A_x-xmin):int(A_x+xmax), int(A_y-ymin):int(A_y+ymax)].astype("float32")-B[0:Channel, int(B_x-xmin):int(B_x+xmax), int(B_y-ymin):int(B_y+ymax)].astype("float32"))**2)/Channel+np.sum((Aprime[int(A_x-xmin):int(A_x+xmax), int(A_y-ymin):int(A_y+ymax)].astype("float32")-Bprime[int(B_x-xmin):int(B_x+xmax), int(B_y-ymin):int(B_y+ymax)].astype("float32"))**2)
    #add disturb to ensure not zero
    if cost == 0:
        return 0
    cost += 1e-10
    cost /= (xmin+xmax)*(ymin+ymax)
    return cost

def E(a, b, aprime, bprime, u):
    return np.sum(u*(a.astype("float32")-b.astype("float32"))**2+(aprime.astype("float32")-bprime.astype("float32"))**2)