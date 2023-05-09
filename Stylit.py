import cv2
import math
import numpy as np
import random
import multiprocessing as mp
w = 5
interation = 4
channel = 1

def stylit(A, B, Aprime, Bprime):
    avg = np.zeros([3])
    avg.fill(2)
    for iter_time in range(0,2):
        print("Round "+str(iter_time+1)+" begins...")
        if iter_time == 0:
            u = 10000000
        else:
            u = 1
        nnf = nnf_init(B, A, Bprime, Aprime, u)
        nnf = nnf_iter(nnf, B, A, Bprime, Aprime, u)
        for i in range(Bprime.shape[0]):
            print("Row "+str(i+1)+" generate in result...")
            for j in range(Bprime.shape[1]):
                # print(i,j)
                if iter_time == 0:
                    Bprime[i,j] = average(A,Aprime,nnf,i,j).astype("uint8")
                else:
                    Bprime[i,j] = ((average(A,Aprime,nnf,i,j)+Bprime[i,j].astype("float32"))/avg).astype("uint8")
    # q.put(Bprime)

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
    # cost = Aprime[A_x-xmin:A_x+xmax, A_y-ymin:A_y+ymax]/(xmax+xmin)*(ymax+ymin)
    # cost = [(cost[i]/cnt).astype(np.uint8) for i in range(0,3)]
    cost = (cost / num)
    return cost

#NNF: nearest neighbor field
#Data Structure: 3-dimension array [row_x, col_y, value_diff]
def task_init(x, y, nnf, A, B, Aprime, Bprime, u):
    Arows = A.shape[1]
    Acols = A.shape[2]
    Brows = B.shape[1]
    Bcols = B.shape[2]
    for i in range(x, y):
        for j in range(Acols):
            print(i, j)
            nnf[i,j,0] = random.randint(0,Brows)
            nnf[i,j,1] = random.randint(0,Bcols)
            nnf[i,j,2] = calculate(A, B, Aprime, Bprime, i, j, nnf[i,j,0], nnf[i,j,1], u)

def nnf_init(A, B, Aprime, Bprime, u):
    print("NNF init begins...")
    Arows = A.shape[1]
    Acols = A.shape[2]
    Brows = B.shape[1]
    Bcols = B.shape[2]

    nnf = np.zeros([Arows, Acols, 3])

    # procs = []
    # for p in range(4):
    #     procs.append(mp.Process(target=task_init, args=(p*int(Arows/4), (p+1)*int(Arows/4), nnf, A, B, Aprime, Bprime, u)))
    
    # for proc in procs:
    #     proc.start()

    # for proc in procs:
    #     proc.join()
    
    for i in range(Arows):
        for j in range(Acols):
            nnf[i,j,0] = random.randint(0,Brows)
            nnf[i,j,1] = random.randint(0,Bcols)
            nnf[i,j,2] = calculate(A, B, Aprime, Bprime, i, j, nnf[i,j,0], nnf[i,j,1], u)
    print("NNF init finished...")
    return nnf

def nnf_iter(nnf, A, B, Aprime, Bprime, u):
    print("Interation begins...")
    Arows = A.shape[1]
    Acols = A.shape[2]

    for i in range(0, interation):
        print("Iteration Num. "+str(i+1)+" begins...")
        if i % 2 == 1:
            for x in range(Arows):
                for y in range(Acols):
                    nnf = propagation(nnf, A, B, Aprime, Bprime, x, y, i%2, u)
                    nnf = random_search(nnf, A, B, Aprime, Bprime, x, y, u)
        else:
            for x in range(Arows-1, -1, -1):
                for y in range(Acols-1, -1, -1):
                    nnf = propagation(nnf, A, B, Aprime, Bprime, x, y, i%2, u)
                    nnf = random_search(nnf, A, B, Aprime, Bprime, x, y, u)
        print("Iteration Num. "+str(i+1)+" finished...")
    print("Interation finished...")
    return nnf

def propagation(nnf, A, B, Aprime, Bprime, A_x, A_y, odd_flag, u):
    Arows = A.shape[1]
    Acols = A.shape[2]
    Brows = B.shape[1]
    Bcols = B.shape[2]

    B_x = int(nnf[A_x, A_y, 0])
    B_y = int(nnf[A_x, A_y, 1])
    B_diff = nnf[A_x, A_y, 2]
    if odd_flag:
        #left neighbor
        if A_x-1 >= 0:
            temp_bx = int(nnf[A_x-1, A_y, 0])
            temp_by = int(nnf[A_x-1, A_y, 1])
            if temp_bx+1 < Brows:
                temp_bdiff = calculate(A, B, Aprime, Bprime, A_x, A_y, temp_bx+1, temp_by, u)
                if temp_bdiff < B_diff:
                    B_x = temp_bx+1
                    B_y = temp_by
                    B_diff = temp_bdiff
        #up neighbor
        if A_y-1 >= 0:
            temp_bx = int(nnf[A_x, A_y-1, 0])
            temp_by = int(nnf[A_x, A_y-1, 1])
            if temp_by+1 < Brows:
                temp_bdiff = calculate(A, B, Aprime, Bprime, A_x, A_y, temp_bx, temp_by+1, u)
                if temp_bdiff < B_diff:
                    B_x = temp_bx
                    B_y = temp_by+1
                    B_diff = temp_bdiff
    else:
        #left neighbor
        if A_x+1 < Arows:
            temp_bx = int(nnf[A_x+1, A_y, 0])
            temp_by = int(nnf[A_x+1, A_y, 1])
            if temp_bx-1 >= 0:
                temp_bdiff = calculate(A, B, Aprime, Bprime, A_x, A_y, temp_bx-1, temp_by, u)
                if temp_bdiff < B_diff:
                    B_x = temp_bx-1
                    B_y = temp_by
                    B_diff = temp_bdiff
        #up neighbor
        if A_y+1 < Acols:
            temp_bx = int(nnf[A_x, A_y+1, 0])
            temp_by = int(nnf[A_x, A_y+1, 1])
            if temp_by-1 >= 0:
                temp_bdiff = calculate(A, B, Aprime, Bprime, A_x, A_y, temp_bx, temp_by-1, u)
                if temp_bdiff < B_diff:
                    B_x = temp_bx
                    B_y = temp_by-1
                    B_diff = temp_bdiff
    nnf[A_x, A_y, 0] = B_x
    nnf[A_x, A_y, 1] = B_y
    nnf[A_x, A_y, 2] = B_diff
    return nnf

def random_search(nnf, A, B, Aprime, Bprime, A_x, A_y, u):
    Arows = A.shape[1]
    Acols = A.shape[2]
    Brows = B.shape[1]
    Bcols = B.shape[2]
    radius = 8
    alpha = 0.5
    B_x, B_y = -1, -1
    while radius >= 1:
        while B_x < 0 or B_x >= Brows:
            B_x = int(nnf[A_x, A_y, 0] + radius*random.uniform(-1,1))
        while B_y < 0 or B_y >= Bcols:
            B_y = int(nnf[A_x, A_y, 1] + radius*random.uniform(-1,1))
        diff = calculate(A, B, Aprime, Bprime, A_x, A_y, B_x, B_y, u, nnf[A_x, A_y, 2])
        if diff < nnf[A_x, A_y, 2]:
            nnf[A_x, A_y, 0] = B_x
            nnf[A_x, A_y, 1] = B_y
            nnf[A_x, A_y, 2] = diff
        radius *= alpha
    return nnf

def calculate(A, B, Aprime, Bprime, A_x, A_y, B_x, B_y, u, best=float("inf")):
    Arows = A.shape[1]
    Acols = A.shape[2]
    Brows = B.shape[1]
    Bcols = B.shape[2]
    cost = 0
    cnt = 0
    # for i in range(A_x-w//2, A_x+w//2+1):
    #     for j in range(A_y-w//2, A_y+w//2+1):
    #         if i < 0 or j < 0 or i > Arows-1 or j > Acols-1:
    #             continue
    #         if B_x-A_x+i < 0 or B_y-A_y+j < 0 or B_x-A_x+i > Brows-1 or B_y-A_y+j > Bcols-1:
    #             continue
    #         cnt += 1
    #         cost += E(A[i, j], B[int(B_x-A_x+i), int(B_y-A_y+j)], Aprime[i, j], Bprime[int(B_x-A_x+i), int(B_y-A_y+j)], u)
            
    #         if cnt > 12 and cost/(cnt*2) > best:
    #             return float("inf")
    xmin = int(min(w//2, A_x, B_x))
    xmax = int(min(w//2, Arows-A_x-1, Brows-B_x-1)+1)
    ymin = int(min(w//2, A_y, B_y))
    ymax = int(min(w//2, Acols-A_y-1, Bcols-B_y-1)+1)
    cost = np.sum(u*(A[0:channel, int(A_x-xmin):int(A_x+xmax), int(A_y-ymin):int(A_y+ymax)].astype("float32")-B[0:channel, int(B_x-xmin):int(B_x+xmax), int(B_y-ymin):int(B_y+ymax)].astype("float32"))**2)/channel+np.sum((Aprime[int(A_x-xmin):int(A_x+xmax), int(A_y-ymin):int(A_y+ymax)].astype("float32")-Bprime[int(B_x-xmin):int(B_x+xmax), int(B_y-ymin):int(B_y+ymax)].astype("float32"))**2)
    if cost == 0:
        return 0
    # cost /= cnt
    cost /= (xmin+xmax)*(ymin+ymax)
    return cost

# def argmin(A, B, Aprime, Bprime, x, y):
#     temp = float ("inf")
#     min = float ("inf")
#     Arows = A.shape[0]
#     Acols = A.shape[1]
#     for i in range(Arows):
#         for j in range(Acols):
#             temp = E(A[i,j], B[x,y], Aprime[i,j], Bprime[x,y], 100)
#             if temp < min:
#                 min = temp
#                 min_point = i,j
    
#     return min_point

def E(a, b, aprime, bprime, u):
    # diff_a = [aprime[i].astype(np.float32)-bprime[i].astype(np.float32) for i in range(0, len(a))]
    # diff_b = [a[i].astype(np.float32)-b[i].astype(np.float32) for i in range(0, len(b))]

    # return (diff_a[0]*diff_a[0]+diff_a[1]*diff_a[1]+diff_a[2]*diff_a[2])+u*(diff_b[0]*diff_b[0]+diff_b[1]*diff_b[1]+diff_b[2]*diff_b[2])

    # x = np.sum(u*(a.astype("float32")-b.astype("float32"))**2+(aprime.astype("float32")-bprime.astype("float32"))**2)
    return np.sum(u*(a.astype("float32")-b.astype("float32"))**2+(aprime.astype("float32")-bprime.astype("float32"))**2)