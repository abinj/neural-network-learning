import math
import random

import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing

linkLength = 2

Q1 = []
Q2 = []
Q3 = []
posX = []
posY = []
titaEnd = []
samples = 1000

file = open("/home/abin/my_works/github_works/neural-network-learning/dataset/training_data.csv", "w")


def Xe(q1, q2, q3):
    return linkLength*math.cos(q1) + linkLength*math.cos(q1+q2) + linkLength*math.cos(q1+q2+q3)


def Ye(q1, q2, q3):
    return linkLength*math.sin(q1) + linkLength*math.sin(q1+q2) + linkLength*math.sin(q1+ q2 + q3)


def tita(q1, q2, q3):
    return math.degrees(q1) + math.degrees(q2) + math.degrees(q3)


for i in range(0,samples):
    q1 = round(random.uniform(0, math.pi),2)
    q2 = round(random.uniform(-math.pi,0),2)
    q3 = round(random.uniform(-math.pi/2, math.pi/2),2)

    Q1.append(q1)
    file.write(str(q1))
    file.write(",")

    Q2.append(q2)
    file.write(str(q2))
    file.write(",")

    Q3.append(q3)
    file.write(str(q3))
    file.write(",")

    X = Xe(q1,q2,q3)
    posX.append(X)
    file.write(str(round(X, 2)))
    file.write(",")

    Y = Ye(q1,q2,q3)
    posY.append(Y)
    file.write(str(round(Y, 2)))
    file.write(",")

    T = tita(q1,q2,q3)
    titaEnd.append(T)
    file.write(str(round(T,2)))
    file.write("\n")

file.close()

for i in range(0, len(posX)):
    plt.plot([posX[i], posX[i] + 0.2*math.cos(math.radians(titaEnd[i]))], [posY[i], posY[i]+0.2*math.sin(math.radians(titaEnd[i]))])

plt.scatter(posX,posY)
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.title("Data set of 1000 possible endeffector positions  and orientations")
plt.show()

dataMat = np.c_[Q1,Q2,Q3,posX,posY,titaEnd]

for i in range(0,samples):
    check1 = dataMat[i, 3]
    check2 = dataMat[i, 4]
    check3 = dataMat[i, 5]
    for j in range(0, samples):
        if i != j:
            if(dataMat[j, 3] == check1 and dataMat[j, 4] == check2 and dataMat[j, 5] == check3):
                print(i, j, dataMat[j, 3], dataMat[j, 4], dataMat[j, 5])


