import math
import random

import keras
import numpy as np
import tensorflow as tf
from matplotlib import patches as mpatches
from keras.callbacks import TensorBoard
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

def build_model():
    model = keras.Sequential()
    model.add(keras.layers.Dense(100,use_bias=True,activation='tanh'))
    model.add(keras.layers.Dense(3,use_bias=True,activation='linear'))
    model.compile(optimizer=tf.compat.v1.train.AdamOptimizer(0.05), loss=keras.losses.mean_squared_error, metrics=['accuracy'])
    return model


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


data = dataMat[:, [3,4,5]] #X,Y,tita
output = dataMat[:, [0,1,2]] #Q1,Q2,Q3

train_input = data[0:int(0.7*samples), :]
train_output = output[0:int(0.7*samples), :]

test_input = data[int(0.7*samples):int(0.85*samples), :]
test_output = output[int(0.7*samples):int(0.85*samples), :]

validate_input = data[int(0.85*samples):int(samples), :]
validate_output = output[int(0.85*samples):int(samples), :]

print("Train Input--------------------")
print(np.shape(train_input))
print(train_input)

print("Output-----------------------")
print(np.shape(output))
print(output)

x_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
y_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
x_scaler_test = preprocessing.MinMaxScaler(feature_range=(-1,1))
y_scaler_test = preprocessing.MinMaxScaler(feature_range=(-1,1))
x_scaler_eva = preprocessing.MinMaxScaler(feature_range=(-1,1))
y_scaler_eva = preprocessing.MinMaxScaler(feature_range=(-1,1))

dataX = x_scaler.fit_transform((train_input))
dataY = y_scaler.fit_transform(train_output)
dataX_test = x_scaler_test.fit_transform(test_input)
dataY_test = y_scaler_test.fit_transform(test_output)
dataX_eva = x_scaler_eva.fit_transform(validate_input)
dataY_eva = y_scaler_eva.fit_transform(validate_output)

NAME = "Trajectry Tracking"

tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))


model = build_model()

history = model.fit(dataX, dataY, nb_epoch=100, callbacks=[tensorboard])    #train the model

[loss, mae] = model.evaluate(dataX_test,dataY_test, verbose=0)  #Evaluation

print("Testing set Mean Abs Error: ${:7.2f}".format(mae))


dataX_input = x_scaler.transform(validate_input)
test_prediction = model.predict(dataX_input)
real_prediction = y_scaler.inverse_transform(test_prediction)

plt.clf()
plt.scatter(validate_output[:, 0], real_prediction[:, 0], c='b')
plt.scatter(validate_output[:, 1], real_prediction[:, 1],c='g')
plt.scatter(validate_output[:, 2], real_prediction[:, 2], c='r')
plt.xlabel('True Values angles in rad')
plt.ylabel('Predictions angles in rad')
plt.title("True Value Vs Prediction")
plt.legend("If all predicted values equal to the desired(true) value, this will be lie on 45 degree line")

plt.show()

print("***************************")
print(validate_input[100,0], " ", validate_input[100,1])
print(Xe(real_prediction[100,0], real_prediction[100,1], real_prediction[100,2]), " "
      , Ye(real_prediction[100,0], real_prediction[100, 1], real_prediction[100,2]))
print("***************************")

single_data_1 = np.array([[5,2,60]])
single_data = x_scaler.transform((single_data_1))
single_prediction = model.predict(single_data)
single_real_prediction = y_scaler.inverse_transform(single_prediction)

print(single_data_1[0,0], " ", single_data_1[0,1])
print(Xe(single_real_prediction[0,0], single_real_prediction[0,1], single_real_prediction[0,2]), " "
      , Ye(single_real_prediction[0,0], single_real_prediction[0,1], single_real_prediction[0,2]))
print("****************************")

Xc = 0
Yc = 0
r =2
data_points = 100

Input_Circle = np.zeros((data_points, 3), float)
Output_Circle= np.zeros((data_points, 3), float)
Single_input = np.zeros((1,3), float)

titaz = np.linspace(0,6,num=data_points)

tagectory = []

for i in range(0, len(titaz)):
    Input_Circle[i][0]=Xc + titaz[i]
    Input_Circle[i][1]=Yc + titaz[i]
    Input_Circle[i][2]= 90


plt.clf()
plt.scatter(Input_Circle[:,0], Input_Circle[:, 1], c = 'b')
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.title("Desired Tragectory Coordinates and Predicted ")

inin = np.zeros((1,3), float)
inin[0][0] = Input_Circle[0,0]
inin[0][1] = Input_Circle[0,1]
inin[0][2] = Input_Circle[0,2]

Predicted_coordinates = np.zeros((data_points, 3), float)
print(np.shape(Input_Circle))

print(single_data_1[0,0], " ", single_data_1[0,1])
print(Xe(single_real_prediction[0,0], single_real_prediction[0,1], single_real_prediction[0,2]), " "
      , Ye(single_real_prediction[0,0], single_real_prediction[0,1], single_real_prediction[0,2]))

print("********************************")


Joint_angle_predict = np.zeros((data_points,4), float)
Error = np.zeros((data_points, 7), float)

Tita_hat= 0

for q in range(0, data_points):
    single_data_1 = np.array([[Input_Circle[q, 0], Input_Circle[q,1], Input_Circle[q,2]]])
    single_data = x_scaler.transform(single_data_1)
    single_prediction = model.predict(single_data)
    single_real_prediction = y_scaler.inverse_transform(single_prediction)

    X_hat = Xe(single_real_prediction[0, 0], single_real_prediction[0,1], single_real_prediction[0,2])
    Y_hat = Ye(single_real_prediction[0,0], single_real_prediction[0,1], single_real_prediction[0,2])
    Tita_hat = tita(single_real_prediction[0,0], single_real_prediction[0,1], single_real_prediction[0,2])

    Joint_angle_predict[q][0] = q
    Joint_angle_predict[q][1] = single_real_prediction[0,0]
    Joint_angle_predict[q][2] = single_real_prediction[0,1]
    Joint_angle_predict[q][3] = single_real_prediction[0,2]

    Error[q][0] = Input_Circle[q, 0] - X_hat
    Error[q][1] = Input_Circle[q, 1] - Y_hat
    Error[q][2] = Input_Circle[q, 2] - Tita_hat
    Error[q][3] = math.degrees(single_real_prediction[0,0])
    Error[q][4] = math.degrees(single_real_prediction[0,1])
    Error[q][5] = math.degrees(single_real_prediction[0,2])
    Error[q][6] = q

    print("X: ", Input_Circle[q, 0], " Y: ", Input_Circle[q, 1], "Tita: ", Input_Circle[q, 2])
    print("X^:", X_hat, " Y^:", Y_hat, " Tita^:", Tita_hat)
    print(" ")
    plt.scatter(X_hat, Y_hat, c='r')

    plt.savefig('Desired Tragectory Cordinates and  Predicted.png')

    Q1_patch = mpatches.Patch(color='red', label='1st joint')
    Q2_patch = mpatches.Patch(color='blue', label='2nd joint')
    Q3_patch = mpatches.Patch(color='green', label='3rd joint')

    plt.clf()
    plt.plot(Joint_angle_predict[:, 0], Joint_angle_predict[:, 1], c='r')
    plt.plot(Joint_angle_predict[:, 1], Joint_angle_predict[:, 2], c='b')
    plt.plot(Joint_angle_predict[:, 2], Joint_angle_predict[:, 3], c='g')
    plt.legend(handles = [Q1_patch, Q2_patch, Q3_patch])
    plt.title('Joint angle variation over data points')

    plt.savefig('Joint angle variation over data points.png')

    P1_patch = mpatches.Patch(color='red', label='Error in X coordinates')
    P2_patch = mpatches.Patch(color='blue', label='Error in Y coordinates')
    P3_patch = mpatches.Patch(color='green', label='Error in Tita ')

    plt.clf()
    plt.plot(Error[:, 6], Error[:, 0], c='r')
    plt.plot(Error[:, 6], Error[:, 1], c='b')
    plt.plot(Error[:, 6], Error[:, 2], c='g')
    plt.title('Error of X, Y and tita in the evolution')
    plt.legend(handles = [P1_patch, P2_patch, P3_patch])

    plt.savefig('Error of X,Y and Tita in the evolution')
    plt.show()

