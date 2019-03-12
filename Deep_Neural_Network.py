#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 16:06:32 2018

@author: chenjin
"""

import numpy as np
import time
import h5py
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage
from PIL import Image
#from testCases_v2 import *  #提供了一些测试函数所有的数据和方法
#from dnn_utils_v2 import sigmoid, sigmoid_backward, relu, relu_backward  #封装好的方法
#from dnn_app_utils_v2 import *

def load_Dataset():
    #加载数据集
    train_dataset = h5py.File('datasets/train_catvnoncat.h5','r')
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])
    
    test_dataset = h5py.File('datasets/test_catvnoncat.h5','r')
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])
    
    classes = np.array(test_dataset['list_classes'][:])
    
    train_set_y_orig = train_set_y_orig.reshape((1,train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1,test_set_y_orig.shape[0]))
    
    
    return train_set_x_orig,train_set_y_orig,test_set_x_orig,test_set_y_orig,classes,train_set_y

train_set_x_orig,train_set_y,test_set_x_orig,test_set_y,classes,train_set_y = load_Dataset()


m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]

train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

train_set_x = train_set_x_flatten/255
test_set_x = test_set_x_flatten/255


def sigmoid(Z): #sigmoid函数
    
    A = 1/(1 + np.exp(-Z))
    cache = Z
    return A,cache

def sigmoid_backward(dA,cache): #sigmoid 函数的导数，用于反向传播
    
    Z = cache
    s = 1/(1 + np.exp(-Z))
    dZ = dA * s * (1-s) #查了两小时错误，原因：将1-s写成s-1
    assert(Z.shape == dZ.shape)
    return dZ

def Relu(Z):  #Relu 函数
    
    A = np.maximum(0,Z)
    
    assert(A.shape == Z.shape)
    cache = Z
    return A,cache

def Relu_backward(dA,cache):  #Relu 函数的导数，用于反向传播
    
    Z = cache
    dZ = np.array(dA,copy=True)
    dZ[Z <= 0] = 0
    assert(dZ.shape == Z.shape)
    return dZ

def initialize_parameters(layer_dims): #初始化各层的每个节点的W值，以及b值
    #np.random.seed(1)
    parameters = {}
    layer_Num = len(layer_dims)
    print(layer_Num)
    
    for i in range(1,layer_Num):
        parameters['W' + str(i)] = np.random.randn(layer_dims[i],layer_dims[i-1])/np.sqrt(layer_dims[i-1])
        parameters['b' + str(i)] = np.zeros((layer_dims[i],1)) 
        assert(parameters['W' + str(i)].shape == (layer_dims[i],layer_dims[i-1]))
        assert(parameters['b' + str(i)].shape == (layer_dims[i],1)) 
    
    return parameters
'''
layer_dims=[5,4,3]
layer_Num = len(layer_dims)
parameters = initialize_parameters(layer_dims)
for i in range(1,layer_Num):
    print("W" + str(i) + "=" + str(parameters['W' + str(i)]))
    print("B" + str(i) + "=" + str(parameters['B' + str(i)]))

'''
def linear_forward(A,W,b): #向前传播计算Z=W * A + b
    
    Z = np.dot(W,A)+b
    assert(Z.shape == (W.shape[0],A.shape[1]))
    cache = (A,W,b)
    
    return Z,cache

def linear_activation_forward(A_prev,W,b,activation): #向前传播，分为前n-1层（relu）和最后一层二分（sigmoid）
    
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev,W,b)
        A,activation_cache = sigmoid(Z)
        
    if activation == "Relu":
        Z, linear_cache = linear_forward(A_prev,W,b)
        A,activation_cache = Relu(Z)
        
    assert(A.shape == (W.shape[0],A_prev.shape[1]))
   
    cache = (linear_cache, activation_cache)
    
    return A,cache

def L_model_forward(X,parameters): #向前传播的整合
    
    caches = []
    A = X
    L = int(len(parameters)/2)
    #print(L)
    
    for i in range(1,L):
        A_prev = A
        A,cache = linear_activation_forward(A_prev, parameters['W' + str(i)], parameters['b' + str(i)], "Relu")
        caches.append(cache)
        
    AL,cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], "sigmoid")
    caches.append(cache)
    assert(AL.shape == (1, X.shape[1]))
    return AL,caches

def cost_function(AL,Y): #cost函数，用于衡量深度学习的精确度
    
    m = Y.shape[1]
    cost = -(1/m)*np.sum(Y * np.log(AL)+(1 - Y) * np.log(1 - AL)) #np.multiply()
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    return cost

def linear_backward(dZ, cache):
    
    A_prev, W, b = cache
    m = A_prev.shape[1]
    
    dW = (1/m)*np.dot(dZ, A_prev.T)
    db = (1/m)*np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T, dZ)

    assert(dW.shape == W.shape)
    assert(db.shape == b.shape)
    assert(dA_prev.shape == A_prev.shape)
    
    return dA_prev,dW,db


def linear_activation_backward(dA, cache, activation): 
    
    linear_cache, activation_cache =cache
    
    if activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db =linear_backward(dZ, linear_cache)
        
    if activation == "Relu":
        dZ = Relu_backward(dA, activation_cache)
        dA_prev, dW, db =linear_backward(dZ, linear_cache)
        
    return dA_prev, dW, db

def L_model_backward(AL, Y, caches):
    
    grads = {}
    L = len(caches)
    #m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    
    dAL = -(np.divide(Y, AL) - np.divide((1-Y),(1-AL)))
    current_cache = caches[L-1] #caches是从0开始的，所以最后一个sigmoid应为L-1
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, "sigmoid")
        
    for i in reversed(range(L-1)):
        current_cache = caches[i]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(i+2)], current_cache, "Relu")
        grads["dA" + str(i+1)] = dA_prev_temp
        grads["dW" + str(i+1)] = dW_temp
        grads["db" + str(i+1)] = db_temp
        
    return grads
    
def update_parameters(parameters, grads, learning_rate):
    
    L = int(len(parameters)/2)
    
    for i in range(L):
        parameters["W" + str(i+1)] = parameters["W" + str(i+1)] - learning_rate * grads["dW" + str(i+1)]
        parameters["b" + str(i+1)] = parameters["b" + str(i+1)] - learning_rate * grads["db" + str(i+1)]
        
    return parameters
    
def L_layer_model(X, Y, layer_dims, num_iterations, learning_rate, print_cost = False):
    
    np.random.seed(1)
    cost = []
    
    parameters = initialize_parameters(layer_dims)
    
    for i in range(num_iterations):
        
        AL,caches = L_model_forward(X,parameters)
                
        cost = cost_function(AL, Y)
        
        grads = L_model_backward(AL, Y, caches)
        
        parameters = update_parameters(parameters, grads, learning_rate)
        
        if print_cost and i%100 == 0:
            print('after %i interation the cost is:%f' %(i, cost))
    
    return parameters

layer_dims = [12288,20,7,5,1]
parameters = L_layer_model(train_set_x, train_set_y, layer_dims, num_iterations = 2500, learning_rate = 0.0075, print_cost = True)

def predict(X, y, parameters):
    """
    This function is used to predict the results of a  L-layer neural network.
    
    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model
    
    Returns:
    p -- predictions for the given dataset X
    """
    
    m = X.shape[1]
    p = np.zeros((1,m))
    
    # Forward propagation
    probas, caches = L_model_forward(X, parameters)
 
    
    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
    
    print("Accuracy: "  + str(float(np.sum((p == y))/m)))
        
    return p

predictions_train = predict(train_set_x, train_set_y, parameters)
predictions_train = predict(test_set_x, test_set_y,parameters)

my_image = "123456.jpg"   # change this to the name of your image file 
my_label_y = [1]
## END CODE HERE ##
fname = my_image
image = np.array(ndimage.imread(fname, flatten=False))
my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((num_px*num_px*3,1))
my_predicted_image = predict(my_image, my_label_y, parameters)

plt.imshow(image)
print ("y = " + str(np.squeeze(my_predicted_image)) + ", your L-layer model predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")