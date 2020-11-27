import strawberryfields as sf
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
import math
import networkx as nx
from strawberryfields import decompositions as dp
from strawberryfields.ops import *
from strawberryfields.apps import sample
from strawberryfields.apps import plot
from numpy import linalg as la
import random
import copy
'''   
The purpose of this code is to demo the second result in training GBS distribution 
——generative model/ unsupervised learning 
task:
We have a given distribution of each mode and by training GBS, we hope the samples from GBS have the same distribution 
as the input one
Training GBS distribution: you can find the paper in https://arxiv.org/abs/2004.04770.pdf
And the training algorithm refers the same paper.
'''
def Cir_Graph(S):
    '''
    The purpose of this function is to get a Circular graph as the input matrix A
    :param S:the input mode distribution
    :return:The whole graph
    '''
    N = len(S)
    Result = np.zeros((N,N))
    Result[0] = S
    for i in range(N-1):
        for j in range(N-1):
            if i<j:
                Result[i+1,j+1] = S[j-i]
    for k in range(N):
        for m in range(N):
            if k>m:
                Result[k,m]=Result[m,k]
    return Result

def matrix_direct_sum(A,N):
    '''
    the purpose of this function is to calculate the direct sum of a given N-dimension matrix
    :param A: the given matrix
    :param N: the dimension of the matrix
    :return: the result of A (direct sum) A
    '''
    ZEROS = np.zeros((N,N))
    Result = np.block([[A,ZEROS],[ZEROS,A]])
    return  Result
def x_k_GBS(A,N):
    '''
    The purpose of this function is to calculate the expression(43) of "Training GBS distribution"
    :param A: The input matrix corresponding to the input mode
    :param N: The mode GBS used
    :return: The <x_k>_{GBS} in each mode by sampling(1*N array)
    '''
    I = np.identity(N)
    ZEROS = np.zeros((N,N))
    X = np.block([[ZEROS,I],[I,ZEROS]])
    I_2N = np.identity(2*N)
    AA = matrix_direct_sum(A,N)
    Q = np.linalg.inv(I_2N-np.dot(X,AA))
    x_k = np.zeros(N)
    for k in range(N):
        Q_1 = Q[k][k]
        Q_2 = Q[k][k+N]
        Q_3 = Q[k+N][k]
        Q_4 = Q[k+N][k+N]
        Q_k = [[Q_1,Q_2],[Q_3,Q_4]]
        t = np.linalg.det(Q_k)
        x_k[k] = 1-t**(-0.5)
    return x_k
def Grad(x_gbs,x_data,f):
    '''
    the purpose is to calculate the gradient of given data
    :param x_gbs: the average number calculated by <x_k>——corresponding to the threshold detectors
    :param x_data: the average number of photons in each mode
    :param f: the training matrix
    :return: the gradient of theta
    '''
    N = len(x_gbs)
    t = np.zeros(N)
    for i in range(N):
        t = t +(x_data[i]-x_gbs[i])*f[i]
    return t
def linear_increase(N):
    #   crate a linear increase weight matrix
    #   N: the mode of photons
    w = np.linspace(0.2,0.65,N)
    W = np.diag(w)
    return W
def linear_decrease(N):
    #   crate a linear decrease weight matrix
    #   N: the mode of photons
    w = np.linspace(0.5,0.1,N)
    W = np.diag(w)
    return W
def Fid(w1, w2):
    #   to describe the similarity of two matrix
    #   W1 is the target weight matrix; W2 is the trained weight matrix
    w1 = np.array(w1)
    w2 = np.array(w2)
    w = w1 - w2
    w = np.dot(np.transpose(w), w)
    a1, b1 = np.linalg.eig(w)
    lambda1 = np.max(abs(a1))
    fidelity = np.sqrt(abs(lambda1))
    return fidelity
# args:
T = 5000
t = 2000
n_k = 10
S = [0,1,1,0,0,1,1]
N = len(S)
theta = 1 * np.ones(N)
f = np.ones(N)
F = np.diag(f)
F = np.array(F)
speed = 0.1
A = Cir_Graph(S)
print("The initial matrix:\n",A)
W_data = linear_increase(N)
print("The training target:\n",W_data)
A_data = np.dot(W_data,np.dot(A,W_data))
S_data = sample.sample(A_data, n_k, T, True, loss=0.0)    #   sampling for data
S_data = np.array(S_data)
n_data = np.average(S_data, axis=0)#   the mean number of photons for Sampling
n_data = np.array(n_data)
print(n_data)
w = np.zeros(A.shape[0])
for i in range(A.shape[0]):
    w[i]=np.exp(-theta[i])
print(w)
W_gbs = np.diag(w)
x = np.zeros(t)
y = np.zeros(t)
for i in range(t):
    fid = Fid(W_data,W_gbs)
    print("iter:",i,"norm=",fid)
    x[i]=i
    y[i]=fid
    A_gbs = np.dot(W_gbs,np.dot(A,W_gbs))
    n_gbs = x_k_GBS(A_gbs,N)
    print(n_gbs)
    grad = Grad(n_gbs,n_data,F)
    theta = theta - speed* grad
    for i in range(A.shape[0]):
        w[i]=math.exp(-theta[i])
    W_gbs = np.diag(w)
    print(w)
print("average number of photons by GBS sampling:= \n",n_gbs)
plt.figure()
plt.xlabel('cycling')
plt.ylabel('possibility')
plt.plot(x,y)
plt.show()