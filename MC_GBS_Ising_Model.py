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
The purpose is to get a Boltzmann distribution of n nodes by n modes
This is a bad try because it even doesn't make use of GBS. 
Anyway, I still think we can design a coding method for WAW method to get a really useful way to get Ising or other model
'''
def Creat_clique(N,clique):
    A = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if i < j:
                A[i, j] = random.randint(0, 1)
                A[j, i] = A[i, j]
            if clique[i] == clique[j] == 1:
                A[i, j] = 1
            if i == j:
                A[i, j] = 0
    return A
def Hamilton(S):
    '''
    The purpose of this function is calculate the Hamiltonian of the input the 1D spin chain
    :param N: Number of spins
    :param S: S: 1-D spin chain(1*N array)
    :return: The Hamiltonian of the input spin chain
    '''
    hamilton = 0
    N = len(S)
    for i in range(N-1):
        hamilton = hamilton + S[i]*S[i+1]
    hamilton = -hamilton
    return hamilton
def Chain_to_number(Ising_model):
    '''
    The purpose of this function is calculate the number corresponding to the given 1D spin chain
    :param S: 1-D spin chain(1*N array)
    :return: The number corresponding to the given chain
    '''
    t = 0
    N = len(Ising_model)
    S = copy.deepcopy(Ising_model)
    for i in range(N):
        if S[i]<0:
            S[i]=0
    for j in range(N):
        t = t+ S[j]*(2**(N-1-j))
    return t
def Number_to_chain(N,number):
    '''
    The purpose of this function is to convert number to a 1D-Ising chain
    :param N: The decimalism number
    :param number: the number of input spins
    :return: the 1D-Ising chain
    '''
    S = []
    while N:
        if N % 2 == 1:
            S.append(1)
        else:
            S.append(0)
        N = N // 2
    S.reverse()
    S = np.array(S)
    n = len(S)
    for i in range(n):
        if S[i] == 0:
            S[i] = -1
    Result=-np.ones(number)
    t = number - n
    for j in range(n):
        Result[j+t]=S[j]
    return Result
def Creat_all_state(N):
    '''
    The purpose of this function is to creat all the one-dimensional Ising model of numebr N
    :param N: the number of Ising model
    :return:one-dimensional chain
    '''
    S=[]
    for i in range(2**N):
        S.append(Number_to_chain(i, N))
    S = np.array(S)
    return S
def delta_E(S1,S2):
    '''
    The purpose of this function is to calculate the energy difference between S1-spin Chain and S2-spin Chain
    :param S1: the input spin-chain 1
    :param S2: the input spin-chain 2
    :return: the energy difference
    '''
    E1 = Hamilton(S1)
    E2 = Hamilton(S2)
    delta_E=E2 - E1
    return delta_E
def Metropolis(s,n,T):
    '''
    The purpose of this function is to get one sample of Ising model at tem=T
    :param s: The initial input Ising chain
    :param T: The temperature of Ising chain
    :return: one sample of Ising model at temperature is T
    '''
    N = len(s)

    for i in range (n):
        s1 = copy.deepcopy(s)
        t = random.randint(0,N-1)
        s1[t] = -s[t]
        dE= delta_E(s,s1)
        if dE<0:
            s[t] = -s[t]
        else:
            r = math.exp(-dE/T)
            R = random.random()
            if R<r:
                s[t] = -s[t]
    s = np.array(s)
    return s
def Boltzmann_Distribution(n,T):
    '''
    The propose of this function is to calculate the Boltzmann Distribution of 1D-Ising ground state
    :param n: number of spins
    :param T: Temperature
    :param s: 1D Ising Chain
    :return:The Boltzmann Distribution of 1D-Ising ground state
    '''
    S = Creat_all_state(n)
    Z = 0
    p = np.zeros(2**n)
    for i in range(2**n):
        E = Hamilton(S[i])
        p[i] = math.exp(-E/T)
    p = p/sum(p)
    return p
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
def uniform_sampling(N,M):
    '''

    :param N:  the number of types
    :param M: the number of samples
    :return: the sampling result
    '''
    result = np.zeros(N)
    for i in range(M):
        t = random.random()
        for j in range(N):
            if t>j/N and t <(j+1)/N:
                result[j]=result[j]+1
    return result
def Ising_Hamilonian(sigma,l):
    '''
    :param sigma: input state
    :param l:the number of input state
    :return: energy of input state
    '''
    t = 0
    for i in range(l-1):
        t = t+sigma[i]*sigma[i+1]
    return t
N = 6
T = 1
num_itr = 4000
sample_numbers = 3200
s = Creat_all_state(N)
tt = random.randint(0,2**N-1)
s1 = s[tt]
Prob_Sample = np.zeros(2**N)
s1 = Metropolis(s1,num_itr,T)
S = np.zeros((sample_numbers,N))
for i in range(sample_numbers):
    if (i+1)%1000==0:
        print('Progress Bar = : ',(i+1)/ sample_numbers)
    S[i] = Metropolis(s1,40,T)
    t = Chain_to_number(S[i])
    t = int(t)
    Prob_Sample[t]=Prob_Sample[t]+1
Prob = Boltzmann_Distribution(N,T)
Prob_Sample = Prob_Sample/sample_numbers
print("\n","Number of Samples:= \n ",np.shape(S),"\n")
print("\n","Samples:=\n",S,"\n")
print("\n","Probability of each state by Sampling \n",Prob_Sample,"\n")
print("\n","Probability of each state by calculating \n",Boltzmann_Distribution(N,T),"\n")
'''
part 1:
data generation:
above:generate a set of data, which is used to show the distribution of 1D Ising model
'''
'''
part 2:
data pre-processing:
thought: Because there are lots of states that are degenerate, in principle, the states with same energy will have the same probability of occuring
if we take different energy as input and sample the output again, we will get the distribution with less modes
method: 
1. calculate the energy of each possible sequence
2. find the same sequences of energy
3. calculate the probability of different energy
4. calculate the number of states at different energy
'''
AA=Creat_all_state(N)
t = np.zeros(2**N)
for i in range(2**N):
    t[i] =Ising_Hamilonian(AA[i],N)
Sample = S
M = -np.ones([2**N,5])
m = np.zeros(N)
n = np.zeros(N)
for j in range(N):
    m[j]=-(N-1)+j*2
kk = 0
ttt = np.shape(Sample)
for k in m:
    r1 = np.where(t==k)
    for l in range(ttt[0]):
        tt = Chain_to_number(Sample[l])
        if tt in r1[0]:
            n[kk]=n[kk]+1
    kk = kk+1
n = n/sample_numbers
print("The possibility of each energy at T:\n",n)
'''
GBS
'''
n_means = 1
n_samples = 5000
threshold = True
loss = 0.0
speed = 0.1 # learning rate
num_iter = 200 # number of iteration steps
n_data = n
d = N
A = np.ones([d,d])
for i in range(d):
    for j in range(d):
        if i < j:
            A[i, j] = random.randint(0, 1)
            A[j, i] = A[i, j]
print("the initial matrix A:\n",A)
f = np.ones(d)
F = np.diag(f)
theta = np.ones(d)*1
w = np.zeros(d)
for i in range(A.shape[0]):
    w[i]=np.exp(-theta[i])
W_gbs = np.diag(w)
for i in range(num_iter):
    #fid = Fid(W_data,W_gbs)
    #   print("iter:",i,"norm=",fid)
    A_gbs = np.dot(W_gbs,np.dot(A,W_gbs))
    n_gbs = x_k_GBS(A_gbs,d)
    grad = Grad(n_gbs,n_data,F)
    theta = theta - speed* grad
    for i in range(d):
        w[i]=math.exp(-theta[i])
    W_gbs = np.diag(w)
print("The energy distribution by sampling:\n",n_gbs)
S = sample.sample(A_gbs,n_means,n_samples,True,0.0)
n_gbs_sampling = np.sum(S,axis = 0)
'''
next step: sample the states of the same energy
'''
All_State = Creat_all_state(d)
t = np.zeros(2**d)
for i in range(2**d):
    t[i] = Ising_Hamilonian(All_State[i],d)
result = np.zeros(2 ** d)
M = -np.ones([2**d,5])
m = np.zeros(d)
n = np.zeros(d)
for j in range(d):
    m[j] = -(N - 1) + j * 2     # This is all the energy
temp = 0
for k in m:
    r = n_gbs_sampling[temp]
    r1 = np.where(t==k)
    l = len(r1[0])
    sampling = uniform_sampling(l, r)
    for kk in range(l):
        point = r1[0][kk]
        result[point] = sampling[kk]
    temp = temp+1
result = (result/sum(n_gbs_sampling))
Conf = range(2**N)
pl.figure(1)
pl.plot(Conf,Prob,'red')
pl.bar(Conf,result)
pl.xlabel("Configuration")
pl.ylabel("Probability")
pl.show()
