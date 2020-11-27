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
   The purpose of this code is to generate a set of one-dimensional Ising model by Glauber Dynamics
   The algorithm part refers to the following literature(Chinese lecture):
   http://micro.ustc.edu.cn/CompPhy/lecturenote/11.pdf 
   In the literature, This sampling method is named Metropolis sampling, so in my code 
   the name and the function is Metropolis
'''
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
        S.append(Number_to_chain(i, 5))
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

N = 5
T = 0.9
num_itr = 4000
sample_numbers = 3200
s = Creat_all_state(N)
s1 = s[11]
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
print("\n","Probability of each state by Sampling \n",Prob_Sample/sample_numbers,"\n")
print("\n","Probability of each state by calculating \n",Boltzmann_Distribution(N,T),"\n")
Conf = range(2**N)
pl.figure(1)
pl.plot(Conf,Prob,'red')
pl.bar(Conf,Prob_Sample)
pl.xlabel("Configuration")
pl.ylabel("Probability")
pl.title("")
pl.show()