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


def fn_S(sigma,num_size,i):
    """
    Args:
        sigma: a vector whose elements are 1 or -1 that represents a 1D ising model's configuration
        i: index of sigma

    Return:
        S: sum of the values of the two neighboring spins
    """
    temp = num_size-1

    if i == 0:
        S = sigma[i+1]
    if i == temp:
        S = sigma[i-1]
    else:
        S = sigma[i-1] + sigma[i+1]

    return S


def fn_sigma(num_size):
    """
    Args:
        num_size: number of spins

    Return:
        sigma: initial random configuration of 1D ising model
    """
    sigma = np.random.rand(num_size)

    for i in range(num_size):
        if sigma[i] < 0.5:
            sigma[i] = -1
        else:
            sigma[i] = 1

    sigma = np.array(sigma)
    return sigma


def GlauberDynamics(sigma, N, T):
    """
    Args:
        N: number of iterations of the simulation of Glauber Dynamics
        T: temperature

    Return:
        sigma_output: a configuration of 1D ising model that can be regard as a sample from stationary distribution
    """
    for num_iteration in range(N):
        temp = np.random.random_integers(0, num_size - 1, 1)
        i = temp[0]  # pick a random site

        delta_E = 2 * sigma[i] * fn_S(sigma, num_size, i)

        if delta_E < 0:
            sigma[i] = -sigma[i]
        else:
            prob = math.exp(-delta_E / T)
            u = np.random.rand(1)
            u = u[0]
            if u < prob:
                sigma[i] = -sigma[i]

    sigma_output = sigma
    return sigma_output


def convert_sigma_to_decimal(sigma, num_size):
    """
    Args:
        num_size: number of spins
        sigma: a specific configuration of 1D ising model

    Return:
        decimal: the decimal integer corresponding to the configuration sigma
    """
    decimal = 0
    for i in range(num_size):
        if sigma[i] == 1:
            decimal = decimal + (2 ** (num_size - i - 1))

    return decimal
# Initialize values of parameters

def convert_decimal_to_sigma(decimal, num_size):
    """
    Args:
        num_size: number of spins
        decimal: a decimal integer

    Return:
        sigma: the configuration of 1D ising model corresponding to the decimal integer
    """
    sigma = []

    for j in range(num_size):
        k = num_size - j - 1

        bit_number = decimal // (2 ** k)
        if bit_number == 0:
            bit_number = -1

        decimal = decimal % (2 ** k)
        sigma.append(bit_number)

    return sigma


def fn_partition(num_size, T):
    """
    Args:
        num_size: number of spins

    Return:
        partition: the partition function of the 1D ising model given the number of sites
    """
    partition = 0

    for i in range(2 ** num_size):
        sigma = convert_decimal_to_sigma(i, num_size)

        E = 0
        for j in range(num_size - 1):
            E = E - sigma[j] * sigma[j + 1]

        partition = partition + math.exp(-E / T)

    return partition

def B_distribution(num_size,T,sigma,partition):
    """
    Args:
        num_size: number of spins
        T: temperature
        sigma: the configuration whose probability is what we want to compute
        partition: the partion function of the 1D ising model given the number of sites

    Return:
        p: probability of sigma
    """
    E = 0
    for j in range(num_size-1):
        E = E - sigma[j]*sigma[j+1]

    P = math.exp(-E/T)/partition

    return P
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
def Creat_state(N):
    '''
    :param N:the number of spins
    :return: all state of spins
    '''
    m = 2**N
    M = np.zeros([m,N])
    for i in range (m):
        M[i] = convert_decimal_to_sigma(i,N)
    return  M
def cir_matrix(d,pattern):
    """
    Given a such pattern, compute the adjacency matrix of the circulant graph.

    Args:
        d: number of vertices
        pattern: a vector whose length is d and the first element should be 0

    Return:
        A: the adjacency matrix of the circulant graph induced by pattern
    """
    A = [pattern]
    for i in range(d-1):
        temp = pattern[d-1] # take the first element of vector pattern into memory.
        pattern_ = np.delete(pattern,d-1)
        pattern_ = np.append([temp],pattern_)
        A = np.concatenate((A,[pattern_]),axis=0)
        pattern = pattern_

    return A
def direct_sum(A,m):
    """
    Compute the direct sum of two matrices A.

    Args:
        A: the m*m symmetric matrix
        m: dimension of A

    Return:
        cal_A: direct sum of two A
    """
    O = np.zeros((m,m)) # Define O: a zero m*m matrix
    cal_A = np.block([[A,O],[O,A]])
    return cal_A
def fn_Q(A, m):
    """
    Compute matrix Q = ( I-np.dot(X,cal_A) )^(-1)

    Args:
        A: the m*m symmetric matrix
        m: number of modes

    Return:
        Q: matrix Q = ( I-np.dot(X,cal_A) )^(-1)
    """
    I = np.identity(m)
    O = np.zeros((m, m))
    X = np.block([[O, I], [I, O]])
    cal_A = direct_sum(A, m)

    Q_temp = np.identity(2 * m) - np.dot(X, cal_A)
    Q = la.inv(Q_temp)

    return Q
def fn_x_GBS_mean(Q,m):
    """
    Compute the average result of threshold detectors on the k-th mode.

    Args:
        k: which denotes the k-th mode
        Q: Q = ( I-np.dot(X,cal_A) )^(-1)
        m: number of modes

    Return:
        x_k_GBS_mean: the average result of threshold detectors on each mode.
    """
    x_GBS_mean = []
    for k in range(m):
        Q1 = Q[k][k]
        Q2 = Q[k][k+m]
        Q3 = Q[k+m][k]
        Q4 = Q[k+m][k+m]
        Qk = [[Q1,Q2],[Q3,Q4]]
        temp = la.det(Qk)
        x_k_GBS_mean = 1 - temp**(-0.5)
        x_GBS_mean.append(x_k_GBS_mean)

    return x_GBS_mean
def fn_cost_grad(Q, x_data_mean, f, m):
    """
    Compute the gradient of KL divergence cost function.

    Args:
        Q: Q = ( I-np.dot(X,cal_A) )^(-1)
        x_data_mean: a vector whose element is each mode's mean value of datas
        f: a d*d matrix, and each row of it denotes the WAW parametrization vector with respect to w_k.
        m: number of modes
        d: dimension of vector theta and f[k]

    Return:
        cost_grad: a vector whose k-th element is the gradient of the KL cost function with respect to theta_k.
    """
    x_GBS_mean = fn_x_GBS_mean(Q, m)

    temp = np.zeros(m)
    for k in range(m):
        temp = temp + (x_data_mean[k] - x_GBS_mean[k]) * f[k]
    cost_grad = temp
    return cost_grad
def fn_training_data(A, w_model, n_mean, n_samples, threshold, loss):
    """
    Generate a matrix which represents the set of training data, and each row of the matrix is actually one data/sample.
    Compute the average result of the datas on each mode.

    Args:
        A: the m*m symmetric matrix
        w_model: weights vector of the model

    Return:
        training_data: the data/sample matrix
        x_data_mean: the average vector of datas
    """
    Aw = np.dot(w_model,np.dot(A, w_model))
    training_data = sample.sample(Aw, n_mean, n_samples, threshold, loss)

    training_data = np.array(training_data)
    x_data_mean = np.average(training_data, axis=0)
    return x_data_mean
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

num_size = 7 # number of sites
T = 4 # temperature
k = 1.38064852*10**-23 # Boltzmann constant
beta = 1/(k*T)

# number of iteration times. This number needs to be large enough to every site will be visited once. After many iterations, all the sites will get equal attention.
N = 1000*(2**num_size)

num_samples = 3200 # number of output samples

sigma = fn_sigma(num_size) # initialize configuration at first
print("Initial Configuration \sigma=:", "\n", sigma, "\n")

# Doing Glauber Dynamics# Doing
sigma0 = GlauberDynamics(sigma,N,T)
samples = [np.zeros(num_size)]
for k in range(num_samples):
    if (k + 1) % 1000 == 0:
        print('Progress Bar = :', (k + 1) / num_samples)
    sigma0 = GlauberDynamics(sigma0, 2 ** num_size, T)
    samples = np.concatenate((samples, [sigma0]), axis=0)

samples = np.delete(samples, 0, axis=0)
print("\n", "Number of Samples:=", np.shape(samples), "\n")
t = np.shape(samples)
Samples = samples
'''
for i in range (t[0]):
    for j in range(t[1]):
        if samples[i,j]<0:
            samples[i,j]=0

print("Samples:=", "\n", samples, "\n")
Prob_Sample = np.zeros(2**num_size)
Prob = []
for i in range(num_samples):
    sigma = samples[i]
    sigma_decimal = convert_sigma_to_decimal(sigma, num_size)
    sigma_decimal = int(sigma_decimal)
    Prob_Sample[sigma_decimal] = Prob_Sample[sigma_decimal]+1
Prob_Sample = Prob_Sample/num_samples
'''
partition = fn_partition(num_size,T)
Prob = []
for i in range(2**num_size):
    sigma_B = convert_decimal_to_sigma(i, num_size)
    P = B_distribution(num_size,T,sigma_B,partition)
    Prob.append(P)
'''
# Compute the probability distribution by counting samples
Prob_Sample = np.zeros(2**num_size)
for i in range(num_samples):
    sigma = samples[i]
    sigma_decimal = convert_sigma_to_decimal(sigma, num_size)
    sigma_decimal = int(sigma_decimal)
    Prob_Sample[sigma_decimal] = Prob_Sample[sigma_decimal]+1
Prob_Sample = Prob_Sample/num_samples

Conf = range(2**num_size)
pl.figure(1)
pl.plot(Conf,Prob,'red')
pl.bar(Conf,Prob_Sample)
pl.xlabel("Configuration")
pl.ylabel("Probability")
pl.title("")
pl.show()
'''
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
N = num_size
AA=Creat_state(N)
t = np.zeros(2**N)
for i in range(2**N):
    t[i] =Ising_Hamilonian(AA[i],N)
Sample = Samples
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
        tt = convert_sigma_to_decimal(Sample[l],N)
        if tt in r1[0]:
            n[kk]=n[kk]+1
    kk = kk+1
n = n/num_samples
print(n)

#GBS

n_means = 1
n_samples = 5000
threshold = True
loss = 0.0
eta = 0.1 # learning rate
num_iter = 200 # number of iteration steps
#x_data_mean = np.average(samples, axis=0)
x_data_mean = n
d = num_size
A = np.ones([d,d])
for i in range(d):
    for j in range(d):
        if i < j:
            A[i, j] = random.randint(0, 1)
            A[j, i] = A[i, j]
print(A)
f = np.identity(d)
theta0 = np.ones(d)*1
w0 = np.zeros(d)
for i in range (num_iter):
    for j in range(d):
        w0[j] = np.exp(-1 * np.inner(theta0, f[j]))
    print("w0=:", w0, "\n")
    W0 = np.diag(w0)
    Aw = np.dot(W0,np.dot(A,W0))
    print(Aw)
    Q = fn_Q(Aw,d)
    print(Q)
    n_gbs = fn_x_GBS_mean(Q,d)
    print(n_gbs)
    grad = fn_cost_grad(Q,x_data_mean,f,d)
    theta0 = theta0 - eta*grad
S = sample.sample(Aw,n_means,n_samples,True,0.0)
ss = np.sum(S,axis=0)
print(ss)
'''
next step: sample the states of the same energy
'''
AAA=Creat_state(d)
t = np.zeros(2**d)
for i in range(2**d):
    t[i] =Ising_Hamilonian(AAA[i],d)
SS = AAA
result = np.zeros(2 ** d)
Sample = SS
M = -np.ones([2**d,5])
m = np.zeros(d)
n = np.zeros(d)
for j in range(d):
    m[j]=-(N-1)+j*2
kk = 0
ttt = np.shape(Sample)
ssss =ss
result = np.zeros(2**d)
for k in m:
    r = ssss[kk]
    r1 = np.where(t==k)
    print(r1[0][1])
    l = len(r1[0])
    print(l)
    sampling = uniform_sampling(l,r)
    for kkk in range(l):
        point = r1[0][kkk]
        print(point)
        print(kkk)
        result[point]=sampling[kkk]
    kk = kk+1
print(result/sum(ssss))
result = result/sum(ssss)
Conf = range(2**num_size)
pl.figure(1)
pl.plot(Conf,Prob,'red')
pl.bar(Conf,result)
pl.xlabel("Configuration")
pl.ylabel("Probability")
pl.title("4K")
pl.show()

'''
S_Sample = np.zeros(2**num_size)
for i in range(n_samples):
    sigma = S[i]
    sigma_decimal = convert_sigma_to_decimal(sigma, num_size)
    sigma_decimal = int(sigma_decimal)
    S_Sample[sigma_decimal] = S_Sample[sigma_decimal]+1
S_Sample = S_Sample/n_samples
print(S_Sample)

'''