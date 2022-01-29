import numpy as np
from config import *

if model == 1:  # Prof. André example

    # Symbols index:    0  1
    n_symbols = 2

    # States index:     A: 0   B: 1
    n_states = 2

    # Initial probabilities
    I = np.array([0.5,0.5])

    # Transition matrix
    # AA     AB
    # BA     BB
    A = np.array([[0.99,0.01], 
                  [0.3 ,0.7 ]])

    # Emission matrix
    # A0    A1
    # B0    B1
    B = np.array([[0.5,0.5],
                  [0.1,0.9]])

elif  model == 2:

    n_symbols = 6
    n_states  = 2

    # Indexes of states:
    # fair: 0   loaded: 1
    I = np.array([0.5,0.5])

    # Transition matrix:
    # fair-fair     fair-loaded
    # loaded-fair   loaded-loaded
    A = np.array([[0.95,0.05], 
                  [0.1 ,0.9 ]])

    # Emission matrix:
    # fair-1 ... fair-6
    # loaded-1 ... loaded-6
    B = np.array([[1/6,1/6,1/6,1/6,1/6,1/6],
                  [0.1,0.1,0.1,0.1,0.1,0.5]])

elif  model == 3:

    # Symbols index:    A:0   C:1   G:2   T:3
    n_symbols = 4

    # States index:     5UTR: 0     CDS: 1    3UTR: 2
    n_states = 3

    # Initial probabilities
    I = np.array([0.8,0.1,0.1])

    # Transition matrix
    # 00     01     02
    # 10     11     12
    # 20     21     22
    A = np.array([[0.9,0.1,0.0], 
                  [0.0,0.9,0.1],
                  [0.0,0.0,1.0]])

    # Emission matrix
    # 00    01      02      03
    # 10    11      12      13
    # 20    21      22      23
    B = np.array([[0.25,0.25,0.25,0.25],
                  [0.1,0.4,0.4,0.1],
                  [0.25,0.25,0.25,0.25]])

## HMM FEATURE FUNCTION
def f_HMM(t,y_t,y_t_minus_1,x):
    emission_prob = B[y_t,x[t]]
    if t == 0:
        return emission_prob * I[y_t]
    else:
        return emission_prob * A[y_t_minus_1,y_t]

## HMM represented in 10 indicator features functions
def f1(t,y_t,y_t_minus_1,x):
    if y_t == 0 and x[t] == 0: return 1
    else: return 0

def f2(t,y_t,y_t_minus_1,x):
    if y_t == 0 and x[t] == 1: return 1
    else: return 0

def f3(t,y_t,y_t_minus_1,x):
    if y_t == 1 and y_t_minus_1 == 0: return 1
    else: return 0

def f4(t,y_t,y_t_minus_1,x):
    if y_t == 1 and x[t] == 0: return 1
    else: return 0

def f5(t,y_t,y_t_minus_1,x):
    if y_t == 1 and x[t] == 1: return 1
    else: return 0

def f6(t,y_t,y_t_minus_1,x):
    if y_t == 1 and y_t_minus_1 == 1: return 1
    else: return 0

def f7(t,y_t,y_t_minus_1,x):
    if y_t == 0 and y_t_minus_1 == 1: return 1
    else: return 0

def f8(t,y_t,y_t_minus_1,x):
    if y_t == 0 and y_t_minus_1 == 0: return 1
    else: return 0

def f9(t,y_t,y_t_minus_1,x):
    if t == 0 and y_t == 0: return 1
    else: return 0

def f10(t,y_t,y_t_minus_1,x):
    if t == 0 and y_t == 1: return 1
    else: return 0

## FEATURES FUNCTIONS LIST
ff_list = []
n_features = 10
W = np.full((n_features), 0.0)  # weigths vector

if features_config == 1:
    n_features = 1
    ff_list.append(f_HMM)
    W[0] = 1.0
elif features_config == 2:
    ff_list.append(f1)
    W[0] = 0.5
    ff_list.append(f2)
    W[1] = 0.5
    ff_list.append(f3)
    W[2] = 0.01
    ff_list.append(f4)
    W[3] = 0.1
    ff_list.append(f5)
    W[4] = 0.9
    ff_list.append(f6)
    W[5] = 0.7
    ff_list.append(f7)
    W[6] = 0.3
    ff_list.append(f8)
    W[7] = 0.99
    ff_list.append(f9)
    W[8] = 0.5
    ff_list.append(f10)
    W[9] = 0.5
