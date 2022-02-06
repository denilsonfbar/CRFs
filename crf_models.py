from hmm_models import *

## HMM represented with one real feature function and weight = 1.0
def f_HMM(t,y_t,y_t_minus_1,x):
    emission_prob = E[y_t,x[t]]
    if t == 0:
        return np.log(emission_prob * I[y_t])
    else:
        return np.log(emission_prob * T[y_t_minus_1,y_t])

## HMM with 2 hidden states represented with 10 indicator features functions 
#  and weights corresponding to the probabilities of the HMM model
def f0(t,y_t,y_t_minus_1,x):
    if t == 0 and y_t == 0: return 1  # State 0 in first position
    else: return 0
def f1(t,y_t,y_t_minus_1,x):
    if t == 0 and y_t == 1: return 1  # State 1 in first position
    else: return 0
def f2(t,y_t,y_t_minus_1,x):
    if y_t == 0 and y_t_minus_1 == 0: return 1  # Transition states 0 -> 0
    else: return 0
def f3(t,y_t,y_t_minus_1,x):
    if y_t == 1 and y_t_minus_1 == 0: return 1  # Transition states 0 -> 1
    else: return 0
def f4(t,y_t,y_t_minus_1,x):
    if y_t == 0 and y_t_minus_1 == 1: return 1  # Transition states 1 -> 0
    else: return 0
def f5(t,y_t,y_t_minus_1,x):
    if y_t == 1 and y_t_minus_1 == 1: return 1  # Transition states 1 -> 1
    else: return 0
def f6(t,y_t,y_t_minus_1,x):
    if y_t == 0 and x[t] == 0: return 1  # Emission of 0 in state 0
    else: return 0
def f7(t,y_t,y_t_minus_1,x):
    if y_t == 0 and x[t] == 1: return 1  # Emission of 1 in state 0
    else: return 0
def f8(t,y_t,y_t_minus_1,x):
    if y_t == 1 and x[t] == 0: return 1  # Emission of 0 in state 1
    else: return 0
def f9(t,y_t,y_t_minus_1,x):
    if y_t == 1 and x[t] == 1: return 1  # Emission of 1 in state 1
    else: return 0

## FEATURES FUNCTIONS LIST
ff_list = []

# WEIGHTS VECTOR
W = np.full((1), 0.0)

# HMM represented with one real feature function and weight = 1.0
if CRF_model == 1:  
    ff_list.append(f_HMM)
    W[0] = 1.0

## HMM with 2 hidden states represented with 10 indicator features functions 
#  and weights corresponding to the probabilities of the HMM model
elif CRF_model == 2:
    W = np.full((10), 0.0)
    ff_list.append(f0)  # State 0 in first position
    W[0] = np.log(I[0])
    ff_list.append(f1)  # State 1 in first position
    W[1] = np.log(I[1])
    ff_list.append(f2)  # Transition states 0 -> 0
    W[2] = np.log(T[0,0])
    ff_list.append(f3)  # Transition states 0 -> 1
    W[3] = np.log(T[0,1])
    ff_list.append(f4)  # Transition states 1 -> 0
    W[4] = np.log(T[1,0])
    ff_list.append(f5)  # Transition states 1 -> 1
    W[5] = np.log(T[1,1])
    ff_list.append(f6)  # Emission of 0 in state 0
    W[6] = np.log(E[0,0])
    ff_list.append(f7)  # Emission of 1 in state 0
    W[7] = np.log(E[0,1])
    ff_list.append(f8)  # Emission of 0 in state 1
    W[8] = np.log(E[1,0])
    ff_list.append(f9)  # Emission of 1 in state 1
    W[9] = np.log(E[1,1])
