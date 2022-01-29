import numpy as np
from config import *

if HMM_model == 1:  # Flips of a fair coin and a biased coin alternately

    # Symbols index:    0: head     1: tail
    n_symbols = 2
    # States index:     0: A (fair) 1: B (biased)
    n_states = 2

    # Initial probabilities
    # A     B
    I = np.array([0.5,0.5])

    # Transition probabilities
    # AA     AB
    # BA     BB
    T = np.array([[0.99,0.01], 
                  [0.3 ,0.7 ]])

    # Emission probabilities
    # A0    A1
    # B0    B1
    E = np.array([[0.5,0.5],
                  [0.1,0.9]])

elif  HMM_model == 2:  # Dishonest casino

    n_symbols = 6
    n_states  = 2

    # Indexes of states:
    # fair: 0   loaded: 1
    I = np.array([0.5,0.5])

    # Transition matrix:
    # fair-fair     fair-loaded
    # loaded-fair   loaded-loaded
    T = np.array([[0.95,0.05], 
                  [0.1 ,0.9 ]])

    # Emission matrix:
    # fair-1 ... fair-6
    # loaded-1 ... loaded-6
    E = np.array([[1/6,1/6,1/6,1/6,1/6,1/6],
                  [0.1,0.1,0.1,0.1,0.1,0.5]])

elif  HMM_model == 3:  # DNA simulation (3 states)

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
    T = np.array([[0.9,0.1,0.0], 
                  [0.0,0.9,0.1],
                  [0.0,0.0,1.0]])

    # Emission matrix
    # 00    01      02      03
    # 10    11      12      13
    # 20    21      22      23
    E = np.array([[0.25,0.25,0.25,0.25],
                  [0.1,0.4,0.4,0.1],
                  [0.25,0.25,0.25,0.25]])
