import numpy as np

model = 3

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
