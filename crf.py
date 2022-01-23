import numpy as np
from models import * 

verbose = True  # Print the detailed execution log

# Converting the probabilities of model to log in base 2
I = np.log2(I)  # Initial probabilities
A = np.log2(A)  # Transition matrix
B = np.log2(B)  # Emission matrix


### HMM FEATURE FUNCTION

class f_HMM:
    
    w = 1.0  # weight of feature function

    def f(t,y_t,y_t_minus_1,x):  # feature function

        emission_prob = B[y_t,x[t]]
        
        if t == 0:
            return emission_prob + I[y_t]
        else:
            return emission_prob + A[y_t_minus_1,y_t]


### FEATURES FUNCTIONS LIST

ff_list = []  
ff_list.append(f_HMM)

## Return the weight log sum of features functions results of a time t of sequence x
#
def psi(t,y_t,y_t_minus_1,x):
    log_sum_features = 0
    for ff in ff_list:
        log_factor = ff.w * ff.f(t,y_t,y_t_minus_1,x)
        if log_sum_features == 0:  # log(0) is indetermined
            log_sum_features = log_factor
        else:
            log_sum_features = np.logaddexp2(log_sum_features, log_factor) 
    return log_sum_features

## Return the Viterbi path
#
def viterbi(x):

    n_positions = x.size
    viterbi_matrix = np.zeros((n_states,n_positions))
    viterbi_paths = np.zeros((n_states,n_positions))

    for t in range(n_positions):

        for j in range(n_states):
    
            position_probs = np.zeros(n_states)

            for i in range(n_states):

                position_probs[i] = psi(t,j,i,x)

                if t != 0: 
                    position_probs[i] += viterbi_matrix[i,t-1]
                
            viterbi_matrix[j,t] = np.amax(position_probs)

            if t != 0: 
                viterbi_paths[j,t-1] = np.argmax(position_probs)
    
    # Finishing the paths:
    for j in range(n_states):
        viterbi_paths[j,n_positions-1] = j

    probability_best_path = np.amax(viterbi_matrix[:,n_positions-1])
    best_path = viterbi_paths[np.argmax(viterbi_matrix[:,n_positions-1])]

    if verbose:
        print("\nVITERBI DECODING")
        print("\nViterbi matrix:")
        print (viterbi_matrix)
        print("\nPaths: ")
        print (viterbi_paths)
        print("\nBest path: ")
        print(best_path)
        print("Log probability of best path: ", probability_best_path)

    return best_path





# sequence = np.array([0,1,1])
# sequence = np.array([0,0,0,0,1,0,1,1,1,0])

sequence = np.array([0,1,2,3,0,1,2,3])
viterbi(sequence)
