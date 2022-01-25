import numpy as np
from models import * 

verbose = True  # print the detailed execution informations


### HMM FEATURE FUNCTION
class f_HMM:
    
    w = 1.0  # weight of feature function

    ## Return the log2 value of feature function of a time t of a sequence x
    #
    def f(t,y_t,y_t_minus_1,x):  # feature function
        emission_prob = B[y_t,x[t]]
        if t == 0:
            return np.log2(emission_prob * I[y_t])
        else:
            return np.log2(emission_prob * A[y_t_minus_1,y_t])


### FEATURES FUNCTIONS LIST
ff_list = []  
ff_list.append(f_HMM)


## Return the weighted log sum of features functions results of a time t of a sequence x
#
def log_sum_features(t,y_t,y_t_minus_1,x):
    for i in range(len(ff_list)):
        weighted_log_factor = ff_list[i].w * ff_list[i].f(t,y_t,y_t_minus_1,x)
        if i == 0:
            log_sum = weighted_log_factor
        else:
            log_sum = np.logaddexp2(log_sum, weighted_log_factor) 
    return log_sum


## Return the Viterbi path
#
def viterbi(x):
    n_positions = x.size
    viterbi_matrix = np.full((n_states,n_positions), np.NINF)
    viterbi_paths = np.full((n_states,n_positions), -1)

    for t in range(n_positions):
        for j in range(n_states):
            log_position_probs = np.full((n_states), np.NINF)
            for i in range(n_states):
                log_position_probs[i] = log_sum_features(t,j,i,x)
                if t != 0: 
                    log_position_probs[i] += viterbi_matrix[i,t-1]
            viterbi_matrix[j,t] = np.amax(log_position_probs)
            if t != 0: 
                viterbi_paths[j,t-1] = np.argmax(log_position_probs)
    
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
