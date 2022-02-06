from crf_models import *

X = []  # train set
X.append(np.array([0,1,1]))
X.append(np.array([0,0,0,0,1,0,1,1,1,0]))

def l(W):
    ret = 0.0
    for x in X:
        pass

def Z(x):
    L = n_states
    N = x.size  # length of sequence
    alpha_matrix = forward(x)
    Z_value = np.NINF
    for j in range(L):
        Z_value = np.logaddexp(Z_value,alpha_matrix[j,N-1])
    return Z_value



## FEATURES SUM
# Return the weighted sum of features functions results of a time t of a sequence x
def sum_features_t(t,y_t,y_t_minus_1,x):
    sum_weighted_factors = 0.0
    for i in range(len(ff_list)):
        weighted_factor = W[i] * ff_list[i](t,y_t,y_t_minus_1,x)
        sum_weighted_factors += weighted_factor
    return sum_weighted_factors

## FORWARD
# Receives a sequence x
# Returns the alpha matrix according current vector of weights W
def forward(x):  
    L = n_states
    T = x.size
    log_alpha_matrix = np.full((L,T),np.NINF)
    # Inicialization:
    for j in range(L):
        log_alpha_matrix[j,0] = sum_features_t(0,j,None,x)
    # Induction:
    for t in range(1,T):
        for j in range(L):
            for i in range(L):
                aux = sum_features_t(t,j,i,x) + log_alpha_matrix[i,t-1] 
                log_alpha_matrix[j,t] = np.logaddexp(log_alpha_matrix[j,t],aux)
    if verbose:
        print("\nFORWARD")
        print("Alpha matrix (log values):")
        print(log_alpha_matrix)
        print("Alpha matrix (real values):")
        print(np.exp(log_alpha_matrix))
        # Termination:
        seq_prob = np.NINF
        for j in range(L):
            seq_prob = np.logaddexp(seq_prob,log_alpha_matrix[j,T-1])
        print("Sequence probability:")
        print("log: ", seq_prob, "\treal: ", np.exp(seq_prob))
    return log_alpha_matrix

## BACKWARD
# Receives a sequence x
# Returns the beta matrix according current vector of weights W
def backward(x):  
    L = n_states
    T = x.size
    log_beta_matrix = np.full((L,T), np.NINF)
    # Inicialization:
    for j in range(L):
        for i in range(L):
            aux = sum_features_t(T-1,i,j,x)
            log_beta_matrix[j,T-1] = np.logaddexp(log_beta_matrix[j,T-1],aux)        
    # Induction:
    for t in reversed(range(1,T-1)):
        for j in range(L):
            for i in range(L):
                aux = sum_features_t(t,i,j,x) + log_beta_matrix[i,t+1]
                log_beta_matrix[j,t] = np.logaddexp(log_beta_matrix[j,t],aux)
    # Termination:
    for j in range(L):
        log_beta_matrix[j,0] =  sum_features_t(0,j,None,x) + log_beta_matrix[j,1]
    if verbose:
        print("\nBACKWARD")
        print("Beta matrix (log values):")
        print(log_beta_matrix)
        print("Beta matrix (real values):")
        print(np.exp(log_beta_matrix))
        seq_prob = np.NINF
        for j in range(L):
            seq_prob = np.logaddexp(seq_prob,log_beta_matrix[j,0])
        print("Sequence probability:")
        print("log: ", seq_prob, "\treal: ", np.exp(seq_prob))
    return log_beta_matrix

## POSTERIOR DECODING
# Receives alpha and beta matrix
# Returns the posterior decoding path of states
def posterior_decoding(alpha_matrix,beta_matrix):
    L = len(alpha_matrix[:,0])
    T = len(alpha_matrix[0,:])
    seq_prob = np.NINF
    for j in range(L):
        seq_prob = np.logaddexp(seq_prob,alpha_matrix[j,T-1])
    posterior_matrix = np.full((L,T), np.NINF)
    for t in range(T-1):
        for j in range(L):
            posterior_matrix[j,t] = alpha_matrix[j,t] + beta_matrix[j,t+1] - seq_prob
    for j in range(L):
        posterior_matrix[j,T-1] = alpha_matrix[j,T-1] - seq_prob
    path = np.argmax(posterior_matrix, axis=0)
    if verbose:
        print("\nPOSTERIOR DECODING")
        print("Posterior decoding matrix (log values):")
        print(posterior_matrix)
        print("Posterior decoding matrix (real values):")
        print(np.exp(posterior_matrix))
        print("Path of states:")
        print(path)
    return path

## VITERBI DECODING
# Receives a sequence x
# Returns the Viterbi decoding path of states
def viterbi_decoding(x):
    L = n_states    
    T = x.size
    viterbi_matrix = np.full((L,T), np.NINF)
    paths = np.full((L,T), -1)
    # Inicialization:
    for j in range(L):
        viterbi_matrix[j,0] = sum_features_t(0,j,None,x)
    # Induction:
    for t in range(1,T):
        for j in range(L):
            log_position_probs = np.full((L), np.NINF)
            for i in range(L):
                log_position_probs[i] = sum_features_t(t,j,i,x) + viterbi_matrix[i,t-1]
            viterbi_matrix[j,t] = np.amax(log_position_probs)
            paths[j,t-1] = np.argmax(log_position_probs)
    # Termination:
    for j in range(L):
        paths[j,T-1] = j
    best_path = paths[np.argmax(viterbi_matrix[:,T-1])]
    if verbose:
        print("\nVITERBI DECODING")
        print("Viterbi matrix (log values):")
        print(viterbi_matrix)
        print("Viterbi matrix (real values):")
        print(np.exp(viterbi_matrix))
        print("Paths of states: ")
        print (paths)
        print("Best path: ")
        print(best_path)
        prob_best_path = np.amax(viterbi_matrix[:,T-1])
        print("Best path probability:")
        print("log: ", prob_best_path, "\treal: ", np.exp(prob_best_path))
    return best_path
