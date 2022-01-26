from features import *

# Return the weighted log sum of features functions results of a time t of a sequence x
def log_sum_features(t,y_t,y_t_minus_1,x):
    sum = 0.0
    for i in range(len(ff_list)):
        factor = ff_list[i](t,y_t,y_t_minus_1,x)
        sum += factor
    weighted_factor = W[i] * sum
    log_weighted_factor = np.log2(weighted_factor)
    return log_weighted_factor

## FORWARD
# Receives a sequence x and the number of states L
# Returns the alpha matrix
def forward(x,L):  
    N = x.size  # length of sequence
    alpha_matrix = np.full((L,N), np.NINF)
    # Inicialization:
    for j in range(L):
        alpha_matrix[j,0] = log_sum_features(0,j,None,x)
    # Induction:
    for t in range(1,N):
        for j in range(L):
            for i in range(L):
                aux = alpha_matrix[i,t-1] + log_sum_features(t,j,i,x)
                alpha_matrix[j,t] = np.logaddexp2(alpha_matrix[j,t],aux)
    if verbose:
        print("\nFORWARD")
        print("Alpha matrix (log values):")
        print(alpha_matrix)
        print("Alpha matrix (real values):")
        print(np.exp2(alpha_matrix))
        # Termination:
        seq_prob = np.NINF
        for j in range(L):
            seq_prob = np.logaddexp2(seq_prob,alpha_matrix[j,N-1])
        print("Sequence probability:")
        print("log: ", seq_prob, "\treal: ", np.exp2(seq_prob))
    return alpha_matrix

## BACKWARD
# Receives a sequence x and the number of states L
# Returns the beta matrix
def backward(x,L):  
    N = x.size  # length of sequence
    beta_matrix = np.full((L,N), np.NINF)
    # Inicialization:
    for j in range(L):
        for i in range(L):
            aux = log_sum_features(N-1,i,j,x)
            beta_matrix[j,N-1] = np.logaddexp2(beta_matrix[j,N-1],aux)        
    # Induction:
    for t in reversed(range(1,N-1)):
        for j in range(L):
            for i in range(L):
                aux = beta_matrix[i,t+1] + log_sum_features(t,i,j,x)
                beta_matrix[j,t] = np.logaddexp2(beta_matrix[j,t],aux)
    # Termination:
    for j in range(L):
        beta_matrix[j,0] = beta_matrix[j,1] + log_sum_features(0,j,None,x)
    if verbose:
        print("\nBACKWARD")
        print("Beta matrix (log values):")
        print(beta_matrix)
        print("Beta matrix (real values):")
        print(np.exp2(beta_matrix))
        seq_prob = np.NINF
        for j in range(L):
            seq_prob = np.logaddexp2(seq_prob,beta_matrix[j,0])
        print("Sequence probability:")
        print("log: ", seq_prob, "\treal: ", np.exp2(seq_prob))
    return beta_matrix

## POSTERIOR DECODING
# Receives alpha and beta matrix
# Returns the posterior decoding path of states
def posterior_decoding(alpha_matrix,beta_matrix):
    N = len(alpha_matrix[0,:])
    L = len(alpha_matrix[:,0])
    seq_prob = np.NINF
    for j in range(L):
        seq_prob = np.logaddexp2(seq_prob,alpha_matrix[j,N-1])
    posterior_matrix = np.full((L,N), np.NINF)
    for t in range(N-1):
        for j in range(L):
            posterior_matrix[j,t] = alpha_matrix[j,t] + beta_matrix[j,t+1] - seq_prob
    for j in range(L):
        posterior_matrix[j,N-1] = alpha_matrix[j,N-1] - seq_prob
    path = np.argmax(posterior_matrix, axis=0)
    if verbose:
        print("\nPOSTERIOR DECODING")
        print("Posterior decoding matrix (log values):")
        print(posterior_matrix)
        print("Posterior decoding matrix (real values):")
        print(np.exp2(posterior_matrix))
        print("Path of states:")
        print(path)
    return path

## VITERBI DECODING
# Receives a sequence x and the number of states L
# Returns the Viterbi decoding path of states
def viterbi_decoding(x,L):
    N = x.size  # length of sequence
    viterbi_matrix = np.full((L,N), np.NINF)
    paths = np.full((L,N), -1)
    # Inicialization:
    for j in range(L):
        viterbi_matrix[j,0] = log_sum_features(0,j,None,x)
    # Induction:
    for t in range(1,N):
        for j in range(L):
            log_position_probs = np.full((L), np.NINF)
            for i in range(L):
                log_position_probs[i] = log_sum_features(t,j,i,x) + viterbi_matrix[i,t-1]
            viterbi_matrix[j,t] = np.amax(log_position_probs)
            paths[j,t-1] = np.argmax(log_position_probs)
    # Termination:
    for j in range(L):
        paths[j,N-1] = j
    best_path = paths[np.argmax(viterbi_matrix[:,N-1])]
    if verbose:
        print("\nVITERBI DECODING")
        print("Viterbi matrix (log values):")
        print(viterbi_matrix)
        print("Viterbi matrix (real values):")
        print(np.exp2(viterbi_matrix))
        print("Paths of states: ")
        print (paths)
        print("Best path: ")
        print(best_path)
        prob_best_path = np.amax(viterbi_matrix[:,N-1])
        print("Best path probability:")
        print("log: ", prob_best_path, "\treal: ", np.exp2(prob_best_path))
    return best_path
