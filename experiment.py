from crf import *

if model == 1:
    sequence = np.array([0,1,1])
    # sequence = np.array([0,0,0,0,1,0,1,1,1,0])
elif model == 3:
    sequence = np.array([0,1,2,3,0,1,2,3])

alpha_matrix = forward(sequence,n_states)
beta_matrix = backward(sequence,n_states)
posterior_decoding_path = posterior_decoding(alpha_matrix,beta_matrix)
viterbi_decoding_path = viterbi_decoding(sequence,n_states)
