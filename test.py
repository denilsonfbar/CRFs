from crf import *

if HMM_model == 1:  # Flips of a fair coin and a biased coin alternately
    sequence = np.array([0,1,1])
#   sequence = np.array([0,0,0,0,1,0,1,1,1,0])

elif HMM_model == 2:  # Dishonest casino
    sequence = np.array([6,4,1,2])
#    sequence = np.array([6,6,4,1,2])
#    sequence = np.array([6,4,1,2,6])
#    sequence = np.array([1,2,3,4,6,6,6,1,2,3])

elif HMM_model == 3:  # DNA simulation (3 states)
    sequence = np.array([0,1,2,3,0,1,2,3])

alpha_matrix = forward(sequence,n_states)
beta_matrix = backward(sequence,n_states)
posterior_decoding_path = posterior_decoding(alpha_matrix,beta_matrix)
viterbi_decoding_path = viterbi_decoding(sequence,n_states)