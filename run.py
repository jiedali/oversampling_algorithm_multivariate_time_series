from em_workflow import em_workflow
from em_algorithm import train_gmm
from em_algorithm import *

data_dir = '/Users/jiedali/Documents/research/dataset/Multivariate_ts/RacketSports/'
file_name_train = 'RacketSports_TRAIN.ts'
file_name_test = 'RacketSports_TEST.ts'
minority_label = 'badminton_clear'
data_label = 'RacketSports'

# step 1: create an instance of em_workflow class
workflow1 = em_workflow(data_dir=data_dir, file_name_train=file_name_train, file_name_test=file_name_test,
                        minority_label=minority_label, data_label=data_label, down_sample_minority=False,
                        minority_div=1)
#
# train_x_expanded, train_y_binary = workflow1.pre_process()
train_p, train_n, eigen_signal, pos_low_d_transposed, neg_low_d_transposed = workflow1.raw_data_to_eigen_signal_space()


## Step 1: Run algorithms with new sample generations separated
n_clusters = 2
n_epochs = 2
##
clusters, clustering_results, likelihoods, scores, sample_likelihoods, history, total_new_samples_c0, \
total_new_samples_c1 = train_gmm(pos_low_d_transposed, neg_low_d_transposed, n_clusters, n_epochs, 0.01, 45, eigen_signal)

# # step 2: call the top level method (which generates new samples and run classification)
# f1_score = workflow1.run_em_sampling_classification(n_clusters=2,n_epochs=2,epsilon=0.01,num_new_samples=45, num_ADASYN=20)
