from em_workflow import em_workflow
from em_algorithm import train_gmm
from em_algorithm import *
import statistics
from visualization import visualize

##########
# Parameters for selected data set
##########
data_dir = '/Users/jiedali/Documents/research/dataset/Multivariate_ts/RacketSports/'
file_name_train = 'RacketSports_TRAIN.ts'
file_name_test = 'RacketSports_TEST.ts'
minority_label = 'badminton_clear'
data_label = 'RacketSports'
down_sample_minority = True
minority_div = 4
##########
#parameters related to the choice of method and number repeats
##########
num_repeats = 1
###########
#parameters related to file names
plot_name = ''
###########

# step 1: create an instance of em_workflow class
workflow1 = em_workflow(data_dir=data_dir, file_name_train=file_name_train, file_name_test=file_name_test,
                        minority_label=minority_label, data_label=data_label, down_sample_minority=down_sample_minority,
                        minority_div=minority_div)
#
# train_x_expanded, train_y_binary = workflow1.pre_process()
train_p, train_n, eigen_signal, pos_low_d_transposed, neg_low_d_transposed = workflow1.raw_data_to_eigen_signal_space()

n_clusters = 2
n_epochs = 2
clusters, clustering_results, likelihoods, scores, sample_likelihoods, history, total_new_samples_c0, \
total_new_samples_c1 = train_gmm(pos_low_d_transposed, neg_low_d_transposed, n_clusters, n_epochs, 0.01, 68,
                                 eigen_signal)

#
f1_score = workflow1.workflow_70_inos(num_ADASYN=29, train_p=train_p, train_n=train_n, total_new_samples_c0=total_new_samples_c0, total_new_samples_c1=total_new_samples_c1)
print(f1_score)
# ## Step 1: Run algorithms with new sample generations separated
# n_clusters = 2
# n_epochs = 2
# ##
# f1score_list=[]
# for i in range(num_repeats):
# 	clusters, clustering_results, likelihoods, scores, sample_likelihoods, history, total_new_samples_c0, \
# 	total_new_samples_c1 = train_gmm(pos_low_d_transposed, neg_low_d_transposed, n_clusters, n_epochs, 0.01, 68, eigen_signal)
# 	#
# 	f1_score = workflow1.workflow_70_inos(num_ADASYN=29, train_p=train_p, train_n=train_n, total_new_samples_c0=total_new_samples_c0, total_new_samples_c1=total_new_samples_c1)
# 	f1score_list.append(f1_score)
# 	#
# print(f1score_list)
# # 100% ADASYN samples
# f1_socre_list=[]
# for i in range(num_repeats):
#
# 	f1_score = workflow1.workflow_100_adasyn(num_ADASYN=65, train_p=train_p, train_n=train_n)
# 	f1_socre_list.append(f1_score)
#
# print(f1_socre_list)
# print("mean f1_score: %d" % statistics.mean(f1_socre_list))


# visualize
# visualize(neg_low_d_transposed,pos_low_d_transposed,eigen_signal,total_new_samples_c0,total_new_samples_c1,plot_name)
# step 2: call the top level method (which generates new samples and run classification)
