from em_workflow import em_workflow
from em_algorithm import train_gmm
from em_algorithm import *
import statistics
from visualization import visualize
import constants as const
##########
# Parameters for selected data set
##########
data_dir = const.DATA_DIR
file_name_train = const.FILE_NAME_TRAIN
file_name_test = const.FILE_NAME_TEST
minority_label = const.MINORITY_LABEL
data_label = const.DATA_LABEL
down_sample_minority = const.DOWN_SAMPLE_MINORITY
minority_div = const.MINORITY_DIV
##########
#parameters related to the choice of method and number repeats
##########
num_repeats = const.NUM_REPEATS
###########
#parameters related to file names
plot_name = const.PLOT_NAME
###########

# step 1: create an instance of em_workflow class
workflow1 = em_workflow(data_dir=data_dir, file_name_train=file_name_train, file_name_test=file_name_test,
                        minority_label=minority_label, data_label=data_label, down_sample_minority=down_sample_minority,
                        minority_div=minority_div)
#
# train_x_expanded, train_y_binary = workflow1.pre_process()
train_p, train_n, eigen_signal, pos_low_d_transposed, neg_low_d_transposed = workflow1.raw_data_to_eigen_signal_space()

f1_score_list=[]
for i in range(1):
	n_clusters = 2
	n_epochs = 2
	clusters, clustering_results, likelihoods, scores, sample_likelihoods, history, total_new_samples_c0, \
	total_new_samples_c1 = train_gmm(pos_low_d_transposed, neg_low_d_transposed, n_clusters, n_epochs, 0.01, 98,
	                                 eigen_signal)

	#
	f1_score = workflow1.workflow_70_inos(num_ADASYN=42, train_p=train_p, train_n=train_n, total_new_samples_c0=total_new_samples_c0, total_new_samples_c1=total_new_samples_c1)
	f1_score_list.append(f1_score)
print(f1_score_list)



