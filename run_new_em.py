from em_workflow import em_workflow
from new_em_algorithm import train_new_gmm
from new_em_algorithm import *
import statistics
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
##
# step 1: create an instance of em_workflow class
workflow1 = em_workflow(data_dir=data_dir, file_name_train=file_name_train, file_name_test=file_name_test,
                        minority_label=minority_label, data_label=data_label, down_sample_minority=down_sample_minority,
                        minority_div=minority_div)
# train_x_expanded, train_y_binary = workflow1.pre_process()
train_p, train_n, eigen_signal, pos_low_d_transposed, neg_low_d_transposed = workflow1.raw_data_to_eigen_signal_space()
#===================
model_name = 'rf'
n_clusters = 2
n_epochs = 10
adasyn_percentage=0.3
#
num_new_samples_to_gen = train_n.shape[1] - train_p.shape[1]
num_em_samples = round(num_new_samples_to_gen * (1 - adasyn_percentage))
num_adasyn_samples = round(num_new_samples_to_gen * adasyn_percentage)
#
clusters, likelihoods, sample_likelihoods, history, new_samples_all_clusters,new_samples_c0_epoch0, new_samples_c1_epoch0, new_samples_c0_last_epoch, new_samples_c1_last_epoch= \
	train_new_gmm(pos_low_d_transposed, neg_low_d_transposed, n_clusters, n_epochs, 0.01, num_em_samples, eigen_signal)

# classification at last epoch
f1_score, precision, recall = workflow1.workflow_70_inos(num_ADASYN=num_adasyn_samples, train_p=train_p, train_n=train_n,
		                    new_samples_all_clusters=new_samples_all_clusters, remove_tomeklinks=False, model_name=model_name)

# do classification with samples at epoch 0

