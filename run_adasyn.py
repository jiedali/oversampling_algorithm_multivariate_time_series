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
train_x_expanded, train_y_binary = workflow1.pre_process()
# input shape for train_p is n_features * n_samples
train_p = train_x_expanded[train_y_binary == 1].transpose()
train_n = train_x_expanded[train_y_binary == 0].transpose()


# train_p, train_n, eigen_signal, pos_low_d_transposed, neg_low_d_transposed = workflow1.raw_data_to_eigen_signal_space()


f1_socre_list=[]
for i in range(10):

	f1_score = workflow1.workflow_100_adasyn(num_ADASYN=140, train_p=train_p, train_n=train_n)
	f1_socre_list.append(f1_score)

print(f1_socre_list)
print("mean f1_score: %d" % statistics.mean(f1_socre_list))
print("max f1_score: %d" % max(f1_socre_list))