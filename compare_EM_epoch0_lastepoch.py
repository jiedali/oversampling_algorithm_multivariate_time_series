import csv
from em_workflow import em_workflow
from em_algorithm import train_gmm
from em_algorithm import *
import statistics
from visualization import visualize
import constants as const
from helper import get_mean_max_f1score
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


# step 1: create an instance of em_workflow class
workflow1 = em_workflow(data_dir=data_dir, file_name_train=file_name_train, file_name_test=file_name_test,
                        minority_label=minority_label, data_label=data_label, down_sample_minority=down_sample_minority,
                        minority_div=minority_div)
#
# train_x_expanded, train_y_binary = workflow1.pre_process()
train_p, train_n, eigen_signal, pos_low_d_transposed, neg_low_d_transposed = workflow1.raw_data_to_eigen_signal_space()
#
model_name = 'rf'
n_clusters = 2
n_epochs = 2
f1_mean_list=[]
f1_max_list=[]
# adasyn_percentage = 0.3
for adasyn_percentage in [0.3]:
	num_new_samples_to_gen = train_n.shape[1] - train_p.shape[1]
	#
	num_em_samples = round(num_new_samples_to_gen*(1-adasyn_percentage))
	num_adasyn_samples = round(num_new_samples_to_gen*adasyn_percentage)
	print("Number of samples to gen for em and adasyn %d, %d" %(num_em_samples, num_adasyn_samples))
	f1_score_list=[]
	precision_list=[]
	recall_list=[]

	# get data for epoch 0

	new_sample_e0_list=[]
	# reader=csv.reader(open("./new_sample_c0_epoch0.csv","rb"),delimiter=",")
	new_samples_c0_epoch0 = np.genfromtxt("./new_sample_c0_epoch0.csv", delimiter=',', dtype=float)
	print("show loaded")
	print(new_samples_c0_epoch0)
	# x=list(reader)
	# new_samples_c0_epoch0 = np.array(reader).astype("float")
	new_sample_e0_list.append(new_samples_c0_epoch0)

	new_samples_c1_epoch0 = np.genfromtxt("./new_sample_c1_epoch0.csv",delimiter=',', dtype=float)
	# x=list(reader)
	# new_samples_c0_epoch0 = np.array(reader).astype("float")
	new_sample_e0_list.append(new_samples_c1_epoch0)
	# reader=csv.reader(open("./new_sample_c1_epoch0.csv","rb"),delimiter=",")
	# # x=list(reader)
	# new_samples_c1_epoch0 = np.array(reader).astype("float")
	new_samples_all_clusters = new_sample_e0_list



	for i in range(1):
		# clusters, clustering_results, likelihoods, scores, sample_likelihoods, history, new_samples_all_clusters = \
		# 	train_gmm(pos_low_d_transposed, neg_low_d_transposed, n_clusters, n_epochs, 0.01, num_em_samples, eigen_signal)
		f1_score, precision, recall = workflow1.workflow_70_inos(num_ADASYN=num_adasyn_samples, train_p=train_p, train_n=train_n,\
		                                                         new_samples_all_clusters=new_samples_all_clusters, remove_tomeklinks=False, model_name=model_name)
		f1_score_list.append(f1_score)
		precision_list.append(precision)
		recall_list.append(recall)
	print(f1_score_list)
	print(precision_list)
	print(recall_list)
	print("Mean and Max for F1score, precision and recall:")
	f1_stats=get_mean_max_f1score(f1_score_list)
	precision_stats=get_mean_max_f1score(precision_list)
	recall_stats=get_mean_max_f1score(recall_list)
	print(f1_stats)
	print(precision_stats)
	print(recall_stats)

	f1_mean_list.append(f1_stats)
