import numpy as np
import matplotlib.pyplot as plt
import constants as const
from em_workflow import em_workflow
from em_algorithm import train_gmm


def visualize(ts_neg_low_d, ts_pos_low_d, eigen_signal_overall, total_new_samples_c0, total_new_samples_c1,file_name):
	"""

	:param ts_neg_low_d: original majority class in eigen signal space, n_samples * n_features
	:param ts_pos_low_d: original minority class in eigen signal space, n_samples * n_features
	:param total_new_samples_c0: EM new samples, n_samples * n_features
	:param total_new_samples_c1: EM new samples, n_samples * n_features
	:return:
	"""
	# majority class
	x_neg=ts_neg_low_d[:,0]
	y_neg=ts_neg_low_d[:,1]
	# minority class
	x_pos=ts_pos_low_d[:,0]
	y_pos=ts_pos_low_d[:,1]
	#
	plt.plot(x_neg,y_neg,'k.')
	plt.plot(x_pos,y_pos,'b+')
	# plot the EM synthesized samples
	# first convert the new samples in eigen signal space
	total_new_samples_c0_eigen = np.real(np.dot(total_new_samples_c0, eigen_signal_overall))
	total_new_samples_c1_eigen = np.real(np.dot(total_new_samples_c1, eigen_signal_overall))
	#
	em_c0_dim0 = total_new_samples_c0_eigen[:,0]
	em_c0_dim1 = total_new_samples_c0_eigen[:,1]
	#
	em_c1_dim0 = total_new_samples_c1_eigen[:,0]
	em_c1_dim1 = total_new_samples_c1_eigen[:,1]
	#
	plt.plot(em_c0_dim0,em_c0_dim1,'g*')
	plt.plot(em_c1_dim0,em_c1_dim1,'r*')

	plt.savefig('/Users/jiedali/Documents/research/notes/plots/'+ file_name)

def plot_ground_truth(ts_neg_low_d, ts_pos_low_d, file_name):
	# majority class
	x_neg=ts_neg_low_d[:,0]
	y_neg=ts_neg_low_d[:,1]
	# minority class
	x_pos=ts_pos_low_d[:,0]
	y_pos=ts_pos_low_d[:,1]
	#
	plt.plot(x_neg,y_neg,'k.')
	plt.plot(x_pos,y_pos,'b+')

	plt.savefig('/Users/jiedali/Documents/research/notes/plots/'+ file_name)

def plot_adasyn(ts_neg_low_d, ts_pos_low_d, X_adasyn, eigen_signal, file_name):
	#
	x_neg=ts_neg_low_d[:,0]
	y_neg=ts_neg_low_d[:,1]
	# minority class
	x_pos=ts_pos_low_d[:,0]
	y_pos=ts_pos_low_d[:,1]
	#
	plt.plot(x_neg,y_neg,'k.')
	plt.plot(x_pos,y_pos,'b+')
	#
	adasyn_low_d = np.real(np.dot(X_adasyn, eigen_signal))
	adasyn_dim0 = adasyn_low_d[:,0]
	adasyn_dim1 = adasyn_low_d[:,1]
	#
	plt.plot(adasyn_dim0,adasyn_dim1,'y*')

	plt.savefig('/Users/jiedali/Documents/research/notes/plots/' + file_name)





if __name__ == "__main__":

	# visualize
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
	# parameters related to the choice of method and number repeats
	##########
	num_repeats = const.NUM_REPEATS
	###########
	# parameters related to file names
	plot_name = const.PLOT_NAME
	###########

	# step 1: create an instance of em_workflow class
	workflow1 = em_workflow(data_dir=data_dir, file_name_train=file_name_train, file_name_test=file_name_test,
	                        minority_label=minority_label, data_label=data_label,
	                        down_sample_minority=down_sample_minority,
	                        minority_div=minority_div)
	#
	# train_x_expanded, train_y_binary = workflow1.pre_process()
	train_p, train_n, eigen_signal, pos_low_d_transposed, neg_low_d_transposed = workflow1.raw_data_to_eigen_signal_space()

	X_adasyn = workflow1.create_adasyn_samples(num_ADASYN=141,train_p=train_p,train_n=train_n)
	# plot adasyn
	plot_adasyn(neg_low_d_transposed, pos_low_d_transposed, X_adasyn, eigen_signal, 'adasyn_samples_FingerMovements_remove_adasyn.png')
	# #
	# #
	# n_clusters=2
	# n_epochs=2
	# clusters, clustering_results, likelihoods, scores, sample_likelihoods, history, total_new_samples_c0, \
	# total_new_samples_c1 = train_gmm(pos_low_d_transposed, neg_low_d_transposed, n_clusters, n_epochs, 0.01, 141,
	#                                  eigen_signal)
	# # plot_ground_truth(neg_low_d_transposed,pos_low_d_transposed,'1to10_imblance_FingerMovements.png')
	# visualize(neg_low_d_transposed,pos_low_d_transposed,eigen_signal,total_new_samples_c0,total_new_samples_c1,'em_samples_FingerMovements.png')
	# step 2: call the top level method (which generates new samples and run classification)



