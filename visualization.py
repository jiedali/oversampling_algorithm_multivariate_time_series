import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import constants as const
from em_workflow import em_workflow
from em_algorithm import train_gmm

from cycler import cycler

# Update matplotlib defaults to something nicer
mpl_update = {
    'font.size': 14,
    'axes.prop_cycle': cycler('color', ['#0085ca', '#888b8d', '#00c389', '#f4364c', '#e56db1']),
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
}
mpl.rcParams.update(mpl_update)

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
	# plt.plot(em_c0_dim0,em_c0_dim1,'y*',label='EM samples for cluster0')
	# plt.plot(em_c1_dim0,em_c1_dim1,'p*',label='EM samples for cluster1')
	plt.plot(em_c0_dim0,em_c0_dim1,'y*',label='EM samples')
	plt.plot(em_c1_dim0,em_c1_dim1,'y*',label='EM samples')
	plt.legend()

	plt.savefig('/Users/jiedali/Documents/research/notes/plots/'+ file_name)

def visualize_two_cluster(ts_neg_low_d, ts_pos_low_d, total_new_samples_c0, total_new_samples_c1, file_name):
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
	#
	em_c0_dim0 = total_new_samples_c0[:,0]
	em_c0_dim1 = total_new_samples_c0[:,1]
	#
	em_c1_dim0 = total_new_samples_c1[:,0]
	em_c1_dim1 = total_new_samples_c1[:,1]
	#
	# plt.plot(em_c0_dim0,em_c0_dim1,'y*',label='EM samples for cluster0')
	# plt.plot(em_c1_dim0,em_c1_dim1,'p*',label='EM samples for cluster1')
	plt.plot(em_c0_dim0,em_c0_dim1,'g*',label='C0')
	plt.plot(em_c1_dim0,em_c1_dim1,'y*',label='C1')
	# plt.legend()

	plt.savefig('/Users/jiedali/Documents/research/notes/plots/'+ file_name)

def visualize_one_cluster(ts_neg_low_d, ts_pos_low_d, eigen_signal_overall, total_new_samples_c0,file_name):
	"""

	:param ts_neg_low_d: original majority class in eigen signal space, n_samples * n_features
	:param ts_pos_low_d: original minority class in eigen signal space, n_samples * n_features
	:param total_new_samples_c0: EM new samples, n_samples * n_features
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
	#
	em_c0_dim0 = total_new_samples_c0_eigen[:,0]
	em_c0_dim1 = total_new_samples_c0_eigen[:,1]
	#
	# plt.plot(em_c0_dim0,em_c0_dim1,'y*',label='EM samples for cluster0')
	# plt.plot(em_c1_dim0,em_c1_dim1,'p*',label='EM samples for cluster1')
	plt.plot(em_c0_dim0,em_c0_dim1,'y*',label='INOS samples')
	# plt.plot(em_c1_dim0,em_c1_dim1,'y*',label='EM samples')
	plt.legend()

	plt.savefig('/Users/jiedali/Documents/research/notes/plots/'+ file_name)

def plot_ground_truth(ts_neg_low_d, ts_pos_low_d, file_name):
	mpl.rc('font', family='serif')
	font = {'family': 'serif',
	        'color': 'black',
	        'weight': 'normal',
	        'size': 17,
	        }


	# majority class
	x_neg=ts_neg_low_d[:,0]
	y_neg=ts_neg_low_d[:,1]
	# minority class
	x_pos=ts_pos_low_d[:,0]
	y_pos=ts_pos_low_d[:,1]
	#
	plt.figure()
	plt.plot(x_neg,y_neg,'k.', label='Majority')
	plt.plot(x_pos,y_pos,'b+', label='Minority')
	# plt.ylim([None,1250])
	# plt.xlim([-3000,4000])
	plt.legend()

	# turn off tick marks
	plt.tick_params(
		axis='both',  # changes apply to the x-axis
		which='both',  # both major and minor ticks are affected
		bottom=False,  # ticks along the bottom edge are off
		top=False,  # ticks along the top edge are off
		labelbottom=False,
		left = False,
		labelleft=False)


	# labels along the bottom edge are off

	# plt.text(  # position text relative to Figure
	# 	0.0, 1.0, 'RacketSports',
	# 	ha='left', va='top',
	# )
	plt.title('Uni-Modal Gaussian', fontdict=font)

	plt.gca().xaxis.set_major_locator(plt.NullLocator())
	plt.gca().yaxis.set_major_locator(plt.NullLocator())
	plt.savefig('/Users/jiedali/Documents/research/notes/plots/'+ file_name, bbox_inches='tight',pad_inches=0.1)



def plot_ground_truth_plus_test(ts_neg_low_d, ts_pos_low_d, test_pos, test_neg,file_name):

	# majority class
	x_neg=ts_neg_low_d[:,0]
	y_neg=ts_neg_low_d[:,1]
	# minority class
	x_pos=ts_pos_low_d[:,0]
	y_pos=ts_pos_low_d[:,1]
	#
	plt.plot(x_neg,y_neg,'k.', label='Majority Class')
	plt.plot(x_pos,y_pos,'b+', label='Minority Class')
	#
	test_pos_dim0 = test_pos[:,0]
	test_pos_dim1 = test_pos[:,1]
	#
	test_neg_dim0 = test_neg[:,0]
	test_neg_dim1 = test_neg[:,1]
	#
	plt.plot(test_pos_dim0,test_pos_dim1,'r.',label='Test data - Majority Class')
	plt.legend()

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
	plt.plot(adasyn_dim0,adasyn_dim1,'y*', label ='ADASYN')
	plt.legend()

	plt.savefig('/Users/jiedali/Documents/research/notes/plots/' + file_name)


def plot_smote(ts_neg_low_d, ts_pos_low_d, X_smote, eigen_signal, file_name):
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
	adasyn_low_d = np.real(np.dot(X_smote, eigen_signal))
	adasyn_dim0 = adasyn_low_d[:,0]
	adasyn_dim1 = adasyn_low_d[:,1]
	#
	plt.plot(adasyn_dim0,adasyn_dim1,'y*', label ='SMOTE samples')
	plt.legend()

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
	###########
	# step 1: create an instance of em_workflow class
	workflow1 = em_workflow(data_dir=data_dir, file_name_train=file_name_train, file_name_test=file_name_test,
	                        minority_label=minority_label, data_label=data_label,
	                        down_sample_minority=down_sample_minority,
	                        minority_div=minority_div)
	#
	# train_x_expanded, train_y_binary = workflow1.pre_process()
	train_p, train_n, eigen_signal, pos_low_d_transposed, neg_low_d_transposed = workflow1.raw_data_to_eigen_signal_space()

	#
	# visualize_two_cluster(neg_low_d_transposed, pos_low_d_transposed, new_samples_all_clusters[0], new_samples_all_clusters[1])
	#
	# test_x, test_y = workflow1.pre_process(test_data=True)
	# # covert test data into eigen space
	# test_x_eigen = np.real(np.dot(test_x,eigen_signal))
	# test_x_eigen_pos = test_x_eigen[test_y==1]
	# test_x_eigen_neg = test_x_eigen[test_y==0]
	#
	# #
	# plot_ground_truth_plus_test(neg_low_d_transposed, pos_low_d_transposed, test_x_eigen_pos, test_x_eigen_neg, 'RackeSports_original_plus_test.png')
	plot_ground_truth(neg_low_d_transposed, pos_low_d_transposed, 'RacketSports_try1.png')
	#
	#
	# plot adasyn
	# X_adasyn = workflow1.create_adasyn_samples(num_ADASYN=90,train_p=train_p,train_n=train_n)
	# plot_adasyn(neg_low_d_transposed, pos_low_d_transposed, X_adasyn, eigen_signal, 'adasyn_samples_racketsports.png')
	#
	# x_smote = workflow1.create_smote_samples(num_SMOTE=97,train_p=train_p,train_n=train_n)
	# plot_smote(neg_low_d_transposed, pos_low_d_transposed, x_smote, eigen_signal, 'smote_samples_racketsports.png')
	# #
	# n_clusters=2
	# n_epochs=2
	# clusters, clustering_results, likelihoods, scores, sample_likelihoods, history, new_samples_all_clusters= \
	# 	train_gmm(pos_low_d_transposed, neg_low_d_transposed, n_clusters, n_epochs, 0.01, 90, eigen_signal)
	# #
	# total_new_samples_c0 = new_samples_all_clusters[0]
	# total_new_samples_c1 = new_samples_all_clusters[1]
	# #
	# visualize(neg_low_d_transposed, pos_low_d_transposed, eigen_signal, total_new_samples_c0, total_new_samples_c1,'racketsports_original_plus_em')

	# # plot_ground_truth(neg_low_d_transposed,pos_low_d_transposed,'1to10_imblance_FingerMovements.png')
	# visualize_one_cluster(neg_low_d_transposed,pos_low_d_transposed,eigen_signal,total_new_samples_c0,'em_samples_FingerMovements_1_cluster.png')
	# step 2: call the top level method (which generates new samples and run classification)




