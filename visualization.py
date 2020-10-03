# visualize k-means results
import numpy as np
import matplotlib.pyplot as plt

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
	# plt.plot(em_c0_dim0,em_c0_dim1,'g*')
	plt.plot(em_c1_dim0,em_c1_dim1,'r*')

	plt.savefig('/Users/jiedali/Documents/research/notes/plots/'+ file_name)