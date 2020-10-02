import os
import logging
import numpy as np
import sktime
import pandas as pd
from sktime.utils.load_data import load_from_tsfile_to_dataframe
from sktime.transformers.series_as_features.compose import ColumnConcatenator
from sktime.classification.compose import TimeSeriesForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from sklearn.mixture import GaussianMixture

# import imageio
import matplotlib.animation as ani
import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Ellipse
# from PIL import Image
from sklearn import datasets
from sklearn.cluster import KMeans

# import my own library to run em_sampling
import em_algorithm as em
# from em_algorithm import train_gmm_with_covmat_pertubation_and_sample_generation_gmm_init


class em_workflow(object):

	def __init__(self, data_dir, file_name_train, file_name_test, minority_label, data_label, down_sample_minority=False, minority_div=1):

		self.data_dir = data_dir
		self.file_name_train = file_name_train
		self.file_name_test = file_name_test
		self.minority_label = minority_label
		self.data_label = data_label
		self.down_sample_minority = down_sample_minority
		self.minority_div = minority_div
		# pre_process() returns the training data in original feature space: n_samples * _features


	def pre_process(self):

		train_x, train_y = load_from_tsfile_to_dataframe(os.path.join(self.data_dir, self.file_name_train))
		# train_x is  samples * k features
		# expand train_x to be samples * (kn), where n is the number of temporal dimension
		concatenator = ColumnConcatenator()
		conc_fit = concatenator.fit(train_x)
		train_x_trans = conc_fit.transform(train_x)
		#
		for i in range(0, train_x_trans.shape[0]):
			row = train_x_trans.iloc[i, 0]
			row_df = row.to_frame()
			if i == 0:
				temp = row_df
			else:
				temp = pd.concat([temp, row_df], axis=1)
		# temp_t is samples * nk df
		train_x_expanded = temp.transpose()
		# convert y to binary label
		train_y_binary = self.convert_y_to_binary_label(train_y)

		# usually the original data has a balanced ratio, in that case, we will downsample the minority
		if self.down_sample_minority == True:
			train_x_to_be_downsample = train_x_expanded[train_y_binary == 1]
			# calculate the number of samples to keep for minority (we keep 1/3)
			sample_size = round(train_x_to_be_downsample.shape[0] / self.minority_div)
			# downsample minority
			train_x_downsampled = train_x_to_be_downsample.iloc[0:sample_size]
			# concat x_minoriy and x_majority
			train_x_maj = train_x_expanded[train_y_binary == 0]
			train_x_all = pd.concat([train_x_downsampled, train_x_maj], axis=0)
			# get y_labels after downsample
			train_y_binary_minority = np.ones(sample_size)
			train_y_binary_majority = np.zeros(train_y_binary[train_y_binary == 0].shape[0])
			# concat y_ones and y_zeros
			train_y_binary = np.concatenate((train_y_binary_minority, train_y_binary_majority), axis=0)
			# set train_x_all = train_x_expanded
			train_x_expanded = train_x_all

		return train_x_expanded, train_y_binary


	def convert_y_to_binary_label(self, train_y):

		train_y_binary = np.where(train_y == self.minority_label, 1, 0)
		return train_y_binary


	def raw_data_to_eigen_signal_space(self):
		"""
		This function will first process the raw data and convert it to algorithm-usable eigen-signal space data
		In other words, it converts raw data into lower dimensional eigen spectrum space
		:return:
		self.train_p: minority data chunk from the original dataset, shape: n_features * n_samples
		self.train_n: majority data chunk from the original dataset, shape: n_features * n_samples
		self.eigen_signal: the matrix to transform from original feature space to eigen signal space,
		                    shape: original_feature_dimensions * eigen_signal_dimensions
		self.pos_low_d_transposed: minority data chunk in eigen signal space: n_samples * n_eigen_features
		"""
		train_x_expanded, train_y_binary = self.pre_process()

		train_x = train_x_expanded
		train_y_binary = train_y_binary
		#
		train_p = train_x[train_y_binary == 1]
		train_n = train_x[train_y_binary == 0]
		#
		train_p = train_p.transpose()
		train_n = train_n.transpose()
		#
		ts_neg = train_n
		ts_pos = train_p
		# first compute x_bar
		sum_ts_n = np.sum(ts_neg, axis=1)
		sum_ts_p = np.sum(ts_pos, axis=1)
		ts_bar = (1 / (train_p.shape[1] + train_n.shape[1])) * (sum_ts_n + sum_ts_p)
		#
		# compute centered matrix of obsrvations ts_neg, ts_pos
		ts_neg_centered_t = ts_neg.transpose() - ts_bar.transpose()
		ts_pos_centered_t = ts_pos.transpose() - ts_bar.transpose()
		#
		ts_neg_centered = ts_neg_centered_t.transpose()
		ts_pos_centered = ts_pos_centered_t.transpose()
		#
		P = ts_pos_centered.shape[1]
		N = ts_neg_centered.shape[1]
		#
		ts_pos_centered = ts_pos_centered.to_numpy()
		ts_neg_centered = ts_neg_centered.to_numpy()
		#
		feature_len = train_p.shape[0]
		sum_p = np.zeros(shape=(feature_len, feature_len))
		sum_n = np.zeros(shape=(feature_len, feature_len))
		for i in range(0, P):
			sum_p += np.dot(ts_pos_centered[:, i].reshape((feature_len, 1)),
			                ts_pos_centered[:, i].transpose().reshape((1, feature_len)))
		for i in range(0, N):
			sum_n += np.dot(ts_neg_centered[:, i].reshape((feature_len, 1)),
			                ts_neg_centered[:, i].transpose().reshape((1, feature_len)))
		##
		cov_mat = (sum_p + sum_n) / (P + N)
		# eigen decomposition of covariance matrix
		w, v = np.linalg.eig(cov_mat)
		null_index = np.where(w < 5e-5)
		null_index = null_index[0][0]
		#
		# separate the eigen vectors into signal space and null space
		# so signal space would be eigenvecotrs 0~19, null space would be eigenvectors 20~63
		eigen_signal = v[:, 0:null_index]
		# transform all samples into the lower dimensional space, from 64 to 19
		ts_neg_low_d = np.dot(eigen_signal.transpose(), ts_neg)
		ts_pos_low_d = np.dot(eigen_signal.transpose(), ts_pos)
		######
		# covert the minority data trunk to be n_samples * n_features, also convert complex number to real
		pos_low_d_transposed = np.real(ts_pos_low_d.transpose())
		neg_low_d_transposed = np.real(ts_neg_low_d.transpose())

		return train_p, train_n, eigen_signal, pos_low_d_transposed, neg_low_d_transposed


	###########
	#convert samples from eigen signal space to original feature space
	###########
	# Convert new samples back to original feature space, then do a classification
	# convert back to original feature space
	def back_original_feature_space(self, new_samples):

		# the new_samples generated is in the reduced eigen signal space, convert it original feature space
		# eigen_signal matrix is n_original_features * n_reduced_features
		new_samples_original_feature = np.dot(self.eigen_signal, new_samples.transpose())

		return new_samples_original_feature

	############
	# classification workflows for 3 different methods: EM, 100% adasyn, 100% SMOTE
	############
	# def workflow_70_inos(self, data_dir, file_name_train, minority_label, total_new_samples_c0, total_new_samples_c1,
	#                      X_adasyn):
	def workflow_70_inos(self, num_ADASYN):

		inos_p_old = self.train_x_expanded[self.train_y_binary == 1]
		inos_n = self.train_x_expanded[self.train_y_binary == 0]
		# generate 30% ADASYN samples
		# prepare data to run ADASYN: ADASYN trains on entire original training data
		X = pd.concat((self.train_p.transpose(),self.train_n.transpose()), axis=0)
		# create y
		y_p = np.ones(self.train_p.shape[1])
		y_n = np.zeros(self.train_n.shape[1])
		y = np.concatenate((y_p, y_n))
		# We will generate equal number of minority samples as majority samples
		majority_sample_cnt = self.train_n.shape[1]
		ada = ADASYN(sampling_strategy={1: majority_sample_cnt, 0: majority_sample_cnt})
		# X contains all data, should be in format of n_samples*n_features
		X_res, y_res = ada.fit_resample(X, y)
		starting_index = majority_sample_cnt - num_ADASYN
		X_adasyn = X_res.iloc[starting_index:majority_sample_cnt, :]
		# combine p all clusters
		inos_p = pd.concat([inos_p_old, self.total_new_samples_c0, self.total_new_samples_c1, X_adasyn], axis=0)
		# combine p and n
		x_res = pd.concat([inos_p, inos_n], axis=0)
		# create y_res
		y_res_p = np.ones(inos_p.shape[0])
		y_res_n = np.zeros(inos_n.shape[0])
		y_res = np.concatenate([y_res_p, y_res_n])
		#
		tmo = self.build_model(x_res, y_res)
		# evaluates performance
		x_test, y_test_binary = self.pre_process(self.data_dir, self.file_name_test, self.minority_label, down_sample_minority=False)
		#
		f1_score = self.eval_model(tmo, x_test, y_test_binary)

		return f1_score

	############
	# Integrate all function to fun EM sampling and then classification
	############

	def run_em_sampling_classification(self, n_clusters, n_epochs, epsilon, num_em_samples, num_ADASYN):

		"""
		This is the top-level method to run EM oversampling and get f1_score
		:return:
		f1_score: classification results
		"""
		###
		self.train_p, self.train_n, self.eigen_signal, self.pos_low_d_transposed = self.raw_data_to_eigen_signal_space()
		### Step 1: Run EM algorithm to get EM samples
		self.clusters, self.likelihoods, self.scores, self.sample_likelihoods, self.history, self.total_new_samples_c0, self.total_new_samples_c1 = \
			em.train_gmm_with_covmat_pertubation_and_sample_generation_gmm_init(self.pos_low_d_transposed, n_clusters, n_epochs, epsilon, num_em_samples)
		# convert new samples back to original feature space
		total_new_samples_c0 = self.back_original_feature_space(self.total_new_samples_c0)
		total_new_samples_c1 = self.back_original_feature_space(self.total_new_samples_c1)
		# format the new samples (transpose it and convert it to pandas dataframe)
		self.total_new_samples_c0 = pd.DataFrame(np.real(total_new_samples_c0.transpose()))
		self.total_new_samples_c1 = pd.DataFrame(np.real(total_new_samples_c1.transpose()))
		#
		# STEP 2: get classification results
		self.f1_score= self.workflow_70_inos(num_ADASYN)


		return self.f1_score



