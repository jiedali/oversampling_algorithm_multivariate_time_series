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
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import TomekLinks
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

	def __init__(self, data_dir, file_name_train, file_name_test, minority_label, data_label,
	             down_sample_minority=False, minority_div=1):

		self.data_dir = data_dir
		self.file_name_train = file_name_train
		self.file_name_test = file_name_test
		self.minority_label = minority_label
		self.data_label = data_label
		self.down_sample_minority = down_sample_minority
		self.minority_div = minority_div

	# pre_process() returns the training data in original feature space: n_samples * _features

	def pre_process(self, test_data=False):
		#
		if test_data == True:
			train_x, train_y = load_from_tsfile_to_dataframe(os.path.join(self.data_dir, self.file_name_test))
		elif test_data == False:
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
		if test_data == False and self.down_sample_minority == True:
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



	def raw_data_to_eigen_signal_space(self,test_data=False):
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
		train_x_expanded, train_y_binary = self.pre_process(test_data=test_data)

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
		print("debug, shape of v is:")
		print(v.shape)
		print(v.shape[1])
		null_index_array = np.where(w < 5e-5)
		if len(null_index_array[0]>0):
			null_index = null_index_array[0][0]
		elif len(null_index_array[0])==0:
			print("there is no eigen value less than 5e-5")
			# null index should be the same as the index of last column of v
			null_index = v.shape[1]
		# print("debug, null_index is")
		# print(null_index)
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
	# convert samples from eigen signal space to original feature space
	###########
	# Convert new samples back to original feature space, then do a classification
	# convert back to original feature space
	def back_original_feature_space(self, new_samples):

		# the new_samples generated is in the reduced eigen signal space, convert it original feature space
		# eigen_signal matrix is n_original_features * n_reduced_features
		new_samples_original_feature = np.dot(self.eigen_signal, new_samples.transpose())

		return new_samples_original_feature

	## wrap classiciation and evaluation in function
	from sklearn.ensemble import RandomForestClassifier

	def build_model(self, x_train, y_train, model_name):
		"""
		x_train: organized training data, i samples * n features (n is expanded time dimensioon)
		y_train: labesl of training data
		"""
		# run classification use randomforest classifier
		if model_name == 'rf':
			clf = RandomForestClassifier(n_estimators=100, random_state=0).fit(x_train, y_train)

		elif model_name == 'lr':
			clf = LogisticRegression(random_state=0,max_iter=10000).fit(x_train, y_train)

		elif model_name == 'svm':
			clf = SVC(kernel='rbf',gamma='auto').fit(x_train, y_train)

		elif model_name == 'xgb':
			clf = XGBClassifier().fit(x_train,y_train)
		else:
		# default classification is logistic regression
			clf = LogisticRegression(random_state=0).fit(x_train, y_train)

		return clf


	def eval_model(self, tmo, x_test, y_test):
		"""
		tmo: trained model object, returned from the build_model function
		x_test: test data, i samples * n features (n is expanded time dimension)
		y_test: lables of test set
		"""

		pred_y = tmo.predict(x_test)
		# confusion matrix
		cm = confusion_matrix(y_test, pred_y, labels=[1, 0])
		print(cm)
		#
		if cm[0][0] == cm[1][0] == 0:
			precision = 0
		else:
			precision = cm[0][0] / (cm[0][0] + cm[1][0])
		recall = cm[0][0] / (cm[0][0] + cm[0][1])
		#

		f1_score = 2 * (precision * recall) / (precision + recall)
		acc = (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[1][1] + cm[0][1] + cm[1][0])
		#
		print('precision:', precision, 'recall:', recall, 'f1_score:', f1_score, 'accuracy:', acc)

		return f1_score, precision, recall

	############
	# classification workflows for 3 different methods plus no oversampling: EM, 100% adasyn, 100% SMOTE
	############

	def workflow_no_oversampling(self, remove_tomeklinks, model_name):
		"""
		This function performs the workflow of classification without any oversampling
		:return: f1 score without oversampling
		"""
		train_x_expanded, train_y_binary = self.pre_process(test_data=False)
		inos_p_old = train_x_expanded[train_y_binary == 1]
		inos_n = train_x_expanded[train_y_binary == 0]
		print("debug, shape of inos_p_old, inos_n")
		print(inos_p_old.shape, inos_n.shape)
		x_res = pd.concat([inos_p_old, inos_n], axis=0)
		# create y_res
		y_res_p = np.ones(inos_p_old.shape[0])
		y_res_n = np.zeros(inos_n.shape[0])
		y_res = np.concatenate([y_res_p, y_res_n])
		print("debug, shape of training data:")
		print(x_res.shape)
		print(y_res.shape)
		if remove_tomeklinks == True:
			tl = TomekLinks()
			x_res, y_res = tl.fit_resample(x_res, y_res)
			print("shape of training data after removing tomek links:")
			print(x_res.shape)
			print(y_res.shape)
		else:
			pass
		tmo = self.build_model(x_res, y_res, model_name)
		# evaluates performance
		x_test, y_test_binary = self.pre_process(test_data=True)
		#
		f1_score, precision, recall = self.eval_model(tmo, x_test, y_test_binary)

		return f1_score, precision, recall

	def workflow_70_inos(self, num_ADASYN, train_p, train_n, new_samples_all_clusters, remove_tomeklinks, model_name):

		# format the new samples (transpose it and convert it to pandas dataframe) and concat them
		new_samples_pd_list = []
		for cluster_index in range(len(new_samples_all_clusters)):
			new_samples_per_cluster = pd.DataFrame(np.real(new_samples_all_clusters[cluster_index]))

			print("debug, shape of new samples for cluster %d" % cluster_index)
			print(new_samples_per_cluster.shape)
			# add the converted dataframe back to the list; Now the list contains as many dataframe as number of clusters
			new_samples_pd_list.append(new_samples_per_cluster)

		# concat new samples for each cluster
		if len(new_samples_all_clusters) == 1:
			new_samples_concated = new_samples_per_cluster
		else:
			new_samples_concated = pd.concat([i for i in new_samples_pd_list], axis=0)
		#
		print("debug, shape of concatenated new samples for %d clusters:" % len(new_samples_all_clusters))
		print(new_samples_concated.shape)

		# concatenated new samples in shape of n_samples * n_features

		train_x_expanded, train_y_binary = self.pre_process(test_data=False)

		inos_p_old = train_x_expanded[train_y_binary == 1]
		inos_n = train_x_expanded[train_y_binary == 0]
		print("debug, shape of inos_p_old, inos_n")
		print(inos_p_old.shape, inos_n.shape)
		#################################
		# generate 30% ADASYN samples
		#################################
		# prepare data to run ADASYN: ADASYN trains on entire original training data
		X = pd.concat((train_p.transpose(), train_n.transpose()), axis=0)
		# create y
		y_p = np.ones(train_p.shape[1])
		y_n = np.zeros(train_n.shape[1])
		y = np.concatenate((y_p, y_n))
		# We will generate equal number of minority samples as majority samples
		majority_sample_cnt = train_n.shape[1]

		if num_ADASYN != 0:

			ada = ADASYN(sampling_strategy=1.0, n_neighbors=3)
			# X contains all data, should be in format of n_samples*n_features
			X_res, y_res = ada.fit_resample(X, y)
			# In X_res, the first segment is original minority class samples, 2nd segment is original majority class samples
			# last segment is synthesized minority samples, we only want the last segment
			num_adasyn_samples_generated = X_res.shape[0] - train_p.shape[1] - train_n.shape[1]
			starting_index = X_res.shape[0] - num_adasyn_samples_generated
			if num_ADASYN >= num_adasyn_samples_generated:
				X_adasyn = X_res.iloc[starting_index:X_res.shape[0], :]
			elif num_ADASYN < num_adasyn_samples_generated:
				X_adasyn = X_res.iloc[starting_index:(starting_index + num_ADASYN)]
			print("debug, X_adasyn shape")
			print(X_adasyn.shape)
			############################combine all samples, prepare for training
			# combine p all clusters
			inos_p = pd.concat([inos_p_old, new_samples_concated, X_adasyn], axis=0)
		else:
			inos_p = pd.concat([inos_p_old, new_samples_concated], axis=0)
		# combine p and n
		x_res = pd.concat([inos_p, inos_n], axis=0)
		# create y_res
		y_res_p = np.ones(inos_p.shape[0])
		y_res_n = np.zeros(inos_n.shape[0])
		y_res = np.concatenate([y_res_p, y_res_n])
		# print("debug, shape of training data:")
		# print(x_res.shape)
		# print(y_res.shape)
		#
		if remove_tomeklinks == True:
			tl = TomekLinks()
			x_res, y_res = tl.fit_resample(x_res, y_res)
			# print("shape of training data after removing tomek links:")
			# print(x_res.shape)
			# print(y_res.shape)
		else:
			pass

		tmo = self.build_model(x_res, y_res, model_name)
		# evaluates performance
		x_test, y_test_binary = self.pre_process(test_data=True)
		#
		f1_score, precision, recall = self.eval_model(tmo, x_test, y_test_binary)

		return f1_score, precision, recall

	def workflow_100_adasyn(self, num_ADASYN, train_p, train_n, model_name):

		train_x_expanded, train_y_binary = self.pre_process(test_data=False)

		inos_p_old = train_x_expanded[train_y_binary == 1]
		inos_n = train_x_expanded[train_y_binary == 0]
		print("debug, shape of inos_p_old, inos_n")
		print(inos_p_old.shape, inos_n.shape)
		# generate 30% ADASYN samples
		# prepare data to run ADASYN: ADASYN trains on entire original training data
		X = pd.concat((train_p.transpose(), train_n.transpose()), axis=0)
		# create y
		y_p = np.ones(train_p.shape[1])
		y_n = np.zeros(train_n.shape[1])
		y = np.concatenate((y_p, y_n))
		# We will generate equal number of minority samples as majority samples
		#
		ada = ADASYN(sampling_strategy=1.0, n_neighbors=5)
		# X contains all data, should be in format of n_samples*n_features
		X_res, y_res = ada.fit_resample(X, y)
		# In X_res, the first segment is original minority class samples, 2nd segment is original majority class samples
		# last segment is synthesized minority samples, we only want the last segment
		num_adasyn_samples_generated = X_res.shape[0] - train_p.shape[1] - train_n.shape[1]
		starting_index = X_res.shape[0] - num_adasyn_samples_generated
		if num_ADASYN >= num_adasyn_samples_generated:
			X_adasyn = X_res.iloc[starting_index:X_res.shape[0], :]
		elif num_ADASYN < num_adasyn_samples_generated:
			X_adasyn = X_res.iloc[starting_index:(starting_index + num_ADASYN)]
		print("debug, X_adasyn shape")
		print(X_adasyn.shape)
		# combine p all clusters
		inos_p = pd.concat([inos_p_old, X_adasyn], axis=0)
		# combine p and n
		x_res = pd.concat([inos_p, inos_n], axis=0)
		# create y_res
		y_res_p = np.ones(inos_p.shape[0])
		y_res_n = np.zeros(inos_n.shape[0])
		y_res = np.concatenate([y_res_p, y_res_n])
		print("debug, shape of training data:")
		print(x_res.shape)
		print(y_res.shape)
		#
		tmo = self.build_model(x_res, y_res, model_name)
		# evaluates performance
		x_test, y_test_binary = self.pre_process(test_data=True)
		#
		f1_score, precision, recall = self.eval_model(tmo, x_test, y_test_binary)

		return f1_score, precision, recall

	def workflow_100_smote(self, num_SMOTE, train_p, train_n, model_name):

		train_x_expanded, train_y_binary = self.pre_process(test_data=False)
		original_p = train_x_expanded[train_y_binary == 1]
		original_n = train_x_expanded[train_y_binary == 0]

		original_P_N = pd.concat((train_p.transpose(), train_n.transpose()), axis=0)
		# create y
		y_p = np.ones(train_p.shape[1])
		y_n = np.zeros(train_n.shape[1])
		y = np.concatenate((y_p, y_n))
		# input: original_P_N: n_samples * n_features, original data set including P and N
		# input: y: corresponding lables for original data set
		# SMOTE
		sm = SMOTE(sampling_strategy=1,k_neighbors=3)
		# x_res included both original and SMOTE synthesized data
		X_smote, y_smote = sm.fit_resample(original_P_N, y)
		starting_index = train_p.shape[1] + train_n.shape[1]
		#
		X_smote_new = X_smote.iloc[starting_index:X_smote.shape[0], :]
		total_P = pd.concat([original_p, X_smote_new], axis=0)
		print("debug, shape of total_P")
		print(total_P.shape)
		# combine p and n
		total_P_N = pd.concat([total_P, original_n], axis=0)
		y_res_p = np.ones(total_P.shape[0])
		y_res_n = np.zeros(original_n.shape[0])
		y_res = np.concatenate([y_res_p, y_res_n])
		#
		tmo = self.build_model(total_P_N, y_res, model_name)
		# evaluates performance
		x_test, y_test_binary = self.pre_process(test_data=True)
		#
		f1_score, precision, recall = self.eval_model(tmo, x_test, y_test_binary)

		return f1_score, precision, recall

	def create_smote_samples(self, num_SMOTE, train_p, train_n):

		train_x_expanded, train_y_binary = self.pre_process(test_data=False)
		original_p = train_x_expanded[train_y_binary == 1]
		original_n = train_x_expanded[train_y_binary == 0]

		original_P_N = pd.concat((train_p.transpose(), train_n.transpose()), axis=0)
		# create y
		y_p = np.ones(train_p.shape[1])
		y_n = np.zeros(train_n.shape[1])
		y = np.concatenate((y_p, y_n))
		# input: original_P_N: n_samples * n_features, original data set including P and N
		# input: y: corresponding lables for original data set
		# SMOTE
		sm = SMOTE(sampling_strategy=1)
		# x_res included both original and SMOTE synthesized data
		x_smote, y_smote = sm.fit_resample(original_P_N, y)
		starting_index = train_p.shape[1] + train_n.shape[1]
		#
		x_smote_new = x_smote.iloc[starting_index:x_smote.shape[0], :]

		return x_smote_new

	def create_adasyn_samples(self, num_ADASYN, train_p, train_n):

		# train_x_expanded, train_y_binary = self.pre_process(test_data=False)

		# inos_p_old = train_x_expanded[train_y_binary == 1]
		# inos_n = train_x_expanded[train_y_binary == 0]
		# generate 30% ADASYN samples
		# prepare data to run ADASYN: ADASYN trains on entire original training data
		X = pd.concat((train_p.transpose(), train_n.transpose()), axis=0)
		# create y
		y_p = np.ones(train_p.shape[1])
		y_n = np.zeros(train_n.shape[1])
		y = np.concatenate((y_p, y_n))
		# We will generate equal number of minority samples as majority samples
		majority_sample_cnt = train_n.shape[1]
		# ada = ADASYN(sampling_strategy={1: majority_sample_cnt, 0: majority_sample_cnt})
		ada = ADASYN(sampling_strategy=1.0, n_neighbors=10)
		# X contains all data, should be in format of n_samples*n_features
		X_res, y_res = ada.fit_resample(X, y)
		# In X_res, the first segment is original minority class samples, 2nd segment is original majority class samples
		# last segment is synthesized minority samples, we only want the last segment
		num_adasyn_samples_generated = X_res.shape[0] - train_p.shape[1] - train_n.shape[1]
		starting_index = X_res.shape[0] - num_adasyn_samples_generated
		X_adasyn = X_res.iloc[starting_index:X_res.shape[0], :]
		print("debug, X_adasyn shape")
		print(X_adasyn.shape)

		return X_adasyn
