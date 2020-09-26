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
from sklearn.mixture import GaussianMixture

# import imageio
import matplotlib.animation as ani
import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np

#
from scipy.stats import multivariate_normal
#
from matplotlib.patches import Ellipse
# from PIL import Image
from sklearn import datasets
from sklearn.cluster import KMeans

################
# Integrated EM sampling algorithm
# This is the top level function that calls all the sub functions
################

def train_gmm_with_covmat_pertubation_and_sample_generation_gmm_init(X, n_clusters, n_epochs, epsilon, num_new_samples):
	"""
	:param X: minority data chunk in eigen-signal space, n_samples * n_features
	:param n_clusters: number of gaussian modes
	:param n_epochs: number of epochs to run for EM algorithm
	:param epsilon: covariance matrix perturbation, typically set to 0.01
	:param num_new_samples: desired number of new samples to generate
	:return: clusters, likelihoods, scores, sample_likelihoods, history, total_new_samples_c0, total_new_samples_c1
	clusters: list with each element a dictionary containing information for each cluster
	total_new_samples_c0: EM generated new samples for cluster 0: n_samples * n_features
	total_new_samples_c1: EM generated new samples for cluster 1: n_samples * n_features
	"""

	# New samples are generated after maximization step, before expectation step
	clusters = initialize_clusters_with_gmm_results(X, n_clusters)
	likelihoods = np.zeros((n_epochs,))
	scores = np.zeros((X.shape[0], n_clusters))
	history = []

	for i in range(n_epochs):

		print('Pi_k at the beginning of %d iteration' % i)
		print('cluster0', clusters[0]['pi_k'])
		print('cluster1', clusters[1]['pi_k'])

		clusters_snapshot = []

		# This is just for our later use in the graphs
		for cluster in clusters:
			clusters_snapshot.append({
				'mu_k': cluster['mu_k'].copy(),
				'cov_k': cluster['cov_k'].copy()
			})

		history.append(clusters_snapshot)

		# generate new samples
		# first get the target number of samples to be generated for each cluster
		# new_samples_c0, new_samples_c1 = generate_new_samples(clusters)
		if i != 0:
			new_samples_c0, new_samples_c1 = generate_new_samples_regu_eigen(clusters, num_new_samples)

			if i == 1:
				total_new_samples_c0 = new_samples_c0
				total_new_samples_c1 = new_samples_c1
			else:
				total_new_samples_c0 = np.vstack((total_new_samples_c0, new_samples_c0))
				total_new_samples_c1 = np.vstack((total_new_samples_c1, new_samples_c1))

		# finally combine new samples together with old one, to feed into expectation step

		#         X = np.vstack((X,np.real(new_samples_c0),np.real(new_samples_c1)))
		#         if i!=0:
		#             print("Before sending total data chunk X and clusters to expectation_step, their size seems to
		#             mismatch:")
		#             print("Shape of X:")
		#             print(X.shape)
		#             print("Shape of clusters:")
		#             print(clusters[0]['gamma_nk'].shape)
		#             print(clusters[1]['gamma_nk'].shape)
		#
		expectation_step(X, clusters, epsilon)
		maximization_step(X, clusters)

		likelihood, sample_likelihoods = get_likelihood(X, clusters)
		likelihoods[i] = likelihood

		print('Epoch: ', i + 1, 'Likelihood: ', likelihood)

	for i, cluster in enumerate(clusters):
		scores[:, i] = np.log(cluster['gamma_nk']).reshape(-1)

	return clusters, likelihoods, scores, sample_likelihoods, history, total_new_samples_c0, total_new_samples_c1


#################
# initialize clusters
#################
def initialize_clusters_with_gmm_results(X, n_clusters):
	"""

	:param X: p class data, shape is n_samples * n_features
	:param n_clusters: number of Gaussian modes (typically set to 2, needs to be experimented with)
	:return: clusters: list with each element being a dictionary containing information for that cluster (like mu, cov)
	"""
	clusters = []
	idx = np.arange(X.shape[0])

	# initialize with GMM results: Mean and Covariances
	gmm = GaussianMixture(n_components=2, covariance_type='full')
	#
	results = gmm.fit_predict(X)
	#
	print("Show GMM initialization:")
	print(results)
	mu_k = gmm.means_

	# initialize with covariance matrix from the GMM results
	cov_mat = gmm.covariances_

	for i in range(n_clusters):
		clusters.append({
			'pi_k': 1.0 / n_clusters,
			'mu_k': mu_k[i],
			'cov_k': cov_mat[i]
		})

	return clusters


### functions to draw samples from regulated eigen spectrum

# perform eigen spectrum regularization
# first compute positive class covariance matrix, do eigen decomposition

def pos_class_covariance(ts_pos_low_d):
	# input:
	# ts_pos_low_d: positive class in eigen signal space
	# return:
	# q1_bar: p_class mean vector
	# cov_pos: covariance matrix of positive class (calculated based on the samples converted to eigen signal space)
	pos_sample_cnt = ts_pos_low_d.shape[1]
	# first compute positive-class mean vector
	q1_bar = (1 / pos_sample_cnt) * np.sum(ts_pos_low_d, axis=1)
	# compute the covariance matrix of positive training data (20 samples in the training set)
	q1_centered_t = ts_pos_low_d.transpose() - q1_bar.transpose()
	cov_pos = (1 / pos_sample_cnt) * np.dot(q1_centered_t.transpose(), q1_centered_t)

	return q1_bar, cov_pos


# define regularized eigen spectrum
def reg_eigen_spectrum(cov_pos):
	# return:
	# v_pos: eigen axes matrix
	# regu_eigen_values: regularized eigen values
	# M: the index where reliable and unreliable eigen spectrum separation point

	# eigen decomposition of positive covariance matrix
	w_pos, v_pos = np.linalg.eig(cov_pos)
	#
	w_pos = np.real(w_pos)
	#
	M = np.where(w_pos < 5e-5)
	#
	M = M[0][0] - 1
	# define the paramters used to generate regulated eigen spectrum
	d = w_pos
	regu_eigen_values = np.zeros(w_pos.shape[0])
	Alpha = d[0] * d[M] * (M - 1) / (d[1] - d[M])
	Beta = (M * d[M] - d[1]) / (d[1] - d[M])
	# populate regulated eigen spectrum values
	for i in range(0, w_pos.shape[0]):
		if (i < M):
			regu_eigen_values[i] = d[i]
		elif (i >= M):
			regu_eigen_values[i] = Alpha / (i + Beta)

	return v_pos, regu_eigen_values, M


# baseline function to draw samples from multivariate gaussian
def draw_samples_regu_eigen_wrapper(n_samples, train_p, train_n):
	# This function draws n_samples for one target_cluster
	# for each cluster, you need to run this method once

	# input
	# n_samples: number of samples to draw
	# mu: mean vector of the target cluster
	# sigma: covariance matrix for that cluster

	# first convert original data into eigen signal space
	ts_neg_low_d, ts_pos_low_d, eigen_signal, w = to_eigen_signal_space_per_cluster(train_p, train_n)
	# compute positive class covariance matrix
	q1_bar, cov_pos = pos_class_covariance(ts_pos_low_d)
	# perform eigen spectrum regularization
	v_pos, regu_eigen_values, M = reg_eigen_spectrum(cov_pos)
	# draw samples
	for i in range(0, n_samples):
		x = draw_samples_eigen_regu(q1_bar, v_pos, regu_eigen_values, M)
		if i == 0:
			new_samples = x
		else:
			new_samples = np.vstack((new_samples, x))

	print('finished generation of %d samples' % n_samples)

	return new_samples


def draw_samples_eigen_regu(q1_bar, v_pos, regu_eigen_values, M):
	# this function draws ONE sample from the regularized eigen spectrum

	# return: x: this is the synthesized sample in eigen signal space

	mu_a = np.zeros(q1_bar.shape[0])
	cov_a = np.identity(q1_bar.shape[0])

	# we draw samples from a MVN of M-length for the reliable eigen spectrum
	# and samples from a MVN for the unreliable eigen spectrum feature_length - M

	Rn = M  # Rn is reliable eigen index range
	Un = q1_bar.shape[0] - M  # Un is unreliable eigen index range

	mu_R = np.zeros(Rn)
	cov_R = np.identity(Rn)
	#
	mu_U = np.zeros(Un)
	cov_U = np.identity(Un)
	#
	aR = np.random.multivariate_normal(mu_R, cov_R, 1)
	aU = np.random.multivariate_normal(mu_U, cov_U, 1)
	#
	a = np.hstack((aR, aU))
	#
	dd = np.sqrt(regu_eigen_values)
	a_1 = np.multiply(a, dd)
	#
	x = a_1.dot(v_pos.transpose()) + q1_bar

	return x

# # NOT USED: baseline function to draw samples from multivariate gaussian
# def draw_samples(n_samples, mu, sigma):
# 	new_samples = np.random.multivariate_normal(mu, sigma, n_samples)
# 	# new_samples shape: n_samples * n_features
#
# 	return new_samples


def to_eigen_signal_space_per_cluster(train_p, train_n):
	# Input is now train_p and train_n
	# train_p: P data in that cluster, format is n_samples * n_features
	# train_n: entire N data in original data set
	##########################
	## first covert data into eigenvector space (lower dimensional space)
	##########################
	#     # (1) seperate negative and positive class
	#     train_x,train_y_binary=pre_process(data_dir,file_name_train,minority_label,down_sample_minority=False,
	#     minority_div=1)
	#     #
	#     train_p = train_x[train_y_binary==1]
	#     train_n = train_x[train_y_binary==0]
	#
	train_p = train_p.transpose()
	train_n = train_n.transpose()
	#
	ts_pos = train_p.transpose()
	ts_neg = train_n.transpose()
	# first compute x_bar
	sum_ts_n = np.sum(ts_neg, axis=1)
	sum_ts_p = np.sum(ts_pos, axis=1)
	#
	n_samples = ts_pos.shape[1] + ts_neg.shape[1]
	#
	ts_bar = (1 / n_samples) * (sum_ts_n + sum_ts_p)
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
	feature_len = train_p.shape[1]
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

	# returned data shape is n_features * n_samples
	# eigen_signal is original_n_features * lower_n_features matrix
	# w is eigen value spectrum
	return ts_neg_low_d, ts_pos_low_d, eigen_signal, w

#########################
# Temporarily modify the code to draw all target samples in ONE run
########################
def generate_new_samples_regu_eigen(clusters, num_new_samples):
	# For initial implementation, assuming we have 2 clusters
	# determine which cluster has a smaller pi
	#     if clusters[0]['pi_k']<clusters[1]['pi_k']:
	#         smaller_pi_cluster_index = 0
	#     else:
	#         smaller_pi_cluster_index = 1

	#     # assign the number of news samples to be generated from each cluster
	#     if smaller_pi_cluster_index == 0:
	# #         num_new_samples_c0 = 1
	# #         num_new_samples_c1 = 1
	#         num_new_samples_c0 = 1
	#         num_new_samples_c1 = int(round(clusters[1]['pi_k'].item(0)/clusters[0]['pi_k'].item(0)))
	#         print("Number of new samples to be generated for c0 and c1:")
	#         print(num_new_samples_c0, num_new_samples_c1)

	#     elif smaller_pi_cluster_index == 1:
	# #         num_new_samples_c1 = 1
	# #         num_new_samples_c0 = 1
	#         num_new_samples_c0 = int(round(clusters[0]['pi_k'].item(0)/clusters[1]['pi_k'].item(0)))
	#         num_new_samples_c1 = 1
	###############
	# Temp: for now we will generate all samples in one run: sample count be proportional to cluster size
	#################
	num_new_samples_c0 = round(num_new_samples * clusters[0]['pi_k'].item(0))
	num_new_samples_c1 = round(num_new_samples * clusters[1]['pi_k'].item(0))
	#
	print("Number of new samples to be generated for c0 and c1:")
	print(num_new_samples_c0, num_new_samples_c1)
	print("cluster 0 pi_k")
	print(clusters[0]['pi_k'])
	print("cluster 1 pi_k")
	print(clusters[1]['pi_k'])

	# now draw samples (number of new samples given in above calculation)
	# draw_samples_regu_eigen_wrapper(n_samples, train_p, train_n)
	new_samples_c0 = draw_samples_regu_eigen_wrapper(num_new_samples_c0, train_p, train_n)
	new_samples_c1 = draw_samples_regu_eigen_wrapper(num_new_samples_c1, train_p, train_n)

	return new_samples_c0, new_samples_c1


###############
#Expecation and Maximization step
###############

def expectation_step(X, clusters, epsilon):
	"""
	X: the original minority data trunk
	new_samples: the additional samples generated
	clusters: contains two gaussion distribution parameters
	"""
	# first combine X and new_samples
	total_X = X
	#
	totals = np.zeros((total_X.shape[0], 1), dtype=np.float64)
	#
	for cluster in clusters:
		pi_k = cluster['pi_k']
		mu_k = cluster['mu_k']
		cov_k = cluster['cov_k']

		# modify cov_k
		cov_k_modified = modify_cov_mat(epsilon, cov_k)

		#         print("shape of mu_k, before compute gamma_nk")
		#         print(mu_k.shape)
		#
		gamma_nk = (pi_k * gaussian(total_X, mu_k, cov_k_modified)).reshape(total_X.shape[0], 1)
		#         print("Debug: inside expectation_step: value of gamma_nk")
		#         print(gamma_nk)
		#         print("Debug: size of gamma_nk")
		#         print(gamma_nk.shape)

		for i in range(total_X.shape[0]):
			totals[i] += gamma_nk[i]

		#         logging.warning("value of totals")
		#         logging.warning(totals)
		cluster['gamma_nk'] = gamma_nk
		cluster['totals'] = totals

	for cluster in clusters:
		cluster['gamma_nk'] /= cluster['totals']


def maximization_step(X, clusters):
	N = float(X.shape[0])

	for cluster in clusters:
		gamma_nk = cluster['gamma_nk']
		cov_k = np.zeros((X.shape[1], X.shape[1]))

		N_k = np.sum(gamma_nk, axis=0)

		pi_k = N_k / N
		mu_k = np.sum(gamma_nk * X, axis=0) / N_k

		for j in range(X.shape[0]):
			diff = (X[j] - mu_k).reshape(-1, 1)
			cov_k += gamma_nk[j] * np.dot(diff, diff.T)

		cov_k /= N_k

		cluster['pi_k'] = pi_k
		cluster['mu_k'] = mu_k
		cluster['cov_k'] = cov_k


################
# helper functions
################
# define multivariate gaussian PDF function
def gaussian(X, mu, cov):
	n = X.shape[1]
	diff = (X - mu).T
	return np.diagonal(1 / ((2 * np.pi) ** (n / 2) * np.linalg.det(cov) ** 0.5) * np.exp(
		-0.5 * np.dot(np.dot(diff.T, np.linalg.inv(cov)), diff))).reshape(-1, 1)


def get_likelihood(X, clusters):
	likelihood = []
	sample_likelihoods = np.log(np.array([cluster['totals'] for cluster in clusters]))
	return np.sum(sample_likelihoods), sample_likelihoods


# compute overall sample covariance matrix
def get_cov_matrix(X):
	# this covariance matrix is on the data that has already been transformed into eigen vector space
	# input data shape: n_samples*n_features
	# input data shape required for np.cov function is n_features*n_samples
	X = X.transpose()
	cov_mat = np.cov(X)

	return cov_mat


# pertub covariance matrix
def modify_cov_mat(epsilon, cov_mat):
	cov_mat_modified = cov_mat + epsilon * np.identity(ts_pos_low_d.shape[0])

	return cov_mat_modified


def back_original_feature_space(new_samples, eigen_signal):
	# the new_samples generated is in the reduced eigen signal space, convert it original feature space
	# eigen_signal matrix is n_original_features * n_reduced_features
	new_samples_original_feature = np.dot(eigen_signal, new_samples.transpose())

	return new_samples_original_feature


# this function used when need to truncate the generated new samples according to the cluster size
# also to get a total of 45 samples
## Take total of 45 samples from SPO samples
def truncate_spo_samples(target_num, total_new_samples_c0, total_new_samples_c1):
	ratio = target_num / (total_new_samples_c0.shape[0] + total_new_samples_c1.shape[0])
	print(ratio)
	num_samples_c0 = int(total_new_samples_c0.shape[0] * ratio)
	num_samples_c1 = int(total_new_samples_c1.shape[0] * ratio)
	#
	truncated_samples_c0 = total_new_samples_c0[0:num_samples_c0, :]
	truncated_samples_c1 = total_new_samples_c1[0:num_samples_c1, :]
	#

	return truncated_samples_c0, truncated_samples_c1



