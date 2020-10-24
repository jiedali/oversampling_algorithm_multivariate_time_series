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

def train_gmm(X, train_n, n_clusters, n_epochs, epsilon, num_new_samples, eigen_signal_overall):
	"""
	This function implement the latest EM algorithm where we initialize the clusters with GMM
	Add perturbation to covariance matrix
	:param X: minority data chunk in eigen-signal space, n_samples * n_features
	:param n_clusters: number of gaussian modes
	:param n_epochs: number of epochs to run for EM algorithm
	:param epsilon: covariance matrix perturbation, typically set to 0.01
	:param num_new_samples: desired number of new samples to generate

	:return: clusters, likelihoods, scores, sample_likelihoods, history, total_new_samples_c0, total_new_samples_c1
	clusters: list with each element a dictionary containing information for each cluster
	new_samples_all_clusters: a LIST of numpy array, each being the new samples for each cluster;
	For each cluster, the numpy array has a shape of: n_samples * n_features
	"""

	clusters, results = initialize_clusters_with_gmm_results(X, n_clusters)

	# return the clustering membership
	clustering_results = results
	likelihoods = np.zeros((n_epochs,))
	scores = np.zeros((X.shape[0], n_clusters))
	history = []

	for i in range(n_epochs):

		# print('Pi_k at the beginning of %d iteration' % i)
		# print('cluster0', clusters[0]['pi_k'])
		# print('cluster1', clusters[1]['pi_k'])

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
			new_samples_all_clusters = generate_new_samples_regu_eigen(clusters, num_new_samples, results, X,
			                                                                 train_n, eigen_signal_overall)
			# if i == 1:
			# 	total_new_samples_c0 = new_samples_c0
			# 	total_new_samples_c1 = new_samples_c1
			# # else:
			# # 	total_new_samples_c0 = np.vstack((total_new_samples_c0, new_samples_c0))
			# # 	total_new_samples_c1 = np.vstack((total_new_samples_c1, new_samples_c1))
		#
		expectation_step(X, clusters, epsilon)
		#
		maximization_step(X, clusters)

		likelihood, sample_likelihoods = get_likelihood(X, clusters)
		likelihoods[i] = likelihood

		print('Epoch: ', i + 1, 'Likelihood: ', likelihood)

	for i, cluster in enumerate(clusters):
		scores[:, i] = np.log(cluster['gamma_nk']).reshape(-1)

	return clusters, clustering_results, likelihoods, scores, sample_likelihoods, history, new_samples_all_clusters


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
	gmm = GaussianMixture(n_components=n_clusters, covariance_type='full')
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

	return clusters, results


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
	# print("debug, w_pos:")
	# print(w_pos)
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
	"""
	This function wraps up the entire process of drawing samples from a regulated eigen covariance structure
	Step 1: convert input data p/n into eigen signal space
	Step 2: compute the regulated eigen covariance matrix
	Step 3: draw new samples from that regulate eigen matrix
	Step 4: covert new samples back to original feature space (that is the same feature dimension as train_p/train_n)
	Step 5: return the new samples in original feature space

	:param n_samples: number of samples to generate
	:param train_p: minority data chunk of a specific cluster, in original feature space
	:param train_n: majority data chunk of the entire data, in original feature space
	:return:
	"""
	# first convert original data into eigen signal space
	ts_neg_low_d, ts_pos_low_d, eigen_signal, w = to_eigen_signal_space_per_cluster(train_p, train_n)
	# compute positive class covariance matrix
	q1_bar, cov_pos = pos_class_covariance(ts_pos_low_d)
	# perform eigen spectrum regularization
	v_pos, regu_eigen_values, M = reg_eigen_spectrum(cov_pos)
	print("Debug: value of M: %d" % M)
	# draw samples
	for i in range(0, n_samples):
		x_eigen_space = draw_samples_eigen_regu(q1_bar, v_pos, regu_eigen_values, M)
		# print("new sample in eigen space:", x_eigen_space.shape)
		x = np.dot(eigen_signal, x_eigen_space.transpose())
		x = np.real(x.transpose())
		# print("new sample in original feature space:", x.shape)
		if i == 0:
			new_samples = x
		else:
			new_samples = np.vstack((new_samples, x))

	print('finished generation of %d samples' % n_samples)
	print('shape of total new samples:')
	print(new_samples.shape)

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


# NOT USED: baseline function to draw samples from multivariate gaussian
def draw_samples(n_samples, mu, sigma):
	new_samples = np.random.multivariate_normal(mu, sigma, n_samples)
	# new_samples shape: n_samples * n_features

	return new_samples


def to_eigen_signal_space_per_cluster(train_p, train_n):
	"""
	This is the first step in drawing samples from regulated eigen matrix (to convert data into eigen vector space)
	:param train_p: P data in that cluster, format is n_samples * n_features (this is a subset of original P class)
	:param train_n: entire N data in original data set
	:return:ts_neg_low_d,ts_pos_low_d,shape is n_features * n_samples
	:return:eigen_signal, is original_n_features * lower_n_features matrix
	:return: w: is eigen value spectrum
	"""
	###############
	# ts_pos shape: n_features * n_samples
	###############
	# print("debug before doing per cluster conversion from original feature space to eigen:")
	# print(train_p.shape)
	# print(train_n.shape)
	ts_pos = train_p
	ts_neg = train_n
	# first compute x_bar
	sum_ts_n = np.sum(ts_neg, axis=1)
	sum_ts_p = np.sum(ts_pos, axis=1)
	# print("debug,shape of sum_ts_n")
	# print(sum_ts_n.shape)
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
	feature_len = train_p.shape[0]
	sum_p = np.zeros(shape=(feature_len, feature_len))
	sum_n = np.zeros(shape=(feature_len, feature_len))
	for i in range(0, P):
		sum_p += np.dot(ts_pos_centered[:, i].reshape((feature_len, 1)),
		                ts_pos_centered[:, i].transpose().reshape((1, feature_len)))
	for i in range(0, N):
		sum_n += np.dot(ts_neg_centered[:, i].reshape((feature_len, 1)),
		                ts_neg_centered[:, i].transpose().reshape((1, feature_len)))

	cov_mat = (sum_p + sum_n) / (P + N)
	# eigen decomposition of covariance matrix
	w, v = np.linalg.eig(cov_mat)
	#
	null_index_array = np.where(w < 5e-5)
	if len(null_index_array[0] > 0):
		null_index = null_index_array[0][0]
	elif len(null_index_array[0]) == 0:
		print("there is no eigen value less than 5e-5")
		# null index should be the same as the index of last column of v
		null_index = v.shape[1]
	# separate the eigen vectors into signal space and null space
	eigen_signal = v[:, 0:null_index]
	# transform all samples into the lower dimensional eigen signal space
	ts_neg_low_d = np.dot(eigen_signal.transpose(), ts_neg)
	ts_pos_low_d = np.dot(eigen_signal.transpose(), ts_pos)

	print("debug: shape of the positive class after converting to lower dimension")
	print(ts_pos_low_d.shape)

	# returned data shape is n_features * n_samples
	# eigen_signal is original_n_features * lower_n_features matrix
	# w is eigen value spectrum
	return ts_neg_low_d, ts_pos_low_d, eigen_signal, w


#########################
# Temporarily modify the code to draw all target samples in ONE run
########################
def generate_new_samples_regu_eigen(clusters, num_new_samples, results, train_p, train_n, eigen_signal_overall):
	#################
	# input:
	# results: initial GMM clustering results (0,1 for each of data poin in original training set)
	# train_p: original training data - minority class, n_samples * n_features
	# train_n: original training data - majority class, n_samples * n_features
	# eigen_signal_overall: from entire original data: shape: original_feature_dimension * lower_eigen_signal_dimension
	# Note: train_p will be divided into 2 (or k) clusters, depending on input "results"
	# return:
	# new_samples_original_feature_all_clusters: this is the list of new samples generated for each cluster
	####################

	###############
	# Temp: for now we will generate all samples in one run: sample count be proportional to cluster size
	#################

	# convert train_p and train_n from eigen signal space to original feature space
	train_p_original_feature_space = np.real(np.dot(eigen_signal_overall, train_p.transpose()))
	train_n_original_feature_space = np.real(np.dot(eigen_signal_overall, train_n.transpose()))

	# transpose train_p_original_feature_space to n_sample * n_features
	train_p_original_feature_space = train_p_original_feature_space.transpose()

	# now draw samples (number of new samples given in above calculation)
	# draw_samples_regu_eigen_wrapper(n_samples, train_p, train_n)
	# print("debug: shape of train_p , train_n, train_p_original_feature_space, train_n_original_feature_space")
	# print(train_p.shape)
	# print(train_n.shape)
	# print(train_p_original_feature_space.shape)
	# print(train_n_original_feature_space.shape)
	# print(train_p_original_feature_space)

	# initialize some lists used to store parameters/results related to multi modes
	num_new_samples_all_clusters = []
	train_p_original_feature_all_clusters = []
	new_samples_original_feature_all_clusters = []

	for cluster_index in range(len(clusters)):
		# get number of new samples to generate for each cluster
		num_new_sample_per_cluster = round(num_new_samples * clusters[cluster_index]['pi_k'].item(0))
		num_new_samples_all_clusters.append(num_new_sample_per_cluster)
		print("pi_k for cluster %d" % cluster_index)
		# get the original train_p for each cluster
		train_p_ori_feature_per_cluster = train_p_original_feature_space[results == cluster_index]
		# Transpose train_p_original_feature_per_cluster, so the shape is: n_features*n_samples
		# this is the format needed for function draw_samples_regu_eigen_wrapper
		train_p_ori_feature_per_cluster = train_p_ori_feature_per_cluster.transpose()
		#
		train_p_original_feature_all_clusters.append(train_p_ori_feature_per_cluster)
		print("debug: original train_p for cluster %d has the shape" % cluster_index)
		print(train_p_ori_feature_per_cluster.shape)
		# draw new samples
		new_samples_per_cluster = draw_samples_regu_eigen_wrapper(num_new_sample_per_cluster,
		                                                          train_p_ori_feature_per_cluster,
		                                                          train_n_original_feature_space)
		print("debug: new samples shape for cluster %d" % cluster_index)
		print(new_samples_per_cluster.shape)
		#
		new_samples_original_feature_all_clusters.append(new_samples_per_cluster)

	#
	print("Number of new samples to be generated for c0 and c1:")
	print(num_new_samples_all_clusters)

	return new_samples_original_feature_all_clusters


###############
# Expecation and Maximization step
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
		cov_k_modified = modify_cov_mat(epsilon, cov_k, X)

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
def modify_cov_mat(epsilon, cov_mat, pos_low_d_transposed):
	cov_mat_modified = cov_mat + epsilon * np.identity(pos_low_d_transposed.shape[1])

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
