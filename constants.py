import os

# select dataset
# dataset = 'RacketSports'
# dataset = 'FingerMovements'
# dataset = 'NATOPS'
# dataset = 'UWaveGestureLibrary'
# dataset = 'EthanolConcentration'
# dataset = 'ERing'

dataset = 'LSST'
# dataset = 'PhonemeSpectra'
# dataset = 'Libras'

# unable to learn:
# dataset ='AtrialFibrillation'



if dataset == 'UWaveGestureLibrary':
	#######
	# Parameters for selected data set
	#######
	DATA_DIR = '/Users/jiedali/Documents/research/dataset/Multivariate_ts/UWaveGestureLibrary/'
	FILE_NAME_TRAIN = 'UWaveGestureLibrary_TRAIN.ts'
	FILE_NAME_TEST = 'UWaveGestureLibrary_TEST.ts'
	MINORITY_LABEL = '1.0'
	DATA_LABEL = 'HandMovementDirection'
	DOWN_SAMPLE_MINORITY = False
	MINORITY_DIV = 1

elif dataset == 'Libras':

	###########
	# Saved configurations for other data sets
	###########
	#RacketSports
	DATA_DIR = '/Users/jiedali/Documents/research/dataset/Multivariate_ts/Libras/'
	FILE_NAME_TRAIN = 'Libras_TRAIN.ts'
	FILE_NAME_TEST = 'Libras_TEST.ts'
	# imbalance ration 12: 168
	MINORITY_LABEL = '1'
	DATA_LABEL = 'Libras'
	DOWN_SAMPLE_MINORITY = False
	MINORITY_DIV = 1

elif dataset == 'ERing':

	###########
	# Saved configurations for other data sets
	###########
	#RacketSports
	DATA_DIR = '/Users/jiedali/Documents/research/dataset/Multivariate_ts/ERing/'
	FILE_NAME_TRAIN = 'ERing_TRAIN.ts'
	FILE_NAME_TEST = 'ERing_TEST.ts'
	MINORITY_LABEL = '2'
	DATA_LABEL = 'ERing'
	DOWN_SAMPLE_MINORITY = False
	MINORITY_DIV = 1

elif dataset == 'LSST':

	###########
	# Saved configurations for other data sets
	###########
	#RacketSports
	DATA_DIR = '/Users/jiedali/Documents/research/dataset/Multivariate_ts/LSST/'
	FILE_NAME_TRAIN = 'LSST_TRAIN.ts'
	FILE_NAME_TEST = 'LSST_TEST.ts'
	# imbalance ratio 128/(2459-128) = 0.05491205491205491
	MINORITY_LABEL = '88'
	DATA_LABEL = 'LSST'
	DOWN_SAMPLE_MINORITY = False
	MINORITY_DIV = 1


elif dataset == 'PhonemeSpectra':

	###########
	# Saved configurations for other data sets
	###########
	#RacketSports
	DATA_DIR = '/Users/jiedali/Documents/research/dataset/Multivariate_ts/PhonemeSpectra/'
	FILE_NAME_TRAIN = 'PhonemeSpectra_TRAIN.ts'
	FILE_NAME_TEST = 'PhonemeSpectra_TEST.ts'
	# imbalance ratio 85/3315 = 0.02564102564102564
	MINORITY_LABEL = 'th'
	DATA_LABEL = 'PhonemeSpectra'
	DOWN_SAMPLE_MINORITY = False
	MINORITY_DIV = 1


elif dataset == 'RacketSports':

	###########
	# Saved configurations for other data sets
	###########
	#RacketSports
	DATA_DIR = '/Users/jiedali/Documents/research/dataset/Multivariate_ts/RacketSports/'
	FILE_NAME_TRAIN = 'RacketSports_TRAIN.ts'
	FILE_NAME_TEST = 'RacketSports_TEST.ts'
	MINORITY_LABEL = 'badminton_clear'
	DATA_LABEL = 'RacketSports'
	DOWN_SAMPLE_MINORITY = True
	MINORITY_DIV = 4




elif dataset == 'NATOPS':
	##########
	# Parameters for selected data set
	##########
	DATA_DIR = '/Users/jiedali/Documents/research/dataset/Multivariate_ts/NATOPS/'
	FILE_NAME_TRAIN = 'NATOPS_TRAIN.ts'
	FILE_NAME_TEST = 'NATOPS_TEST.ts'
	MINORITY_LABEL = '1.0'
	DATA_LABEL = 'NATOPS'
	DOWN_SAMPLE_MINORITY = True
	MINORITY_DIV = 3
	##########

elif dataset == 'EthanolConcentration':
# EthanolConcentration

	DATA_DIR = '/Users/jiedali/Documents/research/dataset/Multivariate_ts/EthanolConcentration/'
	FILE_NAME_TRAIN = 'EthanolConcentration_TRAIN.ts'
	FILE_NAME_TEST = 'EthanolConcentration_TEST.ts'
	MINORITY_LABEL = 'e45'
	DATA_LABEL = 'EthanolConcentration'
	DOWN_SAMPLE_MINORITY = True
	MINORITY_DIV = 3
#
elif dataset == 'FingerMovements':
	# FingerMovements
	DATA_DIR = '/Users/jiedali/Documents/research/dataset/Multivariate_ts/FingerMovements/'
	FILE_NAME_TRAIN = 'FingerMovements_TRAIN.ts'
	FILE_NAME_TEST = 'FingerMovements_TEST.ts'
	MINORITY_LABEL = 'left'
	DATA_LABEL = 'FingerMovements'
	DOWN_SAMPLE_MINORITY = True
	MINORITY_DIV = 10


# elif dataset == 'AtrialFibrillation':
#
# 	###########
# 	# Saved configurations for other data sets
# 	###########
# 	#RacketSports
# 	DATA_DIR = '/Users/jiedali/Documents/research/dataset/Multivariate_ts/AtrialFibrillation/'
# 	FILE_NAME_TRAIN = 'AtrialFibrillation_TRAIN.ts'
# 	FILE_NAME_TEST = 'AtrialFibrillation_TEST.ts'
# 	# total3 classes, 'n' stands for non-termination, the other two both refer to termination, but within different time frames
# 	MINORITY_LABEL = 'n'
# 	DATA_LABEL = 'AtrialFibrillation'
# 	DOWN_SAMPLE_MINORITY = False
# 	MINORITY_DIV = 1

else:
	print("please give a valid dataset name")