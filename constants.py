import os


##########
# Parameters for selected data set
##########
# Assume I am under git repo directory:  ~ /repo_oversampling_algorithm_multivariate_time_series
DATA_DIR = '/Users/jiedali/Documents/research/dataset/Multivariate_ts/NATOPS/'
FILE_NAME_TRAIN = 'NATOPS_TRAIN.ts'
FILE_NAME_TEST = 'NATOPS_TEST.ts'
MINORITY_LABEL = '1.0'
DATA_LABEL = 'NATOPS'
DOWN_SAMPLE_MINORITY = True
MINORITY_DIV = 3
##########
#parameters related to the choice of method and number repeats
##########
NUM_REPEATS = 1
###########
#parameters related to file names
PLOT_NAME= ''
###########


###########
# Saved configurations for other data sets
###########
# RacketSports
# Assume I am under git repo directory:  ~ /repo_oversampling_algorithm_multivariate_time_series
# DATA_DIR = '/Users/jiedali/Documents/research/dataset/Multivariate_ts/RacketSports/'
# FILE_NAME_TRAIN = 'RacketSports_TRAIN.ts'
# FILE_NAME_TEST = 'RacketSports_TEST.ts'
# MINORITY_LABEL = 'badminton_clear'
# DATA_LABEL = 'RacketSports'
# DOWN_SAMPLE_MINORITY = True
# MINORITY_DIV = 4


# EthanolConcentration

# DATA_DIR = '/Users/jiedali/Documents/research/dataset/Multivariate_ts/EthanolConcentration/'
# FILE_NAME_TRAIN = 'EthanolConcentration_TRAIN.ts'
# FILE_NAME_TEST = 'EthanolConcentration_TEST.ts'
# MINORITY_LABEL = 'e45'
# DATA_LABEL = 'EthanolConcentration'
# DOWN_SAMPLE_MINORITY = True
# MINORITY_DIV = 3

# FingerMovements

# DATA_DIR = '/Users/jiedali/Documents/research/dataset/Multivariate_ts/FingerMovements/'
# FILE_NAME_TRAIN = 'FingerMovements_TRAIN.ts'
# FILE_NAME_TEST = 'FingerMovements_TEST.ts'
# MINORITY_LABEL = 'left'
# DATA_LABEL = 'FingerMovements'
# DOWN_SAMPLE_MINORITY = True
# MINORITY_DIV = 10