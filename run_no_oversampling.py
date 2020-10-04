from em_workflow import em_workflow
from em_algorithm import train_gmm
from em_algorithm import *
import statistics
from visualization import visualize
import constants as const

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
#parameters related to the choice of method and number repeats
##########
num_repeats = const.NUM_REPEATS
###########
#parameters related to file names
plot_name = const.PLOT_NAME
###########

# step 1: create an instance of em_workflow class
workflow1 = em_workflow(data_dir=data_dir, file_name_train=file_name_train, file_name_test=file_name_test,
                        minority_label=minority_label, data_label=data_label, down_sample_minority=down_sample_minority,
                        minority_div=minority_div)

# step 2: run classification

f1_score = workflow1.workflow_no_oversampling()
print(f1_score)