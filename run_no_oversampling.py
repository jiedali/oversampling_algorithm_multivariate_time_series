from em_workflow import em_workflow
from em_algorithm import train_gmm
from em_algorithm import *
import statistics
from visualization import visualize

##########
# Parameters for selected data set
##########
data_dir = '/Users/jiedali/Documents/research/dataset/Multivariate_ts/RacketSports/'
file_name_train = 'RacketSports_TRAIN.ts'
file_name_test = 'RacketSports_TEST.ts'
minority_label = 'badminton_clear'
data_label = 'RacketSports'
down_sample_minority = True
minority_div = 4
##########
#parameters related to the choice of method and number repeats
##########
num_repeats = 1
###########
#parameters related to file names
plot_name = ''
###########

# step 1: create an instance of em_workflow class
workflow1 = em_workflow(data_dir=data_dir, file_name_train=file_name_train, file_name_test=file_name_test,
                        minority_label=minority_label, data_label=data_label, down_sample_minority=down_sample_minority,
                        minority_div=minority_div)

# step 2: run classification

f1_score = workflow1.workflow_no_oversampling()
print(f1_score)