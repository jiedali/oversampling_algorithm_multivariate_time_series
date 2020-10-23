import os
from sktime.utils.load_data import load_from_tsfile_to_dataframe
import constants as const

#
data_dir = const.DATA_DIR
file_name_train = const.FILE_NAME_TRAIN
train_x, train_y = load_from_tsfile_to_dataframe(os.path.join(data_dir, file_name_train))