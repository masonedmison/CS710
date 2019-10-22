import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder
import random
import bisect

# # data
# zoo1_df = pd.read_csv('/Users/MasonBaran/PycharmProjects/CS710/hw2/data/zoo1.csv', header=None) # 3 classes (str values)
# # drop animal col for both zoo1 and zoo2 (col_index 1) assuming they will not tell us much
# zoo1_df.drop(0,axis=1, inplace=True)
#
# # seperate train and test sets
# test_indices = [91, 92, 93, 95, 97, 99]
# _,col_shape = zoo1_df.shape # length of column index - 1 == our label feature
# ###############################################
#
# # factorize specicies labels - use global label_str to retrieve string using encoded label as index
# label_col = zoo1_df.iloc[:,col_shape-1]
# labels_encoded, label_str = label_col.factorize()
# zoo1_df.iloc[:,col_shape-1] = labels_encoded
#
# # zoo1
#     # training data
# zoo1_train = zoo1_df.drop(zoo1_df.index[test_indices])
# zoo1_train_labels = zoo1_train.iloc[:,col_shape-1]
# zoo1_train_features = zoo1_train.drop(col_shape, axis=1)
#     # testing data
# zoo1_test = zoo1_df.iloc[test_indices,:]
# zoo1_test_labels = zoo1_test.iloc[:,col_shape-1]
# zoo1_test_features = zoo1_test.drop(col_shape, axis=1)
# ##############################################
# zoo1_X = zoo1_train.to_numpy()
#
mag_look_up = {1:{'cost':20, 'reach':6}, 2:{'cost':30, 'reach':5}, 3:{'cost':60, 'reach':8}, 4:{'cost':70, 'reach':9},
               5: {'cost': 50, 'reach': 6}, 6:{'cost':90, 'reach':7}, 7:{'cost':40, 'reach':3}}

seq = [0., 0., 0., 4., 1.]
print(sum([mag_look_up[m]['reach'] for m in seq if m != 0]))