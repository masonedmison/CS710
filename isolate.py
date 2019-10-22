import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder
import random
import bisect

# data
zoo1_df = pd.read_csv('/Users/MasonBaran/PycharmProjects/CS710/hw2/data/zoo1.csv', header=None) # 3 classes (str values)
# drop animal col for both zoo1 and zoo2 (col_index 1) assuming they will not tell us much
zoo1_df.drop(0,axis=1, inplace=True)

# seperate train and test sets
test_indices = [91, 92, 93, 95, 97, 99]
_,col_shape = zoo1_df.shape # length of column index - 1 == our label feature
###############################################

# factorize specicies labels - use global label_str to retrieve string using encoded label as index
label_col = zoo1_df.iloc[:,col_shape-1]
labels_encoded, label_str = label_col.factorize()
zoo1_df.iloc[:,col_shape-1] = labels_encoded

# zoo1
    # training data
zoo1_train = zoo1_df.drop(zoo1_df.index[test_indices])
zoo1_train_labels = zoo1_train.iloc[:,col_shape-1]
zoo1_train_features = zoo1_train.drop(col_shape, axis=1)
    # testing data
zoo1_test = zoo1_df.iloc[test_indices,:]
zoo1_test_labels = zoo1_test.iloc[:,col_shape-1]
zoo1_test_features = zoo1_test.drop(col_shape, axis=1)
##############################################
zoo1_X = zoo1_train.to_numpy()

def crossover(parents, offspring_size=1):
    offspring = np.empty(offspring_size)
    # The point at which crossover takes place between two parents. Usually, it is at the center.
    # crossover_point = numpy.uint8(offspring_size[1]/2)
    n = len(parents[0])
    c = random.randrange(0, n)
    print(c)

    return np.append(parents[0, :c], parents[1,c:], axis=0)


n = np.array([ [1,2,3,4,5], [6,7,8,9,10] ])

print(n[0, :])

print(crossover(n))