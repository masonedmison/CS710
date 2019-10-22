# dependencies
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from math import sqrt
import numpy as np
from collections import namedtuple
from itertools import cycle
from homework1.maze import SequenceMaze, Node
from homework1.mazes import mazes_to_test
from random import choice
import operator
from sklearn.tree import DecisionTreeClassifier



# print(one_right, one_left)

from homework1.priority_queue import PriorityQueue


# data
zoo1_df = pd.read_csv('/Users/MasonBaran/PycharmProjects/CS710/hw2/data/zoo1.csv', header=None) # 3 classes (str values)
zoo2_df = pd.read_csv('/Users/MasonBaran/PycharmProjects/CS710/hw2/data/zoo2.csv', header=None) # 2 values (numberic)

# drop animal col for both zoo1 and zoo2 (col_index 1) assuming they will not tell us much
zoo1_df.drop(0,axis=1, inplace=True)

# seperate train and test sets
test_indices = [91, 92, 93, 95, 97, 99]
_,col_shape = zoo1_df.shape # length of column index - 1 == our label feature
###############################################
# zoo1
# training data
zoo1_train = zoo1_df.drop(zoo1_df.index[test_indices])
zoo1_train_labels = zoo1_train.iloc[:,col_shape-1]
zoo1_train_features = zoo1_train.drop(col_shape, axis=1)
# testing data
zoo1_test = zoo1_df.iloc[test_indices,:]
zoo1_test_labels = zoo1_test.iloc[:,col_shape-1]
zoo1_test_features = zoo1_test.drop(col_shape, axis=1)
###############################################
# zoo2
# training data
zoo2_train = zoo2_df.drop(zoo1_df.index[test_indices])
zoo2_train_labels = zoo2_train.iloc[:,col_shape-1]
zoo2_train_features = zoo2_train.drop(col_shape-1, axis=1)
# testing data
zoo2_test = zoo2_df.iloc[test_indices,:]
zoo2_test_labels = zoo2_test.iloc[:,col_shape-1]
zoo2_test_features = zoo2_test.drop(col_shape-1, axis=1)


class NaiveBayes:
    """simple naive bayes implementation
    parameters: training features (as np array) and training labels (list of string values)"""

    def __init__(self):
        pass

    def fit(self, X, y):
        def occurrences(list1):
            no_of_examples = len(list1)
            prob = dict(Counter(list1))
            for key in prob.keys():
                prob[key] = prob[key] / float(no_of_examples)
            return prob

        self.classes = np.unique(y)
        rows, cols = np.shape(X)
        self.likelihoods = {}
        for cls in self.classes:
            self.likelihoods[cls] = defaultdict(list)
        self.class_probabilities = occurrences(y)
        for cls in self.classes:
            row_indices = np.where(y == cls)[0]
            subset = X[row_indices, :]
            r, c = np.shape(subset)
            for j in range(0, c):
                self.likelihoods[cls][j] += list(subset[:, j])

        for cls in self.classes:
            for j in range(0, cols):
                self.likelihoods[cls][j] = occurrences(self.likelihoods[cls][j])

    def predict(self, sample):
        results = {}
        for cls in self.classes:
            class_probability = self.class_probabilities[cls]
            for i in range(0, len(sample)):
                relative_values = self.likelihoods[cls][i]
                if sample[i] in relative_values.keys():
                    class_probability *= relative_values[sample[i]]
                else:
                    class_probability *= 0
                results[cls] = class_probability
        return (results, max(results, key=lambda key: results[key]))

def zoo_to_np(zoo_features, zoo_labels):
    X = zoo_features.to_numpy()
    y_encoded, y_features = zoo_labels.factorize()
    return X, y_encoded, y_features

# sanity check FOR NAIVE BAYES to run after rafactor - from example
def sanity_check():
    training   = np.asarray(((1,0,1,1),(1,1,0,0),(1,0,2,1),(0,1,1,1),(0,0,0,0),(0,1,2,1),(0,1,2,0),(1,1,1,1)));
    y = np.asarray((0,1,1,1,0,1,0,1))
    new_sample = np.asarray((1,0,1,0))
    y_labels = pd.Series(['No','Yes','Yes', 'Yes', 'No', 'Yes', 'No', 'Yes'])
    y_encoded, y_labels = y_labels.factorize()
    nb = NaiveBayes()
    nb.fit(training, y_encoded)
    ress, best_val = nb.predict(new_sample)
    print('Testing naive bayes results...')
    assert ress == {0: 0.018518518518518517, 1: 0.006000000000000002}
    print('-'*30)
    print('assertion passed!')


def nb_predict_zoo1():
    X, y_encoded, y_labels = zoo_to_np(zoo1_train_features, zoo1_train_labels) # zoo1
    nb = NaiveBayes()
    nb.fit(X,y_encoded)
    print(zoo1_test_features)
    X_test = zoo1_test_features.to_numpy()
    for index, r in enumerate(X_test):
        res, best = nb.predict(r)
        print(res)
        print(best)
        print(f'Predicted animal is: {y_labels[best]}')
        print(f'Correct animal is: {zoo1_test_labels.iloc[index]}')

nb_predict_zoo1()