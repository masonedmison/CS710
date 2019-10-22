""" EXTRA CREDIT: get best 6, 10, 15, and 17 features for zoo1 data
script to run genetic algorithm foe feature selection for zoo data"""

import numpy as np
import pickle
import matplotlib.pyplot
import pandas as pd
import hw2.genetic_fe as GA
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedShuffleSplit

# vars of interest
CLASSIFIER = DecisionTreeClassifier() # classifier to use for fitness - must implement sklearn fit and predict methods
NUM_OPT_FEATURES = 2 # number of features to initialize ininitial population
##################################
# other parameters to tweak
POP = 15 # Population size.
NGENS = 50 # number of generations
##################################

attributes = ['animal name','hair', 'feathers', 'eggs', 'milk', 'airborne', 'aquatic', 'predator', 'toothed',
              'backbone', 'breathes','venomous','fins', 'legs', 'tail', 'domestic', 'catsize' ,'type']

# data
zoo1_df = pd.read_csv('/Users/MasonBaran/PycharmProjects/CS710/hw2/data/zoo1.csv', header=None) # 3 classes (str values)
# drop animal col for both zoo1 and zoo2 (col_index 1) assuming they will not tell us much
zoo1_df.drop(0,axis=1, inplace=True)
zoo1_df.reset_index()

# seperate train and test sets
test_indices = [91, 92, 93, 95, 97, 99]
_,col_shape = zoo1_df.shape # length of column index - 1 == our label feature

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
zoo1_X = zoo1_train_features.to_numpy() # as numpy matrix so we can do fun numpy things
zoo1_Y = zoo1_train_labels.to_numpy()
##############################################

num_samples = zoo1_train_features.shape[0]
num_feature_elements = zoo1_train_features.shape[1]

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(zoo1_train, zoo1_train.iloc[:, 13]):
    train_indices = train_index
    test_indices = test_index
print("Number of training samples: ", train_indices.shape[0])
print("Number of test samples: ", test_indices.shape[0])


# Defining the population shape.
pop_shape = (POP, num_feature_elements)


N = zoo1_train_features.shape[1]
K = N - NUM_OPT_FEATURES # K zeros, N-K ones
# make new population
old_population = np.zeros(pop_shape)
for i in range(POP):
    arr = np.array([0] * K + [1] * (N-K))
    np.random.shuffle(arr)
    old_population[i, :] = arr
###

# best outputs to record the number of features with best fitness score
best_outputs = dict()
best_out_seq = []
new_population = np.zeros(pop_shape)
for generation in range(NGENS):
    num_of_matings = 0 # also tells us index of children being addded

    print("Generation : ", generation)
    while num_of_matings < POP:
        # Measuring the fitness of each chromosome in the population.
        parents = np.asarray([GA.get_parent(old_population, zoo1_X, zoo1_Y, train_indices, test_indices, CLASSIFIER) for i in range(2)])

        # Generating next generation using crossover.
        offspring_crossover = GA.crossover(parents)
        # Adding some variations to the offspring using mutation.
        offspring_mutation = GA.mutation(offspring_crossover)


        new_population[num_of_matings,:] = offspring_mutation # use num_of_matings as index
        num_of_matings += 1

    # get most fit individual at end of generation
    fitnesses = GA.cal_pop_fitness(new_population, zoo1_X, zoo1_Y, train_indices, test_indices, CLASSIFIER)
    max_fit = np.max(fitnesses)
    max_fit_i = np.where(fitnesses == max_fit)
    max_fit_i = max_fit_i[0][0]
    best_outputs[max_fit] = new_population[max_fit_i,:]
    best_out_seq.append(max_fit) # to see progression of iters
    old_population = new_population # point old_population to new_population as this is where we will look for parents for next gen


best_fitnesses = list(best_outputs.keys())
best_fitness = max(best_fitnesses) # key that points to solution and possibly other things
best_solution = best_outputs[best_fitness]
best_solution_indices = np.where(best_solution == 1)[0]

print('[SOLUTION]')
print("best_solution : ", best_solution)
print("Selected indices : ", best_solution_indices)
print("Number of selected elements : ", best_solution_indices.shape[0])
print("Best solution fitness : ", best_fitness)


# plot
matplotlib.pyplot.plot(best_out_seq)
matplotlib.pyplot.xlabel("Generation")
matplotlib.pyplot.ylabel("Fitness")

matplotlib.pyplot.savefig('figures/50gen_15pop.png', format='png') # save