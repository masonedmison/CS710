""" EXTRA CREDIT: get best 6, 10, 15, and 17 features for zoo1 data
**code modified from <https://github.com/ahmedfgad/GeneticAlgorithmPython>**"""

import numpy as np
import pickle
import matplotlib.pyplot
import pandas as pd
import hw2.genetic as GA
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedShuffleSplit

# variables to change for features to optimize and classifier
CLASSIFIER = DecisionTreeClassifier()
NUM_OPT_FEATURES = 6
##################################
# other parameters to tweak
POP = 35 # Population size.
num_parents_mating = 2 # Number of parents inside the mating pool.
num_mutations = 2 # Number of elements to mutate.
num_generations = 8
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

"""
Genetic algorithm parameters:
    Population size
    Mating pool size
    Number of mutations
"""

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
best_outputs = []
new_population = np.zeros(pop_shape)
for generation in range(num_generations):
    num_of_matings = 0 # also tells us index of children being addded

    print("Generation : ", generation)
    while num_of_matings < POP:
        # Measuring the fitness of each chromosome in the population.
        # fitness, fit_num_features = GA.cal_pop_fitness(old_population, zoo1_X, zoo1_Y, train_indices, test_indices, CLASSIFIER)
        parents = np.asarray([GA.get_parent(old_population, zoo1_X, zoo1_Y, train_indices, test_indices, CLASSIFIER) for i in range(2)])
        # best_outputs.append(np.max(fitness))
        # The best result in the current iteration.
        # print("Best result : ", best_outputs[-1])

        # Selecting the best parents in the population for mating.
        # parents = GA.select_mating_pool(new_population, fitness, num_parents_mating)

        # Generating next generation using crossover.
        offspring_crossover = GA.crossover(parents)
        # Adding some variations to the offspring using mutation.
        offspring_mutation = GA.mutation(offspring_crossover)

        # Creating the new population based on the parents and offspring.
        # new_population[0:parents.shape[0], :] = parents
        # new_population[parents.shape[0]:, :] = offspring_mutation
        new_population[num_of_matings,:] = offspring_mutation # use num_of_matings as index
        num_of_matings += 1
    print('at end of generation', new_population.shape)

    old_population = new_population # point old_population to new_population as this is where we will look for parents for next gen





# Getting the best solution after iterating finishing all generations.
# At first, the fitness is calculated for each solution in the final generation.
fitness, _ = GA.cal_pop_fitness(new_population, zoo1_X, zoo1_Y, train_indices, test_indices, CLASSIFIER)
# Then return the index of that solution corresponding to the best fitness.
best_match_idx = np.where(fitness == np.max(fitness))[0]
best_match_idx = best_match_idx[0]

best_solution = new_population[best_match_idx, :]
best_solution_indices = np.where(best_solution == 1)[0]
best_solution_num_elements = best_solution_indices.shape[0]
best_solution_fitness = fitness[best_match_idx]

print("best_match_idx : ", best_match_idx)
print("best_solution : ", best_solution)
print("Selected indices : ", best_solution_indices)
print("Number of selected elements : ", best_solution_num_elements)
print("Best solution fitness : ", best_solution_fitness)
#
# matplotlib.pyplot.plot(best_outputs)
# matplotlib.pyplot.xlabel("Iteration")
# matplotlib.pyplot.ylabel("Fitness")
# matplotlib.pyplot.show()