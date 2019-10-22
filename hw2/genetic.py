""" modified from <https://github.com/ahmedfgad/GeneticAlgorithmPython>"""

import random
import numpy
from sklearn.metrics import accuracy_score

def reduce_features(solution, features):
    selected_elements_indices = numpy.where(solution == 1)[0]
    reduced_features = features[:, selected_elements_indices]
    return reduced_features


def classification_accuracy(labels, predictions):
    return accuracy_score(labels, predictions)


def cal_pop_fitness(pop, features, labels, train_indices, test_indices, clf):
    """ takes instantiated classifier that is ready to fit and predict - must conform to sklearn interface, ie fit and predict func """
    accuracies = numpy.zeros(pop.shape[0])
    idx = 0

    for curr_solution in pop:
        reduced_features = reduce_features(curr_solution, features)
        if reduced_features.size == 0:
            continue
        train_data = reduced_features[train_indices, :]
        test_data = reduced_features[test_indices, :]

        train_labels = labels[train_indices]
        test_labels = labels[test_indices]

        clf.fit(train_data, train_labels)

        predictions = clf.predict(test_data)
        accuracies[idx] = classification_accuracy(test_labels, predictions)
        idx = idx + 1
    return accuracies


def get_parent(pop, features, labels, train_indices, test_indicies, clf):
    # get 3 random indices
    indices = numpy.random.choice(pop.shape[0], 3)
    random_picks = pop[indices]
    fitnesses = cal_pop_fitness(random_picks,features, labels, train_indices, test_indicies, clf)
    max_fit_i = numpy.where(fitnesses == numpy.max(fitnesses))
    max_fit_i = max_fit_i[0][0]
    return random_picks[max_fit_i,:]


def crossover(parents):
    n = len(parents[0])
    c = random.randrange(0, n)

    return numpy.append(parents[0, :c], parents[1,c:], axis=0)


def mutation(offspring, gene_pool=[0, 1], pmut=.2):
    if random.uniform(0, 1) >= pmut:
        return offspring

    n = len(offspring)
    g = len(gene_pool)
    c = random.randrange(0, n)
    r = random.randrange(0, g)

    new_gene = gene_pool[r]
    plus_gene = numpy.append(offspring[:c], [new_gene], axis=0)
    rest = numpy.append(plus_gene, offspring[c + 1:], axis=0)
    return rest