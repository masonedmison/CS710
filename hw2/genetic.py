"""modified from <https://github.com/ahmedfgad/GeneticAlgorithmPython>"""

import random
import numpy




LOW_VAL = -999999



def cal_pop_fitness(pop, look_up, max_budget=100000):
    """ takes instantiated classifier that is ready to fit and predict - must conform to sklearn interface, ie fit and predict func """
    fitnesses = numpy.zeros(pop.shape[0])

    def get_sol_cost(seq):
        cost = sum([look_up[m]['cost'] for m in seq if m != 0])
        return cost * 1000 # b/c cost in lookup table is in 1000's
    def get_reach(seq):
        return sum([look_up[m]['reach'] for m in seq if m != 0])

    for i, sol in enumerate(pop):
        if get_sol_cost(sol) > max_budget:
            fitnesses[i] = LOW_VAL # flag the over budget val so it does not
        else: # get the reach of solution sequence
            fitnesses[i] = get_reach(sol)

    return fitnesses


def get_parent_tourn(pop, look_up, max_budget=100000, k=2):
    """ tournament method of parent selection, ie randomly pick k then pick the most fit
    parameter k: number of random individuals to pick """
    # get k random inviduals or solution sequences
    rand_indices = numpy.random.choice(pop.shape[0], k)
    random_picks = pop[rand_indices]

    # get the fittest of the rando's
    fitnesses = cal_pop_fitness(random_picks, look_up, max_budget=max_budget)
    max_fit_i = numpy.where(fitnesses == numpy.max(fitnesses))
    max_fit_i = max_fit_i[0][0]

    return random_picks[max_fit_i,:] # return most fit as parent


def get_parent_fit(pop, max_budget=100000):
    """flawed as parents will always be the same"""
    fitnesses = cal_pop_fitness(pop, max_budget)
    max_fit_i = numpy.where(fitnesses == numpy.max(fitnesses))
    max_fit_i = max_fit_i[0][0]


    return pop[max_fit_i, :]  # return most fit as parent


def crossover(parents, p_c = .8):
    n = len(parents[0])
    c = random.randrange(0, n)

    return numpy.append(parents[0, :c], parents[1,c:], axis=0)


def mutation(offspring, gene_pool=[0,1,2,3,4,5,6], pmut=.3):
    # introduce probability of mutation
    if random.uniform(0, 1) >= pmut:
        return offspring

    n = len(offspring)
    g = len(gene_pool)
    c = random.randrange(0, n) # random index of offspring
    r = random.randrange(0, g) # random gene
    new_gene = gene_pool[r]
    plus_gene = numpy.append(offspring[:c], [new_gene], axis=0)
    mutated = numpy.append(plus_gene, offspring[c + 1:], axis=0)
    return mutated