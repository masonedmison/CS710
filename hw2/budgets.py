"""driver for budget optimization"""
import numpy as np
from hw2.data.advertising_dicts import mag_look_up, ads_look_up
import hw2.genetic as GA

# size and gens
POP = 200
NGENS = 200
INDIVIDUAL_LEN = 5 # based off of the most we could spend, eg 5 * cost of magazine 1
####
# data specific
MAX_BUDG = 5000000
LOOK_UP = ads_look_up # set advertising data here from imports above
OPTS = len(ads_look_up.keys())
####

# shape of population
pop_shape = (POP, INDIVIDUAL_LEN)
# initialize population where len(individual) is 5
init_pop = np.random.randint(OPTS+1, size=pop_shape)
# sprinkle in some zeros so the majority of solution aren't over budget
for i,r in enumerate(init_pop):
    rand_i =np.random.randint(INDIVIDUAL_LEN, size=(3,))
    r[rand_i] = 0
####

# best outputs to record the number of features with best fitness score
best_outputs = dict()
best_out_seq = []
new_population = np.zeros(pop_shape)
for generation in range(NGENS):
    num_of_matings = 0 # matings per population - also tells us index of children being addded

    print("Generation : ", generation)
    while num_of_matings < POP:
        # Measuring the fitness of each chromosome in the population.
        parents = np.asarray([GA.get_parent_tourn(init_pop, LOOK_UP, max_budget=MAX_BUDG) for i in range(2)])

        # Generating next generation using crossover.
        offspring_crossover = GA.crossover(parents)
        # Adding some variations to the offspring using mutation.
        offspring_mutation = GA.mutation(offspring_crossover)


        new_population[num_of_matings,:] = offspring_mutation # use num_of_matings as index
        num_of_matings += 1

    # get most fit individual at end of generation
    fitnesses = GA.cal_pop_fitness(new_population, LOOK_UP, max_budget=MAX_BUDG)

    max_fit = np.max(fitnesses)

    max_fit_i = np.where(fitnesses == max_fit)
    max_fit_i = max_fit_i[0][0]

    best_outputs[max_fit] = list(new_population[max_fit_i,:])
    best_out_seq.append(max_fit) # to see progression of iters
    init_population = new_population # point old_population to new_population as this is where we will look for parents for next gen

# find best solution stored in best_outputs
best_fitnesses = list(best_outputs.keys())
best_fitness = max(best_fitnesses) # key that points to solution and possibly other things
best_solution = best_outputs[best_fitness]
# best_solution_indices = np.where(best_solution != 0)[0]


print('[SOLUTION]')
print("best_solution : ", best_solution)
print("Best solution fitness : ", best_fitness)

print('best_outputs', best_outputs)





