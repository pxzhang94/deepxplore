from deap import base
from deap import creator
from deap import tools

import argparse
from scipy.misc import imsave
from utils import *
import random
import numpy as np
from driving_models import *
import os
import time

def initialization(img):
    method = random.randint(1, 3)
    trans_matrix = generate_value(len(img[0]), len(img[0][0]))
    rect_shape = (random.randint(1, 10), random.randint(1, 10))
    start_point = (
        random.randint(0, np.asarray(img).shape[1] - rect_shape[0]),
        random.randint(0, np.asarray(img).shape[2] - rect_shape[1]))

    if method == 1:
        transformation = c_light(trans_matrix)
    elif method == 2:
        transformation = c_occl(trans_matrix, start_point, rect_shape)
    elif method == 3:
        transformation = c_black(trans_matrix, start_point, rect_shape)

    store_path = './genetic_algorithm/initialization/'
    isExists = os.path.exists(store_path)
    if not isExists:
        os.makedirs(store_path)

    imsave(store_path + str(time.time()) + '.png', deprocess_image(img + transformation * stepsize))
    return img + transformation * stepsize

#it is easy to choose each gene randomly between parents
def crossover(ind1, ind2):
    ind1 = ind1[0]
    ind2 = ind2[0]
    gen_img1 = np.zeros_like(ind1)
    gen_img2 = np.zeros_like(ind1)
    for i in range(len(ind1)):
        for j in range(len(ind1[i])):
            for k in range(len(ind1[i][j])):
                for l in range(len(ind1[i][j][k])):
                    choose = random.randint(1, 2)
                    if choose == 1:
                        gen_img1[i][j][k][l] = ind1[i][j][k][l]
                        gen_img2[i][j][k][l] = ind2[i][j][k][l]
                    elif choose == 2:
                        gen_img1[i][j][k][l] = ind2[i][j][k][l]
                        gen_img2[i][j][k][l] = ind1[i][j][k][l]
    return gen_img1, gen_img2

def mutatation(individual, indpb):
    ind = individual[0]
    if random.random() < indpb:
        method = random.randint(1, 3)
        trans_matrix = generate_value(len(ind), len(ind[0]))
        rect_shape = (random.randint(1, 5), random.randint(1, 5)) #mutation area is smaller than initialization
        start_point = (
            random.randint(0, np.asarray(ind).shape[1] - rect_shape[0]),
            random.randint(0, np.asarray(ind).shape[2] - rect_shape[1]))

        if method == 1:
            transformation = c_light(trans_matrix)
        elif method == 2:
            transformation = c_occl(trans_matrix, start_point, rect_shape)
        elif method == 3:
            transformation = c_black(trans_matrix, start_point, rect_shape)
        return ind + transformation * stepsize
    else:
        return ind

def fitness(individual):
    x_differential = distance_x(ori_img, individual[0])
    result1 = model.predict(ori_img)[0]
    result2 = model.predict(individual)[0]
    y_differential = distance_y(result1, result2)
    if x_differential == 0:
        return 0,
    else:
        return y_differential / x_differential,

# parser = argparse.ArgumentParser(
#     description='Main function for genetic algorithm for input generation in Driving dataset')
# parser.add_argument('step', help="step size of gradient descent", type=float)
# args = parser.parse_args()
# stepsize = args.step
stepsize = 10

# input image dimensions
img_rows, img_cols = 100, 100
input_shape = (img_rows, img_cols, 3)

# define input tensor as a placeholder
input_tensor = Input(shape=input_shape)

# load multiple models sharing same input tensor
K.set_learning_phase(0)
model = Dave_orig(input_tensor=input_tensor, load_weights=True)

img_paths = image.list_pictures('./testing/center', ext='jpg')
ori_img = preprocess_image(random.choice(img_paths))

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
# # Attribute generator
toolbox.register("gen_img", initialization, ori_img)
# # Structure initializers
# toolbox.register("individual", tools.initRepeat, creator.Individual,
#     toolbox.gen_img, 100)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.gen_img, 1)
# toolbox.register("individual", initialization, ori_img)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

#----------
# Operator registration
#----------
# register the goal / fitness function
toolbox.register("evaluate", fitness)

# register the crossover operator
toolbox.register("mate", crossover)

# register a mutation operator with a probability to
# flip each attribute/gene of 0.05
toolbox.register("mutate", mutatation, indpb=0.05)

# operator for selecting individuals for breeding the next
# generation: each individual of the current generation
# is replaced by the 'fittest' (best) of three individuals
# drawn randomly from the current generation.
toolbox.register("select", tools.selTournament, tournsize=3)


# def main():
random.seed(64)

# create an initial population of 300 individuals (where
# each individual is a list of integers)
pop = toolbox.population(n=300)

# CXPB  is the probability with which two individuals
#       are crossed
#
# MUTPB is the probability for mutating an individual
CXPB, MUTPB = 0.5, 0.2

print("Start of evolution")

# Evaluate the entire population
fitnesses = list(map(toolbox.evaluate, pop))
for ind, fit in zip(pop, fitnesses):
    ind.fitness.values = fit

print("  Evaluated %i individuals" % len(pop))

# Extracting all the fitnesses of
fits = [ind.fitness.values[0] for ind in pop]

# Variable keeping track of the number of generations
g = 0

# Begin the evolution
# while max(fits) < 100 and g < 1000:
while g < 10:
    # A new generation
    g = g + 1
    print("-- Generation %i --" % g)

    # Select the next generation individuals
    offspring = toolbox.select(pop, len(pop))
    # Clone the selected individuals
    offspring = list(map(toolbox.clone, offspring))

    # Apply crossover and mutation on the offspring
    for child1, child2 in zip(offspring[::2], offspring[1::2]):

        # cross two individuals with probability CXPB
        if random.random() < CXPB:
            toolbox.mate(child1, child2)

            # fitness values of the children
            # must be recalculated later
            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:

        # mutate an individual with probability MUTPB
        if random.random() < MUTPB:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    print("  Evaluated %i individuals" % len(invalid_ind))

    # The population is entirely replaced by the offspring
    pop[:] = offspring

    # Gather all the fitnesses in one list and print the stats
    fits = [ind.fitness.values[0] for ind in pop]

    length = len(pop)
    mean = sum(fits) / length
    sum2 = sum(x * x for x in fits)
    std = abs(sum2 / length - mean ** 2) ** 0.5

    print("  Min %s" % min(fits))
    print("  Max %s" % max(fits))
    print("  Avg %s" % mean)
    print("  Std %s" % std)

print("-- End of (successful) evolution --")

best_ind = tools.selBest(pop, 1)[0]
print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))

store_path = './genetic_algorithm/best/'
isExists = os.path.exists(store_path)
if not isExists:
    os.makedirs(store_path)

imsave(store_path + best_ind.fitness.values + "_" + str(time.time()) + '.png', deprocess_image(best_ind))


# if __name__ == "__main__":
#     main()



