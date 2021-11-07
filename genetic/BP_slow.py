#    This file is was created based on a DEAP template of the knapsack implementation

import random

import numpy as np
import sys
import os
import mpmath as mpm
from mpmath import *
import cPickle as pickle

from deap import algorithms
from deap import base
from deap import creator
from deap import tools


# set the numerical accuracy in mpm:
mp.dps = 200
BARE_CC = mpf(-1)

# parameters of the GA algorithm
NGEN = 1000  # total numbers of generations evolved
MU = 50  # number of individuals in the next generation
LAMBDA = 80  # number of children in each generation
CXPB = 0.7  # probability that offspring is produced by mating
MUTPB = 0.25  # probability that offspring is produced by mutation

SEED = 64
GENE_MUTATION_RATE = 0.4  # FR: The probability with which each single gene is mutated
random.seed(SEED)

# parameters of the CC model
NMOD = int(sys.argv[1])
SIGMA = mpf(sys.argv[2])
EPS = mpf(sys.argv[3])
L0 = EPS
METRIC = np.array([[]])  # will be computed later
NGVEC = np.array([np.array([]) for _ in range(MU)])  # keep track of the NG vector of each individual in each generation to speed up the computation
ORIGIN_CC = mpf(0)


def init_metric():
    global NMOD, SIGMA, METRIC, ORIGIN, BARE_CC, ORIGIN_CC, NGVEC
    filename = "./metrics/metric" + str(NMOD) + 'sig' + str(float(SIGMA)) + ".pickle"
    if os.path.isfile(filename):
        METRIC = pickle.load(open(filename, "r"))
    else:  # new metric
        set_random_metric()
        filename = "metrics/metric" + str(NMOD) + 'sig' + str(float(SIGMA)) + ".pickle"
        pickle.dump(METRIC, open(filename, 'w'))

    ###
    # shift origin in largest eigenvector direction
    # don't need mpmath precision here since we'll be rounding anyways.
    # linalg does not support mpf or float128, but we're rounding anyways so ordinary float is sufficient
    eig_vals, eig_vecs = np.linalg.eig(np.array([[np.float(ii) for ii in jj] for jj in METRIC]))
    # eig_vals = eig_vals[0]
    # eig_vecs = eig_vecs[0] # for some reason it's in a list itself, take 0th element
    eig_vecs = np.transpose(eig_vecs)

    # print 'metric',self.metric
    new_eig_vecs = []
    for i in range(len(eig_vecs)):
        new_eig_vecs.append(eig_vecs[i] / eig_vals[i] ** .5)
    # print 'evecs normalized? should be all 1', [np.dot(np.dot(self.metric,ev),ev) for ev in new_eig_vecs]

    rounded_evecs = [[int(round(entry)) for entry in ev] for ev in new_eig_vecs]
    origin_prempm = rounded_evecs[np.argmin(eig_vals)]
    ORIGIN = [mpf(i) for i in origin_prempm]
    ORIGIN_CC = BARE_CC + np.dot(np.dot(METRIC, ORIGIN), ORIGIN)
    NGVEC = np.dot(np.array(ORIGIN), METRIC)


def set_random_metric():
    global NMOD, METRIC, SIGMA
    # pos def metric
    A = np.array([[SIGMA * mpm.sqrt(mpf(2)) * mpm.erfinv(mpf(2) * mpm.rand() - mpf(1)) for _ in range(NMOD)] for _ in range(NMOD)])

    METRIC = np.dot(A, A.transpose())


def dist(cc):
    global EPS
    return log10(abs(cc - EPS))


def evaluate_cc(individual):
    global ORIGIN_CC
    cc = ORIGIN_CC + np.dot(np.dot(METRIC, individual), individual)
    d = dist(cc)
    return -np.float(d),
    # return np.float(1./float(d)),


creator.create("Fitness", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.Fitness)

toolbox = base.Toolbox()

# Attribute generator
toolbox.register("attr_item", random.randrange, -NMOD, NMOD)

# Structure initializers
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_item, NMOD)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def cxSet(ind1, ind2):
    """Apply a crossover operation on input sets. The first child has the first 50% of the genes from parent A and the last 50% from parent B, the second child the other way around."""
    global NMOD
    # need to do substitution in-place
    for i in range(NMOD/2, NMOD):
        tmp = ind1[i]
        ind1[i] = ind2[i]
        ind2[i] = tmp
    return ind1, ind2


def mutSet(individual):
    """Mutation that randomly increases/decreases one of the directions by n where n \in {-5, 5}"""
    global NMOD, GENE_MUTATION_RATE
    for i in range(len(individual)):
        if random.random() < GENE_MUTATION_RATE:
            current = abs(individual[i])
            individual[i] += random.randint(-2*current, 2*current)
    return individual,


toolbox.register("evaluate", evaluate_cc)
toolbox.register("mate", cxSet)
toolbox.register("mutate", mutSet)
toolbox.register("select", tools.selNSGA2)


def main():
    global SEED, NGEN, MU, LAMBDA, CXPB, MUTPB
    random.seed(SEED)

    init_metric()

    pop = toolbox.population(n=MU)
    hof = tools.ParetoFront()
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)
    
    algorithms.eaMuPlusLambda(pop, toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN, stats, halloffame=hof)
    
    return pop, stats, hof


if __name__ == "__main__":
    pop, stats, hof = main()
