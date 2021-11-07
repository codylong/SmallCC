#    This file is was created based on a DEAP template of the knapsack implementation

import random

import numpy as np
import sys
import os
import mpmath as mpm
from mpmath import *
import cPickle as pickle
import scipy.optimize as sp
import itertools as iter

from deap import algorithms
from deap import base
from deap import creator
from deap import tools


# set the numerical accuracy in mpm:
mp.dps = 200
BARE_CC = mpf(-1.)

# parameters of the GA algorithm
NGEN = 100000  # total numbers of generations evolved
MU = 100  # number of individuals in the next generation
LAMBDA = 10  # number of children in each generation
CXPB = 0.7  # probability that offspring is produced by mating
MUTPB = 0.3  # probability that offspring is produced by mutation
OUTPUT_FREQ = 1000


SEED = 64
GENE_MUTATION_RATE = 0.5  # FR: The probability with which each single gene is mutated
random.seed(SEED)

# parameters of the CC model
NMOD = int(sys.argv[1])
SIGMA = mpf(sys.argv[2])
EPS = mpf(sys.argv[3])
L0 = EPS
METRIC = np.array([[]])  # will be computed later
NP_METRIC = np.array([[]])  # will be computed later
ORIGIN_CC = mpf(0)


def init_metric():
    global NMOD, SIGMA, METRIC, ORIGIN, BARE_CC, ORIGIN_CC, NP_METRIC
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
    start_point = random.randrange(0, len(eig_vals))
    origin_prempm = rounded_evecs[start_point]
    # origin_prempm = rounded_evecs[np.argmin(eig_vals)]

    # ORIGIN = [origin_prempm[i] + START_POINT[i] for i in range(len(origin_prempm))]
    ORIGIN = origin_prempm
    ORIGIN = [0 for _ in range(NMOD)]

    # convert metric to mpmath matrices
    # ORIGIN = mpm.matrix([mpf(i) for i in origin_prempm])
    print ORIGIN
    NP_METRIC = [[float(METRIC[i][j]) for i in range(NMOD)] for j in range(NMOD)]
    METRIC = mpm.matrix([[mpf(ii) for ii in jj] for jj in METRIC])
    ORIGIN_CC = BARE_CC + (mpm.matrix(ORIGIN).T * METRIC * mpm.matrix(ORIGIN))[0]
    print ORIGIN_CC,"\n\n"

    # for o in rounded_evecs:
    #    print float(BARE_CC + (mpm.matrix(o).T * METRIC * mpm.matrix(o))[0])

    # return


def set_random_metric():
    global NMOD, METRIC, SIGMA
    # pos def metric
    A = np.array([[SIGMA * mpm.sqrt(mpf(2)) * mpm.erfinv(mpf(2) * mpm.rand() - mpf(1)) for _ in range(NMOD)] for _ in range(NMOD)])

    METRIC = np.dot(A, A.transpose())


def dist(cc):
    global EPS
    return log10(abs(cc - EPS))


def evaluate_cc(individual):
    global ORIGIN, BARE_CC, METRIC
    total_vec = mpm.matrix([individual[i] + ORIGIN[i] for i in range(NMOD)])
    cc = BARE_CC + (total_vec.T * METRIC * total_vec)[0]
    d = dist(cc)
    return -np.float(d),
    # return np.float(1./float(d)),


def find_closest(start):
    # Try minimizing over non-integers and round the result:
    def f(x):
        global NMOD, BARE_CC, NP_METRIC, ORIGIN
        return float(BARE_CC) + np.dot(np.dot(NP_METRIC, np.array(x) + np.array(ORIGIN)), np.array(x) + np.array(ORIGIN))

    # Subject to the constraint that we get a pos CC:
    def constr(x):
        global NMOD, BARE_CC, NP_METRIC, ORIGIN
        return float(BARE_CC) + np.dot(np.dot(NP_METRIC, np.array(x) + np.array(ORIGIN)), np.array(x) + np.array(ORIGIN))

    tmp_solution = sp.fmin_cobyla(f, start, constr, maxfun=100000)
    # return [int(round(t)) for t in tmp_solution]
    all_combis = iter.product([True, False], repeat=NMOD)
    record = 1
    record_comb = []
    for comb in all_combis:
        current = []
        for i in range(len(comb)):
            if comb[i]:
                current.append(int(ceil(tmp_solution[i])))
            else:
                current.append(int(floor(tmp_solution[i])))
        res = float(BARE_CC) + np.dot(np.dot(NP_METRIC, np.array(current) + np.array(ORIGIN)), np.array(current) + np.array(ORIGIN))
        # if res > 0 and res < record:
        if abs(res) < abs(record):
            record = res
            record_comb = current

    return record_comb

creator.create("Fitness", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.Fitness)

toolbox = base.Toolbox()

# Attribute generator
toolbox.register("attr_item", random.randrange, -5, 5)

# Structure initializers
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_item, NMOD)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def cxSet(ind1, ind2):
    """Apply a crossover operation on input sets. The first child has the first 50% of the genes from parent A and the last 50% from parent B, the second child the other way around."""
    global NMOD, ORIGIN
    x_pos = random.randint(0,NMOD-1)
    for i in range(NMOD):
        new1 = list(ind1)
        new2 = list(ind2)
        tmp1 = list(ind1)
        if i < x_pos:
            new1[i] = new2[i]
        else:
            new2[i] = tmp1[i]

    # now we use the crossed-over ones to find the closest lattice point
    closest1 = find_closest(new1)
    closest2 = find_closest(new2)

    # substitute in-place
    for i in range(NMOD):
        ind1[i] = closest1[i]
        ind2[i] = closest2[i]

    return ind1, ind2

def cxSet3(ind1, ind2):
    """Apply a crossover operation on input sets. The first child has the first 50% of the genes from parent A and the last 50% from parent B, the second child the other way around."""
    global NMOD
    # need to do substitution in-place
    for i in range(NMOD):
        if random.random() < 0.4:
            ind1[i] = ind2[i]
        elif random.random() > 0.6:
            ind2[i] = ind1[i]
        else:
            ind1[i] = ind1[i] + (-1)**random.randint(0, 1)
            ind2[i] = ind2[i] + (-1)**random.randint(0, 1)
    return ind1, ind2

def cxSet2(ind1, ind2):
    """Apply a crossover operation on input sets. The first child has the first 50% of the genes from parent A and the last 50% from parent B, the second child the other way around."""
    global NMOD
    # need to do substitution in-place
    for i in range(NMOD):
        if random.random() < 0.4:
            ind1[i] = ind2[i]
        elif random.random() > 0.6:
            ind2[i] = ind1[i]
    return ind1, ind2

def cxSet3(ind1, ind2):
    """Apply a crossover operation on input sets. The first child has the first 50% of the genes from parent A and the last 50% from parent B, the second child the other way around."""
    global NMOD
    # need to do substitution in-place
    x_pos = random.randint(0,NMOD-1)
    for i in range(NMOD):
        tmp1 = list(ind1)
        if i < x_pos:
            ind1[i] = ind2[i]
        else:
            ind2[i] = tmp1[i]
    return ind1, ind2

def cxSet0(ind1, ind2):
    """Apply a crossover operation on input sets. The first child has the first 50% of the genes from parent A and the last 50% from parent B, the second child the other way around."""
    global NMOD
    # need to do substitution in-place
    x_pos = random.randint(0,NMOD-1)
    tmp1=list(ind1)
    tmp2=list(ind2)
    for i in range(NMOD):
        if ind1.fitness > ind2.fitness:
            if i < x_pos:
                ind2[i] = ind1[i]
        else:
            if i < x_pos:
                ind1[i] = ind2[i]
    return ind1, ind2


def mutSet(individual):
    """Mutation that randomly increases/decreases one of the directions by n where n \in {-5, 5}"""
    global NMOD, GENE_MUTATION_RATE, ORIGIN
    for i in range(len(individual)):
        if random.random() < GENE_MUTATION_RATE:
            # cur = max(NMOD, individual[i])
            # individual[i] = random.randint(-cur, cur)
            individual[i] = -individual[i]
    return individual,


toolbox.register("evaluate", evaluate_cc)
toolbox.register("mate", cxSet)
toolbox.register("mutate", mutSet)
toolbox.register("select", tools.selNSGA2)


def main():
    global SEED, NGEN, MU, LAMBDA, CXPB, MUTPB, NMOD, OUTPUT_FREQ
    random.seed(SEED)

    init_metric()

    pop = toolbox.population(n=MU)
    hof = tools.HallOfFame(NMOD)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)
    
    algorithms.eaMuPlusLambdaFR(pop, toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN, stats, halloffame=hof, report_freq=OUTPUT_FREQ)
    
    return pop, stats, hof


if __name__ == "__main__":
    pop, stats, hof = main()
