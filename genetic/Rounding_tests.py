import random

import numpy as np
import sys
import os
import mpmath as mpm
from mpmath import *
import cPickle as pickle
import scipy.optimize as sp
import itertools as iter


# parameters of the CC model
NMOD = int(sys.argv[1])
SIGMA = mpf(sys.argv[2])
EPS = mpf(sys.argv[3])

# set the numerical accuracy in mpm:
mp.dps = 200
BARE_CC = mpf(-1.)
METRIC = [[]]
ORIGIN = []
EIGVALS = []


def init_metric():
    global NMOD, SIGMA, METRIC, ORIGIN, BARE_CC, EIGVALS
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
    EIGVALS = mp.eigsy(mp.matrix(METRIC), eigvals_only=True)


def set_random_metric():
    global NMOD, METRIC, SIGMA
    # pos def metric
    A = np.array([[SIGMA * mpm.sqrt(mpf(2)) * mpm.erfinv(mpf(2) * mpm.rand() - mpf(1)) for _ in range(NMOD)] for _ in range(NMOD)])

    METRIC = np.dot(A, A.transpose())


def main():
    global EIGVALS, EPS, NMOD, BARE_CC

    init_metric()

    # Try minimizing over non-integers and round the result:
    def f(x):
        global NMOD, BARE_CC
        return float(BARE_CC) + sum([x[i] * x[i] * float(EIGVALS[i]) for i in range(NMOD)])

    # Subject to the constraint that we get a pos CC:
    def constr(x):
        global NMOD, BARE_CC
        return float(BARE_CC) + sum([x[i] * x[i] * float(EIGVALS[i]) for i in range(NMOD)])

    for tries in range(100):
        start = [random.uniform(-5000, 5000) for _ in range(NMOD)]
        tmp_solution = sp.fmin_cobyla(f, start, constr, maxfun=100000)

        # start = [0 for _ in range(NMOD)]
        # start[tries] = sqrt(1./float(EIGVALS[tries]))
        # tmp_solution = start

        # print [round(t) for t in tmp_solution]
        # print BARE_CC + sum([round(tmp_solution[i]) * round(tmp_solution[i]) * EIGVALS[i] for i in range(NMOD)])
        # print BARE_CC + sum([floor(tmp_solution[i]) * round(tmp_solution[i]) * EIGVALS[i] for i in range(NMOD)])
        # print BARE_CC + sum([ceil(tmp_solution[i]) * round(tmp_solution[i]) * EIGVALS[i] for i in range(NMOD)])

        all_combis = iter.product([True, False], repeat=NMOD)
        record = 1
        record_comb = []
        for comb in all_combis:
            current = []
            for i in range(len(comb)):
                if comb[i]: current.append(ceil(tmp_solution[i]))
                else: current.append(floor(tmp_solution[i]))
            res = BARE_CC + sum([current[i] * current[i] * EIGVALS[i] for i in range(NMOD)])
            if res > 0 and res < record:
                record = res
                record_comb = current

        print float(record), record_comb


if __name__ == "__main__":
    main()
