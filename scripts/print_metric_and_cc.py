import cPickle as pickle
import sys
import numpy as np

nmod = sys.argv[1]

metric, barecc = pickle.load(open("metric_and_barecc_"+str(nmod)+".pickle","r"))

print "N_moduli:", nmod
print "Bare CC:"
print barecc
print "Metric:"
print metric
print "Metric for Mathematica:"
print str(metric).replace("[","{").replace("]","}").replace("  ",",").replace("\n ",",")
