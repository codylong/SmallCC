from scipy.optimize import minimize
from sympy import Matrix as sp_matrix
import uuid
import os
import datetime
from inspect import getsourcefile
import errno
from scipy import linalg
import random
from scipy import random as scipyrandom
import cPickle as pickle
import math
import numpy  as np
import sys
import itertools
import pandas as pd
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt

def compute_cc_norm(metric,v):
    return np.dot(np.dot(metric,v),v)

def compute_cc(metric,v):
    return -1 + compute_cc_norm(metric,v)

def smallest_pos(myl):
    return min([l for l in myl if l > 0])

def origin_analysis(metric,o,eigval=None,box_size=1,p=True):
    if eigval == None:
        if p: print '\norigin analysis', o
    else:
        if p: print "\norigin analysis",o,"e-val",eigval

    o_cc = compute_cc(metric,o)
    ccs = cc_grid_search(metric,o,box_size)
    cc_diffs = [cc - o_cc for cc in ccs]
    meandiffs, stdevdiffs, smallestpos = np.mean(cc_diffs), np.std(cc_diffs), smallest_pos(ccs)
   
    if p:
        print "   origin cc", o_cc
        print '   grid search:'
        print '      mean diffs:', meandiffs
        print '      stdv diffs:', stdevdiffs
        print '      smallest pos cc:', smallestpos
    return {'o_cc': o_cc, 'ccs': ccs, 'meandiffs': meandiffs, 'stdevdiffs': stdevdiffs, 'smallestpos': smallestpos, 'metric': metric, 'o':o, 'eigval':eigval, 'box_size':box_size}

def cc_grid_search(metric,o,box_size):
    nmod = len(metric)
    ccs = []
    for v in itertools.product(*[range(-box_size,box_size+1) for k in range(nmod)]):
        p = np.array(o) + np.array(v)
        ccs.append(compute_cc(metric,p))
    return ccs

def large_step_analysis(nmod,sigma): 
    metric = random_metric(nmod,sigma)
    eig_vals, eig_vecs = np.linalg.eig(metric)
    eig_vecs = np.transpose(eig_vecs)
        
    new_eig_vecs = []
    for i in range(len(eig_vecs)):
        new_eig_vecs.append(eig_vecs[i]/eig_vals[i]**.5)
    #print 'evecs normalized? should be all 1', [compute_cc_norm(metric,ev) for ev in new_eig_vecs]
    rounded_evecs = [[int(round(entry)) for entry in ev] for ev in new_eig_vecs]
    
    cc_norm_units = [[0 for i in range(nmod)] for j in range(nmod)]
    for i in range(nmod): 
        cc_norm_units[i][i] = 1/metric[i][i]**.5
    #print 'cc_norm_units normalized? should be all 1', [compute_cc_norm(metric,v) for v in cc_norm_units]
    rounded_cc_norm_units = [[int(round(entry)) for entry in v] for v in cc_norm_units]
    
    print len(rounded_evecs), len(rounded_cc_norm_units)

    o_as = []
    for i in range(len(rounded_evecs)):
        o_as.append(origin_analysis(metric,rounded_evecs[i],eig_vals[i],p=False))

    for origin in rounded_cc_norm_units:
        o_as.append(origin_analysis(metric,origin,p=False))

    return o_as

def random_metric(nmod,sigma): # pos def metric
    A = np.random.normal(size=(nmod,nmod), scale = sigma)
    return np.dot(A,A.transpose())

#nmod, sigma = int(sys.argv[1]), float(sys.argv[2])
nmod,sigma = 10,1e-10
num_analyses = 25
ls_as = []
for i in range(1,num_analyses+1):
    ls_as.append(large_step_analysis(nmod,sigma))

concat = []
for i in range(len(ls_as)):
    for j in range(len(ls_as[i])): 
        concat.append(ls_as[i][j])

df = pd.DataFrame(concat)

# some of these are better of logged
df['log10eigval']= np.log10(df['eigval'])
df['log10smallestpos']= np.log10(df['smallestpos'])
df['log10o_cc']= np.log10(df['o_cc'])
df['log10meandiffs']= np.log10(df['meandiffs'])
df2 = df.drop(columns=['box_size','eigval','smallestpos','o_cc','meandiffs'])

#df2.plot(kind='box', subplots=True, layout=(6,6), sharex=False, sharey=False)
#plt.show()

df2.hist(bins=50)
plt.show()

scatter_matrix(df2)
plt.show()

# def output_min_pos_cc(self):
#     # create path to file if necessary (it shouldn't be, the path should've been created by the training program
#     if not os.path.exists(os.path.dirname(self._outputFilePath)):
#         try:
#             os.makedirs(os.path.dirname(self._outputFilePath))
#         except OSError as exc: # Guard against race condition
#             if exc.errno != errno.EEXIST:
#                 raise
       
#     # update the file
#     hnd = open(self._outputFilePath, 'a+')
#     hnd.write("p " + str((self.process_idx,self.global_t,self.min_pos_cc,self.state))+"\n")
#     hnd.close()

# def output_max_neg_cc(self):
#     # create path to file if necessary (it shouldn't be, the path should've been created by the training program
#     if not os.path.exists(os.path.dirname(self._outputFilePath)):
#         try:
#             os.makedirs(os.path.dirname(self._outputFilePath))
#         except OSError as exc: # Guard against race condition
#             if exc.errno != errno.EEXIST:
#                 raise
       
#     # update the file
#     hnd = open(self._outputFilePath, 'a+')
#     hnd.write("n " + str((self.process_idx,self.global_t,self.max_neg_cc))+"\n")
#     hnd.close()

# def output_solution(self,cc):
#     # create path to file if necessary (it shouldn't be, the path should've been created by the training program
#     if not os.path.exists(os.path.dirname(self._outputFilePath)):
#         try:
#             os.makedirs(os.path.dirname(self._outputFilePath))
#         except OSError as exc: # Guard against race condition
#             if exc.errno != errno.EEXIST:
#                 raise
       
#     # update the file
#     hnd = open(self._outputFilePath, 'a+')
#     hnd.write("s " + str((self.process_idx,self.global_t,cc,self.state))+"\n")
#     hnd.close()
