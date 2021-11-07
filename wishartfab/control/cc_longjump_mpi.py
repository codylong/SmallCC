import uuid
import os
import datetime
import random
import cPickle as pickle
import math
import numpy  as np
import sys
import itertools
from mpi4py import MPI

comm = MPI.COMM_WORLD
size, rank = int(comm.Get_size()), int(comm.Get_rank())
#numsamples = int(sys.argv[1])

print "\n\nPROCESS #" + str(rank)
print "size, rank:", size, rank

def compute_cc_norm(metric,v):
    return np.dot(np.dot(metric,v),v)

def compute_cc(metric,v):
    return -1 + compute_cc_norm(metric,v)

def smallest_pos(myl):
    pos = [l for l in myl if l > 0]
    if len(pos) > 0: 
        return min(pos)
    else: 
        return None
    
def origin_analysis(metric,o,eigval=None,eigindex=None,box_size=1,p=True):
    if eigval == None:
        if p: print '\norigin analysis', o
    else:
        if p: print "\norigin analysis",o,"e-val",eigval

    o_cc = compute_cc(metric,o)
    ccs = [abs(cc) for cc in cc_grid_search(metric,o,box_size)]
    cc_diffs = [abs(cc - o_cc) for cc in ccs]
    meandiffs, stdevdiffs, smallestpos = np.mean(cc_diffs), np.std(cc_diffs), smallest_pos(ccs)
    meanccs, stdevccs = np.mean(ccs), np.std(ccs)
    if p:
        print "   origin cc", o_cc
        print '   grid search:'
        print '      mean diffs:', meandiffs
        print '      stdv diffs:', stdevdiffs
        print '      smallest pos cc:', smallestpos
    return {'o_cc': o_cc, 'meanccs': meanccs, 'stdevccs': stdevccs, 'meandiffs': meandiffs, 'stdevdiffs': stdevdiffs, 'smallestpos': smallestpos, 'metric': metric, 'o':o, 'eigval':eigval, 'eigindex':eigindex, 'box_size':box_size}

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
        #new_eig_vecs.append(eig_vecs[i]/eig_vals[i]**.5)
        new_eig_vecs.append(eig_vecs[i])
    #print 'evecs normalized? should be all 1', [compute_cc_norm(metric,ev) for ev in new_eig_vecs]
    rounded_evecs = [[int(round(entry)) for entry in ev] for ev in new_eig_vecs]
    #print 'new eig vecs', new_eig_vecs

    cc_norm_units = [[0 for i in range(nmod)] for j in range(nmod)]
    for i in range(nmod): 
        cc_norm_units[i][i] = 1/metric[i][i]**.5
    #print 'cc_norm_units normalized? should be all 1', [compute_cc_norm(metric,v) for v in cc_norm_units]
    rounded_cc_norm_units = [[int(round(entry)) for entry in v] for v in cc_norm_units]
    
    #print len(rounded_evecs), len(rounded_cc_norm_units)

    o_as = []
    sort_evals = list(eig_vals)
    sort_evals.sort()
    for i in range(len(rounded_evecs)):
        o_as.append(origin_analysis(metric,rounded_evecs[i],eigval=eig_vals[i],eigindex=sort_evals.index(eig_vals[i]),p=False))

    for origin in rounded_cc_norm_units:
        o_as.append(origin_analysis(metric,origin,p=False))

    return o_as

def random_metric(nmod,sigma): # pos def metric
    A = np.random.normal(size=(nmod,nmod), scale = sigma)
    return np.dot(A,A.transpose())

nmod, sigma, num_analyses = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
ls_as = []
for i in range(1,num_analyses+1):
    ls_as.append(large_step_analysis(nmod,sigma))

filename = 'pickles/k'+str(nmod)+'s'+str(sigma)+'numanalyses'+str(num_analyses)+"rank"+str(rank)+".pickle"
pickle.dump(ls_as,open(filename,'w'))

#concat = []
#for i in range(len(ls_as)):
#    for j in range(len(ls_as[i])): 
#        concat.append(ls_as[i][j])
#
#df = pd.DataFrame(concat)

# some of these are better of logged
#df['log10eigval']= np.log10(df['eigval'])
#df['log10smallestpos']= np.log10(df['smallestpos'])
##df['log10o_cc']= np.log10(df['o_cc'])
#df['log10meandiffs']= np.log10(df['meandiffs'])
#df2 = df.drop(columns=['box_size','eigval','smallestpos','meandiffs'])

#df2.plot(kind='box', subplots=True, layout=(6,6), sharex=False, sharey=False)
#plt.show()

#df2.hist(bins=50)
#plt.show()

#scatter_matrix(df2)
#plt.show()

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
