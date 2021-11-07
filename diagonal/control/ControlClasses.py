import uuid
import os
import datetime
import random
import cPickle as pickle
import math
import numpy  as np
import sys
import itertools
#import pandas as pd
#import seaborn as sns
#from matplotlib import pyplot as plt
from mpi4py import MPI
import mpmath as mpm
from mpmath import *

class WishartControlExp:
    def __init__(self,nmod, eps, sigma, steps, power=1, process_idx=1, box_size = None, metric = None, metric_index = "RAND", metric_type = "load", total_steps = None):
        self.nmod, self.eps, self.sigma, self.steps, self.pow, self.process_idx, self.box_size, self.metric, self.metric_index, self.mtype = nmod, eps, sigma, steps, power, process_idx, box_size, metric, metric_index, metric_type
        if self.metric == None: 
            if metric_type == 'rand':
                self.metric = self.random_metric(self.nmod,self.sigma)
            elif metric_type == 'load':
                self.metric = pickle.load(open('metrics/metric' + str(nmod)+'sig' + str(sigma)+'v1.pickle','r'))
        print 'metric loaded, ', self.metric[0][0]
        self.min_pos_cc=1e20
        if total_steps == None: # total_steps is across all MPI workers
            self.total_steps = steps
        else: self.total_steps = total_steps

        if self.box_size == None: # if None, use minimum big enough box
            self.box_size = 0
            while (2*self.box_size+1)**self.nmod < self.steps: self.box_size += 1
            print "Using min box size, is", self.box_size

        self.compute_origin()
        self.perform_experiment_and_output()

    def compute_cc_norm(self,v):
        return np.dot(np.dot(self.metric,v),v)

    def compute_cc(self,v):
        return mpf(-1) + self.compute_cc_norm(v)

    def smallest_pos(self,myl):
        pos = [l for l in myl if l > 0]
        if len(pos) > 0: 
            return min(pos)
        else: 
            return None
        
    #def perform_experiment(self,p=True):
    #    self.p, self.s = {'worker': [], 'time': [], 'cc':  [], 'cc_mpf':  [],'state': []}, {'worker': [], 'time': [], 'cc':  [],'cc_mpf':  [], 'state': []}
        
    #    o_cc = self.compute_cc(self.origin)
    #    self.nonmpfstates, self.ccs = self.cc_sampler(self.steps)
    #    self.states = [[mpf(e) for e in s] for s in self.nonmpfstates]
    #    for i in range(len(self.states)):
    #        state, cc = self.states[i], self.ccs[i]
    #        
    #        if cc > self.eps and cc < 2*self.eps:
    #            self.add_s((None,i,cc,state))
    #        if cc < self.min_pos_cc and cc > 0:
    #            self.min_pos_cc = cc
    #            self.add_p((None,i,cc,state))
        #self.p_df = pd.DataFrame(self.p)
        #self.s_df = pd.DataFrame(self.s)
        #cc_diffs = [abs((cc - o_cc)/o_cc) for cc in ccs]
        #meandiffs, stdevdiffs, smallestpos = np.mean(cc_diffs), np.std(cc_diffs), smallest_pos(ccs)
        #meanccs, stdevccs = np.mean(ccs), np.std(ccs)
        #if p:
        #    print "   origin cc", o_cc
        #    print '   grid search:'
        #    print '      mean diffs:', meandiffs
        #    print '      stdv diffs:', stdevdiffs
        #    print '      smallest pos cc:', smallestpos
        #return {'o_cc': o_cc, 'meanccs': meanccs, 'stdevccs': stdevccs, 'meandiffs': meandiffs, 'stdevdiffs': stdevdiffs, 'smallestpos': smallestpos, 'metric': metric, 'o':o, 'eigval':eigval, 'eigindex':eigindex, 'box_size':box_size}

    def dist(self,cc):
        return abs(cc-self.eps)

    def reward(self,cc):
        d = np.float(self.dist(cc))
        return 1/d**self.pow

    def add_p(self,tup): # add positive
        self.p['worker'].append(tup[0])
        self.p['time'].append(int(tup[1]))
        self.p['cc_mpf'].append(tup[2])
        self.p['cc'].append(float(tup[2]))
        self.p['state'].append([int(l) for l in tup[3]])


    def add_s(self,tup): # add solution
        self.s['worker'].append(tup[0])
        self.s['time'].append(int(tup[1]))
        self.s['cc_mpf'].append(tup[2])
        self.s['cc'].append(float(tup[2]))
        self.s['state'].append([int(l) for l in tup[3]])  


    def compute_origin(self): 
        eig_vals, eig_vecs = np.linalg.eig(np.array([[np.float(ii) for ii in jj] for jj in self.metric]))
        eig_vecs = np.transpose(eig_vecs)
            
        new_eig_vecs = []
        for i in range(len(eig_vecs)):
            new_eig_vecs.append(eig_vecs[i]/eig_vals[i]**.5)
        rounded_evecs = [[int(round(entry)) for entry in ev] for ev in new_eig_vecs]
        self.origin_eigval = np.min(eig_vals)
        self.origin = [mpf(e) for e in rounded_evecs[np.argmin(eig_vals)]]

    def random_metric(self,nmod,sigma): # pos def metric
        A = np.array([[sigma*mpm.sqrt(mpf(2))*mpm.erfinv(mpf(2)*mpm.rand()-mpf(1)) for i in range(nmod)] for j in range(nmod)])
        return np.dot(A,A.transpose())

   # def p_plot(self):
    #    ax = sns.lmplot(x="time",y="cc",data=self.p_df,fit_reg=False)
    #    ax.set_xticklabels(rotation=30)
     #   ax.set(yscale="log")
     #   plt.ylim(min(self.p_df['cc'])/3, max(self.p_df['cc'])*3)
     #   ax2 = plt.gca()
     #   ax2.set_title(self.command_string())

    #def s_plot(self):
    #    ax = sns.lmplot(x="time",y="cc",data=self.s_df,fit_reg=False)
    #    ax.set_xticklabels(rotation=30)
    #    ax.set(yscale="log")
    #    plt.ylim(min(self.s_df['cc'])/3, max(self.s_df['cc'])*3)
    #    ax2 = plt.gca()
    #    ax2.set_title(self.command_string())

    #def scores_plot(self):
    #    ax = sns.lmplot(x="steps",y="mean",data=self.scores_df,fit_reg=False)
    #    ax.set_xticklabels(rotation=30)
    #    ax.set(yscale="log")
     #   ax2 = plt.gca()
     #   ax2.set_title(self.command_string())

    def command_string(self):
        raise NotImplementedError

    def output_experiment(self):
        if not os.path.isdir("output"): os.system("mkdir output")
        outdir = "output/"+self.command_string()+"/"
        if not os.path.isdir(outdir): os.system("mkdir "+outdir)
        output = outdir + "output"+str(self.process_idx)+".txt"
        comm = outdir + "command"+str(self.process_idx)+".txt"
        scs = outdir + "scores"+str(self.process_idx)+".txt"
        
        f = open(output,'w')
        f.write("head: " + str((self.eps,self.nmod,self.sigma,self.metric_index))+"\n")
        f.write("avg reward:" + str(np.mean([self.reward(cc) for cc in self.ccs])))
        for i in range(len(self.p['cc'])):
            global_t, min_pos_cc, state = self.p['time'][i],self.p['cc_mpf'][i],self.p['state'][i]
            f.write("p " + str((self.process_idx,global_t,min_pos_cc,state))+"\n")
        for i in range(len(self.s['cc'])):
            global_t, min_pos_cc, state = self.s['time'][i],self.s['cc_mpf'][i],self.s['state'][i]
            f.write("s " + str((self.process_idx,global_t,cc,state))+"\n")
        f.close()

        f = open(comm,'w')
        f.write("32 --steps "+str(self.steps)+" --gamma CONTROL --eps "+str(self.eps)+" --nmod "+str(self.nmod)+" --sigma "+str(self.sigma)+" --beta CONTROL --reward-d-pow CONTROL --arch CONTROL")
        f.close()
        
        f = open(scs,'w')
        f.write("steps  episodes    elapsed mean    median  stdev   max min average_value   average_entropy\n")
        f.write("ishkuts")
        f.close()

    def output_min_pos_cc(self):
        if not os.path.isdir("output"): os.system("mkdir output")
        outdir = "output/"+self.command_string()+"/"
        if not os.path.isdir(outdir): os.system("mkdir "+outdir)
           
        outfile = outdir + "output.txt"
           
        # update the file
        hnd = open(outfile, 'a+')
        hnd.write("p " + str((self.process_idx,self.counter,self.min_pos_cc,self.state))+"\n")
        hnd.close()

    def output_solution(self,cc):
        if not os.path.isdir("output"): os.system("mkdir output")
        outdir = "output/"+self.command_string()+"/"
        if not os.path.isdir(outdir): os.system("mkdir "+outdir)

        outfile = outdir + "output.txt"
           
        # update the file
        hnd = open(outfile, 'a+')
        hnd.write("s " + str((self.process_idx,self.counter,cc,self.state))+"\n")
        hnd.close()

class WishartControlExpGrid(WishartControlExp):

    #def cc_sampler(self,steps): # this is a simple grid-search
    #    nmod = len(self.metric)
    #    states, ccs = [], []
    #    if (2*self.box_size+1)**self.nmod < self.steps: raise Exception("Not enough steps in grid search!")
    #    count = 0
    #    for v in itertools.product(*[range(-self.box_size,self.box_size+1) for k in range(self.nmod)]):
    #        p = np.array(self.origin) + np.array(v)
    #        states.append(v) # append state without origin
    #        ccs.append(self.compute_cc(p))
    #        count += 1
    #        if count == self.steps: break
    #    return (states, ccs)


    def perform_experiment(self): # this is a simple grid-search
        self.p, self.s = {'worker': [], 'time': [], 'cc':  [], 'cc_mpf':  [],'state': []}, {'worker': [], 'time': [], 'cc':  [],'cc_mpf':  [], 'state': []}
        nmod = len(self.metric)
        states, ccs = [], []
        if (2*self.box_size+1)**self.nmod < self.steps: raise Exception("Not enough steps in grid search!")
        self.ccs = []
        counter = 0
        for v in itertools.product(*[range(-self.box_size,self.box_size+1) for k in range(self.nmod)]):
            p = np.array(self.origin) + np.array([mpf(ii) for ii in v])
            s = v
            cc = abs(self.compute_cc(p))
            self.ccs.append(np.float(cc))
            if cc > self.eps and cc < 2*self.eps:
                self.add_s((None,counter,cc,[mpf(e) for e in s]))
            if cc < self.min_pos_cc and cc > 0:
                self.min_pos_cc = cc
                self.add_p((None,counter,cc,[mpf(e) for e in s]))
            counter += 1
            if counter == self.steps: break

    def command_string(self):
        s = "gridcontrol"
        s += "n" + str(self.nmod)
        s += "s" + str(self.steps)
        s += "p" + str(self.pow)
        s += "e" + str(self.eps)
        s += "sig" + str(self.sigma)
        s += "box" + str(self.box_size)
        s += "mtype" + str(self.mtype)
        return s


     

class WishartControlExpRand(WishartControlExp):

   # def cc_sampler(self,steps): # this is a simple grid-search
   #     nmod = len(self.metric)
   #     states, ccs = [], []
   #     if (2*self.box_size+1)**self.nmod < self.steps: raise Exception("Not enough steps in grid search!")
   #     
   #     states = []
    #    while len(states) < steps:
    #        states.append([random.randint(-self.box_size,self.box_size) for i in range(self.nmod)])
    #    ccs = [abs(self.compute_cc(np.array(self.origin)+np.array(s))) for s in states]
#
#        return (states, ccs)

    def perform_experiment(self): # this is a simple grid-search
        self.p, self.s = {'worker': [], 'time': [], 'cc':  [], 'cc_mpf':  [],'state': []}, {'worker': [], 'time': [], 'cc':  [],'cc_mpf':  [], 'state': []}
        nmod = len(self.metric)
        states, ccs = [], []
        if (2*self.box_size+1)**self.nmod < self.steps: raise Exception("Not enough steps in grid search!")
        self.ccs = []
        counter = 0
        while counter < self.steps:
            s = [random.randint(-self.box_size,self.box_size) for i in range(self.nmod)]
            cc = abs(self.compute_cc(np.array(self.origin)+np.array([mpf(ii) for ii in s])))
            self.ccs.append(np.float(cc))
            counter += 1
            if cc > self.eps and cc < 2*self.eps:
                self.add_s((None,counter,cc,[mpf(e) for e in s]))
            if cc < self.min_pos_cc and cc > 0:
                self.min_pos_cc = cc
                self.add_p((None,counter,cc,[mpf(e) for e in s]))

    def perform_experiment_and_output(self): # this is a simple grid-search
        self.p, self.s = {'worker': [], 'time': [], 'cc':  [], 'cc_mpf':  [],'state': []}, {'worker': [], 'time': [], 'cc':  [],'cc_mpf':  [], 'state': []}
        nmod = len(self.metric)
        states, ccs = [], []
        if (2*self.box_size+1)**self.nmod < self.steps: raise Exception("Not enough steps in grid search!")
        self.ccs = []
        self.counter = 0
        while self.counter < self.steps:
            s = [random.randint(-self.box_size,self.box_size) for i in range(self.nmod)]
            self.state = s
            cc = abs(self.compute_cc(np.array(self.origin)+np.array([mpf(ii) for ii in s])))
            self.ccs.append(np.float(cc))
            self.counter += 1
            if cc > self.eps and cc < 2*self.eps:
                self.output_solution(cc)
            if cc < self.min_pos_cc and cc > 0:
                self.min_pos_cc = cc
                self.output_min_pos_cc()
            if self.counter == self.steps: break

    def command_string(self):
        s = "randcontrol"
        s += "n" + str(self.nmod)
        s += "s" + str(self.total_steps)
        s += "p" + str(self.pow)
        s += "e" + str(self.eps)
        s += "sig" + str(self.sigma)
        s += "box" + str(self.box_size)
        s += "mtype" + str(self.mtype)
        return s

class WishartControlExpEig(WishartControlExp):

    def cc_sampler(self,steps): # this is a simple grid-search
        eig_vals, eig_vecs = np.linalg.eig(self.metric)
        eig_vecs = np.transpose(eig_vecs)
        eig_vec = eig_vecs[np.argmin(eig_vals)]
        big_eig_vec = eig_vec/np.min(eig_vals)**.5

        print 'norm squared of cc sampler eigvec is ', np.dot(eig_vec,eig_vec)

        states, ccs = [], []

        r = self.steps
        if r%2 == 1: r += 1

        for k in range(-r/2,r/2+1):
            v = k*eig_vec
            states.append(list(v))
        ccs = [abs(self.compute_cc([round(e) for e in big_eig_vec+np.array(s)])) for s in states]

        return (states, ccs)

    def command_string(self):
        s = "eigcontrol"
        s += "n" + str(self.nmod)
        s += "s" + str(self.steps)
        s += "p" + str(self.pow)
        s += "e" + str(self.eps)
        s += "sig" + str(self.sigma)
        s += "box" + str(self.box_size)
        s += "mtype" + str(self.mtype)
        return s

if __name__ == "__main__":
    ##set the numerical accuracy in mpm:
    mp.dps = 200

    from mpi4py import MPI
    import sys

    comm = MPI.COMM_WORLD
    size, rank = int(comm.Get_size()), int(comm.Get_rank())
    numsamples = int(sys.argv[1])

    print "\n\nPROCESS #" + str(rank)
    print "size, rank:", size, rank
    print "numsamples:", numsamples
    comm = MPI.COMM_WORLD
    size, rank = int(comm.Get_size()), int(comm.Get_rank())
    #numsamples = int(sys.argv[1])

    numtogen = None
    if rank < size - 1:
        numtogen = int(math.floor(1.0*numsamples/size))
    elif rank == size-1:
        numtogen = int(numsamples - (size-1)*math.floor(1.0*numsamples/size))

    print "numtogen, rank:", numtogen, rank

    nmod, eps, sigma,  = int(sys.argv[3]), mpf(sys.argv[4]), mpf(sys.argv[5]) 
    mtype = str(sys.argv[6])
    #print nmod, eps, sigma, 

    if sys.argv[2] == "grid":
        print "GRID MAY NOT BE SENSIBLE, YOU GOOBER."
        #k = WishartControlExpGrid(nmod, eps, sigma, numtogen, process_idx=rank, metric_type = mtype, process_idx = rank)
        #k.output_experiment()
    elif sys.argv[2] == "rand":
        k = WishartControlExpRand(nmod, eps, sigma, numtogen, process_idx=rank, metric_type = mtype,total_steps = numsamples)
        #k.perform_experiment_and_output()