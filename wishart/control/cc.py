import gym
from gym import error, spaces, utils
from gym.utils import seeding
from fractions import gcd
from itertools import permutations
import numpy as np
from scipy.optimize import minimize
from sympy import Matrix as sp_matrix
import random
import cPickle as pickle
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

class CC(gym.Env):

    ########
    # RL related methods
    
    def __init__(self):
        #self.nmod = 10
        #self.sigma = 1e-3
        #self.eps = 1e-3
        self.barecc = -1
        self.action_space = None
        self.observation_space = None
        self.metric_index = None
        self.BP_solved_factor=1e10
        #self._outputFilePath = os.path.split(os.path.abspath(getsourcefile(lambda:0)))[0] + "/../output/Pickle_" + datetime.datetime.now().strftime("%Y%m%dT%H%M%S") + "/outputOvercountsRun_"+ str(uuid.uuid4()) + ".pickle"
        self.min_pos_cc = 1e6
        self.max_neg_cc = -1e6
        self.init_cc_printed = False

    def second_init(self): # after nmod, sigma, eps are set, can run this
        filename = "metrics/metric"+str(self.nmod)+'sig' + str(self.sigma) +"v"+str(self.metric_index)+".pickle"
        if os.path.isfile(filename):
            self.metric = pickle.load(open(filename,"r"))
        else: # new metric
            self.metric = self.random_metric()
            existing_metric_files = [f for f in os.listdir("metrics") if "metric"+str(self.nmod)+"v" in f]
            metric_versions = [int(f[f.index('v')+1:f.index('.')]) for f in existing_metric_files]
            print "Existing metric versions are: ", metric_versions
            if metric_versions == []:
                self.metric_index = 1
            else: self.metric_index = max(metric_versions)+1
            filename = "metrics/metric"+str(self.nmod)+ 'sig' + str(self.sigma) +"v"+str(self.metric_index)+".pickle"
            pickle.dump(self.metric,open(filename,'w'))
            self.metric = pickle.load(open(filename,"r"))

        ### 
        # shift origin in largest eigenvector direction
        eig_vals, eig_vecs = np.linalg.eig(self.metric)
        #weird behavior when we run locally.
	#eig_vals = eig_vals[0] # for some reason it's in a list itself, take 0th element
        #eig_vecs = eig_vecs[0] # for some reason it's in a list itself, take 0th element
        eig_vecs = np.transpose(eig_vecs)
        
        print 'metric',self.metric
        new_eig_vecs = []
	#get the normalized (length-1) eigenvectors
        for i in range(len(eig_vecs)):
            new_eig_vecs.append(eig_vecs[i]/eig_vals[i]**.5)
        print 'evecs normalized? should be all 1', [np.dot(np.dot(self.metric,ev),ev) for ev in new_eig_vecs]
        rounded_evecs = [[int(round(entry)) for entry in ev] for ev in new_eig_vecs]
	#step out along the largest eigenvector direction a length of 1. this is
	#arguably where the most possible small cc points live
        self.origin = rounded_evecs[np.argmax(eig_vals)]

        ##
        # set up act, obs spaces
        self.action_space = spaces.Discrete(2*self.nmod)
        self.observation_space = spaces.Discrete(self.nmod)
        self.state = [0 for i in range(self.nmod)]
        print "Worker has (nmod,sigma,eps) = ", str((self.nmod,self.sigma,self.eps))

    def step(self,action):
        done = False
        idx, sign = (action-action%2)/2, (-1)**(action%2)
        self.state[idx] += sign
	#we started the search with the origin shifted out to the edge of the ellipsoid
	#so we need to shift the state by the same amount
        shifted_state = np.array(self.state) + np.array(self.origin)
        cc = float(self.barecc + np.dot(np.dot(self.metric,shifted_state),shifted_state))
        my_reward = self.reward(cc)

        if cc > self.eps and cc < 2*self.eps:
            done = True
            print 'huzzah!', cc, self.state
            my_reward = self.BP_solved_factor/self.eps
            self.output_solution(cc)
        if cc < self.min_pos_cc and cc > 0:
            self.min_pos_cc = cc
            #print 'smallpos!', self.process_idx, cc, self.state, type(cc)
            self.output_min_pos_cc()
        if cc > self.max_neg_cc and cc < 0:
            self.max_neg_cc = cc
            #self.output_max_neg_cc()

        #if self.process_idx == 1: 
        #    print self.state
        #    print cc

        return np.array(self.state), my_reward, done, {}
        
    def reset(self):
        self.state = [0 for i in range(self.nmod)]
        if self.init_cc_printed == False:
            shifted_state = np.array(self.state) + np.array(self.origin)
            cc = float(self.barecc + np.dot(np.dot(self.metric,shifted_state),shifted_state))
            print 'initial cc:', cc
            self.init_cc_printed = True
        return np.array(self.state)
    
    def dist(self,cc):
        return abs(cc-self.eps)

    def reward(self,cc):
        d = self.dist(cc)
        return 1/d
        
    def random_metric(self): # pos def metric
        A = np.random.normal(size=(self.nmod,self.nmod), scale = self.sigma)
        return np.dot(A,A.transpose())

    def init_output(self):
        if not os.path.exists(os.path.dirname(self._outputFilePath)):
            try:
                os.makedirs(os.path.dirname(self._outputFilePath))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        # write header data
        hnd = open(self._outputFilePath, 'a+')
        hnd.write("head: " + str((self.eps,self.nmod,self.sigma,self.metric_index))+"\n")
        hnd.close()

    def output_min_pos_cc(self):
        # create path to file if necessary (it shouldn't be, the path should've been created by the training program
        if not os.path.exists(os.path.dirname(self._outputFilePath)):
            try:
                os.makedirs(os.path.dirname(self._outputFilePath))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
           
        # update the file
        hnd = open(self._outputFilePath, 'a+')
        hnd.write("p " + str((self.process_idx,self.global_t,self.min_pos_cc,self.state))+"\n")
        hnd.close()

    def output_max_neg_cc(self):
        # create path to file if necessary (it shouldn't be, the path should've been created by the training program
        if not os.path.exists(os.path.dirname(self._outputFilePath)):
            try:
                os.makedirs(os.path.dirname(self._outputFilePath))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
           
        # update the file
        hnd = open(self._outputFilePath, 'a+')
        hnd.write("n " + str((self.process_idx,self.global_t,self.max_neg_cc))+"\n")
        hnd.close()

    def output_solution(self,cc):
        # create path to file if necessary (it shouldn't be, the path should've been created by the training program
        if not os.path.exists(os.path.dirname(self._outputFilePath)):
            try:
                os.makedirs(os.path.dirname(self._outputFilePath))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
           
        # update the file
        hnd = open(self._outputFilePath, 'a+')
        hnd.write("s " + str((self.process_idx,self.global_t,cc,self.state))+"\n")
        hnd.close()
    
    def setOutputFilePath(self,path):
        #self._outputFilePath = path + "/outputOvercountsRun_"+ str(uuid.uuid4()) + ".txt"
        self._outputFilePath = path + "/output.txt"

    def setGlobal_t(self, global_t):
        self.global_t = global_t

    def setProcessIdx(self, idx):
        self.process_idx = idx
