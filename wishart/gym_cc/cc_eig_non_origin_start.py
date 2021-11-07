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

class CCNonOriginStart(gym.Env):

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
        #self._outputFilePath = os.path.split(os.path.abspath(getsourcefile(lambda:0)))[0] + "/../output/Pickle_" + datetime.datetime.now().strftime("%Y%m%dT%H%M%S") + "/outputOvercountsRun_"+ str(uuid.uuid4()) + ".pickle"
        self.min_pos_cc = 1e6
        self.max_neg_cc = -1e6

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
            pickle.dump([self.metric],open(filename,'w'))
	    self.metric = pickle.load(open(filename,"r"))
	evecs = np.linalg.eig(self.metric)[1][0]
	print evecs
	print 'sigma = ', self.sigma
	rounded_evecs = []
	for evec in evecs:
		#print evec
		#print abs(evec)
		minval = min(abs(evec))
		temp_vec = evec/minval
		temp_vec_rounded = []
		for entry in temp_vec:
			temp_vec_rounded.append(round(entry))
		rounded_evecs.append(temp_vec_rounded)
	for i in range(self.nmod):
		idvec = [0 for jj in range(self.nmod)]
		idvec[i] = 1
		rounded_evecs.append(idvec)
	###find smallest eigenvector, move out along it until we've found the correct cc, but not at a lattice point, round to lattice point
	evals = np.linalg.eig(self.metric)[0][0]
	mineval = evals[0]
	pos = 0
	for i in range(len(evals)):
		if evals[i] < mineval:
			mineval = evals[i]
			pos = i
	minevec = evecs[pos]
	scalefactor = ((1+1.5*self.eps)/(np.dot(np.dot(self.metric,minevec),minevec)))**0.5
	non_rounded_start = scalefactor*minevec
	rounded_start = []
	for entry in non_rounded_start:
		rounded_start.append(round(entry))
	self.initial_state = rounded_start
	self.evecs = rounded_evecs
	self.compute_dvol()
	print "dvol = ", self.dvol
        print self.evecs
	self.action_space = spaces.Discrete(4*self.nmod)
        self.observation_space = spaces.Discrete(self.nmod)
        self.state = self.initial_state
        print "Worker has (nmod,sigma,eps) = ", str((self.nmod,self.sigma,self.eps))

    def step(self,action):
        done = False
        idx, sign = (action-action%2)/2, (-1)**(action%2)
	vec = self.evecs[idx]
	for entry in range(len(vec)):
        	self.state[entry] += sign*vec[entry]
        cc = float(self.barecc + np.dot(np.dot(self.metric,self.state),self.state))
        my_reward = self.reward(cc)

        #if cc > self.eps and cc < 2*self.eps:
        if cc > self.eps and cc < 2*self.eps:
            done = True
            print 'huzzah!', cc, self.state
            self.output_solution(cc)
        if cc < self.min_pos_cc and cc > 0:
            self.min_pos_cc = cc
            self.output_min_pos_cc()
        if cc > self.max_neg_cc and cc < 0:
            self.max_neg_cc = cc
            #self.output_max_neg_cc()

        return np.array(self.state), my_reward, done, {}
        
    def reset(self):
        self.state = self.initial_state
        return np.array(self.state)
    
    def dist(self,cc):
        return abs(cc-self.eps)

    def reward(self,cc):
        d = self.dist(cc)
        return 1/d
        
    def random_metric(self): # pos def metric
        A = np.random.normal(size=(self.nmod,self.nmod), scale = self.sigma)
        return np.dot(A,A.transpose())*2/self.nmod

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

    def compute_dvol(self):
	self.vol = math.pi**(self.nmod/2.0)/math.gamma(.5*self.nmod+1.0)/(np.linalg.det(self.metric))**.5
	self.dvol = self.nmod/(2)*self.eps*self.vol
