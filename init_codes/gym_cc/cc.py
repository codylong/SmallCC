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

class CC(gym.Env):

    ########
    # RL related methods
    
    def __init__(self):
        self.nmod = 3
        self.mult=100
        self.action_space = None
        self.observation_space = None
        
        filename = "metrics_and_ccs/metric_and_barecc_"+str(self.nmod)+".pickle"
        if os.path.isfile(filename):
            self.metric, self.barecc = pickle.load(open(filename,"r"))
        else:
            self.metric = self.random_metric()
            self.barecc = self.random_bare_cc()
            pickle.dump([self.metric,self.barecc],open(filename,'w'))

        self.action_space = spaces.Discrete(2*self.nmod)
        self.observation_space = spaces.Discrete(self.nmod)
        self.state = [0 for i in range(self.nmod)]

    def _step(self,action):
        done = False
        idx, sign = (action-action%2)/2, (-1)**(action%2)
        self.state[idx] += sign
        cc = self.barecc + np.dot(np.dot(self.metric,self.state),self.state)
        my_reward = self.reward(cc)
        if my_reward >= 25*self.mult: 
        #if my_reward >= 100: 
            done = True
        if my_reward >= 10:
            f = open("ccs.txt","a+")
            f.write(str(self.state)+ " " + str(my_reward)+"\n")
            f.write(str(cc)+"\n")
            f.close()
        return np.array(self.state), my_reward, done, {}
        
    def _reset(self):
        self.state = [0 for i in range(self.nmod)]
        return np.array(self.state)
        
    def reward(self,cc):
        if cc < 0: return -1000
        return -1*self.mult*math.log10(cc)
        #return 10**(-1*self.mult*math.log10(cc))
        #return 10**(-1*math.log10(cc))

    def random_metric(self): # pos def metric
        A = scipyrandom.rand(self.nmod,self.nmod)
        return np.dot(A,A.transpose())*2/self.nmod

    def random_bare_cc(self):
        x = 0
        while x == 0: x = -1*random.random()
        return x

    ###
    # hacks to get it working on discovery
    
    def setPickleFilePath(self,path):
        self._pickleFilePath = path + "/outputOvercountsRun_"+ str(uuid.uuid4()) + ".pickle"

    def setGlobal_t(self, global_t):
        self.global_t = global_t
