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
import mpmath as mpm
from mpmath import *
import scipy.optimize as sp
import itertools as iter

##set the numerical accuracy in mpm:
mp.dps = 200

class CC(gym.Env):

	########
	# RL related methods
	
	def __init__(self):
		#self.nmod = 10
		#self.sigma = 1e-3
		#self.eps = 1e-3
		self.barecc = mpf(-1)
		
		self.action_space = None
		self.observation_space = None
		self.metric_index = None
		self.BP_solved_factor=mpf(1e10)
		#self._outputFilePath = os.path.split(os.path.abspath(getsourcefile(lambda:0)))[0] + "/../output/Pickle_" + datetime.datetime.now().strftime("%Y%m%dT%H%M%S") + "/outputOvercountsRun_"+ str(uuid.uuid4()) + ".pickle"
		self.min_pos_cc = mpf(1e8)
		self.max_neg_cc = mpf(-1e8)
		self.init_cc_printed = False
		self.global_t = 0
		self.trackscore = 0

	def second_init(self): # after nmod, sigma, eps are set, can run this
		###make sure eps is set at desired accuracy
		self.eps = mpf(self.eps)
		self.sigma = mpf(self.sigma)
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
		self.all_combis = iter.product([True, False], repeat=self.nmod)

		### 
		# shift origin in largest eigenvector direction
		# don't need mpmath precision here since we'll be rounding anyways.
		#linalg does not support mpf or float128, but we're rounding anyways so ordinary float is sufficient
		eig_vals, eig_vecs = np.linalg.eig(np.array([[np.float(ii) for ii in jj] for jj in self.metric]))
		#eig_vals = eig_vals[0]
		#eig_vecs = eig_vecs[0] # for some reason it's in a list itself, take 0th element
		eig_vecs = np.transpose(eig_vecs)
		self.np_metric  = [[float(self.metric[i][j]) for i in range(self.nmod)] for j in range(self.nmod)]
		
		
		#print 'metric',self.metric
		new_eig_vecs = []
		for i in range(len(eig_vecs)):
			new_eig_vecs.append(eig_vecs[i]/eig_vals[i]**.5)
		print 'evecs normalized? should be all 1', [np.dot(np.dot(self.metric,ev),ev) for ev in new_eig_vecs]
	
		origin = new_eig_vecs[np.argmin(eig_vals)]
		print self.origin
		if self.origin == None:
			self.origin = origin
		else:
			strorg = str(self.origin).split(',')
			self.origin = [float(i) for i in strorg if i !=',']
		self.mpf_origin = [mpf(i) for i in self.origin]
		self.mpmat_metric = mpm.matrix(self.metric)

		#print "origin \n", self.origin
	
		###to speed things up using mpm we define the ngvector, which is the state at time t dotted with the metric
		#self.ngvec = np.dot(np.array(self.origin),self.metric) # mpf due to self.origin is mpf above
		#self.cc = self.barecc + np.dot(np.dot(self.metric,self.origin),self.origin)
		#self.occ = self.cc
	
		#print "initial cc ", self.cc
		#print "ng vec\n",self.ngvec

		##
		# set up act, obs spaces
		self.action_space = spaces.Discrete(2*self.nmod)
		self.observation_space = spaces.Box(low=int(self.max_neg_cc), high=int(self.min_pos_cc), shape=(self.nmod, 1))
		self.state = self.origin
		#hnd = open(self._outputFilePath, 'a+')
		#hnd.write("occ" + str((self.process_idx,self.global_t,self.occ,self.state))+"\n")
		#hnd.close()
		#print "Worker has (nmod,sigma,eps) = ", str((self.nmod,self.sigma,self.eps))

	def step(self,action):
		done = False
		idx, sign = (action-action%2)/2, (-1)**(action%2)
		self.state[idx] += sign*10
		new_state,good_state, cc = self.find_closest(start = self.state)
		#shifted_state = np.array(self.state) + np.array(self.origin)
		#cc = float(self.barecc + np.dot(np.dot(self.metric,shifted_state),shifted_state))
		#cc = self.cc + self.metric[idx][idx] + 2*sign*self.ngvec[idx]
		#newngvec = [self.ngvec[i]+sign*self.metric[idx][i] for i in range(len(self.ngvec))]
		#self.ngvec = newngvec
		my_reward = self.reward(cc)
		self.cc = cc
		#self.trackscore += my_reward

		# sanity check for high precision cc calculation
		#shifted_state = np.array(self.state) + np.array(self.origin)
		#checkcc = float(self.barecc + np.dot(np.dot(self.metric,shifted_state),shifted_state))
		#if self.process_idx == 0: print 'sanity check', self.process_idx, (checkcc-cc)/cc

		#if cc > self.eps and cc < 2*self.eps:
		if cc > self.eps and cc < 2*self.eps:
			done = True
			#print 'huzzah!', cc, self.state
			my_reward = float(self.BP_solved_factor/self.eps)
			self.output_solution(cc)
		if cc < self.min_pos_cc and cc > 0:
			self.min_pos_cc = cc
			print 'smallpos!', self.process_idx, cc, self.state, good_state
			self.output_min_pos_cc()
		if cc > self.max_neg_cc and cc < 0:
			self.max_neg_cc = cc
			#self.output_max_neg_cc()

		#if self.process_idx == 1: 
		#    print self.state
		#    print cc
		self.state = new_state

		return np.array([float(k) for k in self.state]), my_reward, done, {}

	def find_closest(self,start):
	    # Try minimizing over non-integers and round the result:
	    def f(x):
	        return float(self.barecc) + np.dot(np.dot(self.np_metric, np.array(x)), np.array(x))

	    # Subject to the constraint that we get a pos CC:
	    def constr(x):
	        return float(self.barecc) + np.dot(np.dot(self.np_metric, np.array(x)), np.array(x))

	    tmp_solution = sp.fmin_cobyla(f, start, constr, maxfun=100000)
	    # return [int(round(t)) for t in tmp_solution]
	    
	    record = 1
	    record_comb = []
	    for comb in self.all_combis:
	        current = []
	        for i in range(len(comb)):
	            if comb[i]:
	                current.append(int(ceil(tmp_solution[i])))
	            else:
	                current.append(int(floor(tmp_solution[i])))
	        current_mpf = [mpf(i) for i in current]
	        res =self.barecc + (mpm.matrix(current_mpf).T * self.mpmat_metric * mpm.matrix(current_mpf))[0]
	        #res = (self.barecc) + np.dot(np.dot(self.metric, np.array(current_mpf)), np.array(current_mpf))
	        # if res > 0 and res < record:
	        if abs(res) < abs(record):
	            record = res
	            record_comb = current

	    return tmp_solution, record_comb, record
		
	def reset(self):
		self.state = self.origin
		#self.ngvec = np.dot(np.array(self.origin),self.metric) # mpf due to self.origin is mpf above
		#self.cc = self.barecc + np.dot(np.dot(self.metric,self.origin),self.origin)
		if self.init_cc_printed == True:
			print 'initial cc:', self.cc
		
		# create path to file if necessary (it shouldn't be, the path should've been created by the training program
		#if not os.path.exists(os.path.dirname(self._outputFilePath)):
		#	try:
		#		os.makedirs(os.path.dirname(self._outputFilePath))
		#	except OSError as exc: # Guard against race condition
		#		if exc.errno != errno.EEXIST:
		#			raise
		   
		# update the file
		#hnd = open(self._outputFilePath, 'a+')
		#hnd.write("rws " + str((self.process_idx,self.global_t,self.trackscore,self.state))+"\n")
		#hnd.close()
		#self.trackscore = 0
		return np.array(self.state)
	
	def dist(self,cc):
		return abs(cc-self.eps)

	def reward(self,cc):
		d = self.dist(cc)
		#do = self.dist(self.occ)
		#print np.float(1/d), np.float(1/d**self.pow) 
		return np.float(1/d**self.pow)
		#return 0
		
	def random_metric(self): # pos def metric
		#A = np.random.normal(size=(self.nmod,self.nmod), scale = self.sigma)
		A = np.array([[self.sigma*mpm.sqrt(mpf(2))*mpm.erfinv(mpf(2)*mpm.rand()-mpf(1)) for i in range(self.nmod)] for j in range(self.nmod)])
		#mpm addition
		#A = np.array([[mpm.npdf(0,self.sigma) for i in range(self.nmod)] for j in range(self.nmod)])
		#print "metric test\n", A
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
