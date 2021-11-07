import gym
from gym import spaces
import os
import errno
import math
import cPickle as pickle
import numpy as np
import mpmath as mpm
from mpmath import *

##set the numerical accuracy in mpm:
mp.dps = 200


class CCMCTS(gym.Env):

	########
	# RL related methods
	
	def __init__(self):
		# use some standard values for initialization, overwritten by second init
		self.nmod = 10
		self.sigma = 1e-3
		self.eps = 1e-3
		self.vol = 1.0e10
		self.dvol = 1.0e9
		self.metric = []
		self.vecs = []
		self.state = []
		self.global_t = 0
		self.process_idx = 0
		self._outputFilePath = "./output.txt"
		self.origin = []

		self.barecc = mpf(-1)
		self.action_space = None
		self.observation_space = None
		self.metric_index = None
		# self._outputFilePath = os.path.split(os.path.abspath(getsourcefile(lambda:0)))[0] + "/../output/Pickle_" + datetime.datetime.now().strftime("%Y%m%dT%H%M%S") + "/outputOvercountsRun_"+ str(uuid.uuid4()) + ".pickle"
		self.min_pos_cc = mpf(1e6)
		self.max_neg_cc = mpf(-1e6)
		self.init_cc_printed = False
	


		# for the MCTS
		self.tree = {}  # dictionary of the form {(a_1,1): [state, reward], ..., (a_1,depth): [state, reward], (a_2,1,a_2,2): [state, acuumumlated reward], ...}
		self.tree_width = 3  # note: for more tree reusability, make the tree deep but not wide
		self.tree_depth = 3

		self.second_init()

	def second_init(self):  # after nmod, sigma, eps are set, can run this
		###make sure eps is set to desired accuracy
		self.eps = mpf(self.eps)
		self.sigma = mpf(self.sigma)
		if not os.path.exists("./metrics"):
			os.makedirs("./metrics")

		filename = "./metrics/metric" + str(self.nmod) + 'sig' + str(self.sigma) + "v" + str(self.metric_index) + ".pickle"
		if os.path.isfile(filename):
			self.metric = pickle.load(open(filename, "r"))
		else:  # new metric
			self.metric = self.random_metric()
			existing_metric_files = [f for f in os.listdir("./metrics") if "metric" + str(self.nmod) + "v" in f]
			metric_versions = [int(f[f.index('v') + 1: f.index('.')]) for f in existing_metric_files]
			print "Existing metric versions are: ", metric_versions
			if metric_versions == []:
				self.metric_index = 1
			else:
				self.metric_index = max(metric_versions) + 1
			filename = "./metrics/metric" + str(self.nmod) + 'sig' + str(self.sigma) + "v" + str(self.metric_index) + ".pickle"
			pickle.dump(self.metric, open(filename, 'w'))
			self.metric = pickle.load(open(filename,"r"))
		
		eig_vals, eig_vecs = np.linalg.eig(np.array([[np.float(ii) for ii in jj] for jj in self.metric]))
		eig_vecs = np.transpose(eig_vecs)

		print 'metric', self.metric
		new_eig_vecs = []
		# get the normalized (length-1) eigenvectors
		for i in range(len(eig_vecs)):
			new_eig_vecs.append(eig_vecs[i] / eig_vals[i] ** .5)
		print 'evecs normalized? should be all 1', [np.dot(np.dot(self.metric, ev), ev) for ev in new_eig_vecs]
		rounded_evecs = [[int(round(entry)) for entry in ev] for ev in new_eig_vecs]
		# step out along the largest eigenvector direction a length of 1. this is arguably where the most possible small cc points live
		origin_prempm = rounded_evecs[np.argmin(eig_vals)]
		self.origin = [mpf(i) for i in origin_prempm]
	
		###to speed things up using mpm we define the ngvector, which is the state at time t dotted with the metric
		self.ngvec = np.dot(np.array(self.origin),self.metric) # mpf due to self.origin is mpf above
		self.cc = self.barecc + np.dot(np.dot(self.metric,self.origin),self.origin)
		#print "initial cc ", self.cc
		#print "ng vec\n",self.ngvec
	
		self.action_space = spaces.Discrete(2*self.nmod)
		self.observation_space = spaces.Box(low=int(self.max_neg_cc), high=int(self.min_pos_cc), shape=(self.nmod, 1))
		self.state = [0 for i in range(self.nmod)]
		#print "Worker has (nmod, sigma, eps) = ", str((self.nmod, self.sigma, self.eps))

	def step(self, action):
		done = False
		new_state, my_reward, cc = self.do_step(action, self.state)

		# update state
		self.state = new_state

		# if cc > self.eps and cc < 2*self.eps:
		if cc > self.eps and cc < 2*self.eps:
			done = True
			print 'huzzah!', cc, self.state
			self.output_solution(cc)
		if cc < self.min_pos_cc and cc > 0:
			self.min_pos_cc = cc
			self.output_min_pos_cc()
		if cc > self.max_neg_cc and cc < 0:
			self.max_neg_cc = cc
			# self.output_max_neg_cc()

		return np.array([int(k) for k in self.state]), my_reward, done, {}

	def do_step(self, action, start_state):
		new_state = list(start_state)
		idx, sign = (action-action % 2)/2, mpf((-1)**(action % 2))

		new_state[idx] += sign
		# we started the search with the origin shifted out to the edge of the ellipsoid, so we need to shift the state by the same amount
		cc = self.cc + self.metric[idx][idx] + 2*sign*self.ngvec[idx]
		newngvec = [self.ngvec[i]+sign*self.metric[idx][i] for i in range(len(self.ngvec))]
		self.ngvec = newngvec
		self.cc = cc
		######TO TEST UNCOMMENT THESE THREE LINES
			#shifted_state = np.array(new_state) + np.array(self.origin)
			#test_cc = float(self.barecc + np.dot(np.dot(self.metric, shifted_state), shifted_state))
		#print (test_cc - cc)/cc
		reward = self.reward(cc)

		return new_state, reward, cc

	def reset(self):
		self.state = [0 for i in range(self.nmod)]
		self.ngvec = np.dot(np.array(self.origin),self.metric) # mpf due to self.origin is mpf above
		self.cc = self.barecc + np.dot(np.dot(self.metric,self.origin),self.origin)
		if self.init_cc_printed == True:
			print 'initial cc:', self.cc
		return np.array(self.state)

	
	def dist(self, cc):
		return abs(cc - self.eps)

	def reward(self, cc):
		d = self.dist(cc)
		return np.float(1./d)
		
	def random_metric(self):  # pos def metric
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
			except OSError as exc:  # Guard against race condition
				if exc.errno != errno.EEXIST:
					raise

		# write header data
		hnd = open(self._outputFilePath, 'a+')
		hnd.write("head: " + str((self.eps, self.nmod, self.sigma, self.metric_index)) + "\n")
		hnd.close()

	def output_min_pos_cc(self):
		# create path to file if necessary (it shouldn't be, the path should've been created by the training program
		if not os.path.exists(os.path.dirname(self._outputFilePath)):
			try:
				os.makedirs(os.path.dirname(self._outputFilePath))
			except OSError as exc:  # Guard against race condition
				if exc.errno != errno.EEXIST:
					raise
		   
		# update the file
		hnd = open(self._outputFilePath, 'a+')
		hnd.write("p " + str((self.process_idx, self.global_t, self.min_pos_cc, self.state)) + "\n")
		hnd.close()

	def output_max_neg_cc(self):
		# create path to file if necessary (it shouldn't be, the path should've been created by the training program
		if not os.path.exists(os.path.dirname(self._outputFilePath)):
			try:
				os.makedirs(os.path.dirname(self._outputFilePath))
			except OSError as exc:  # Guard against race condition
				if exc.errno != errno.EEXIST:
					raise
		   
		# update the file
		hnd = open(self._outputFilePath, 'a+')
		hnd.write("n " + str((self.process_idx, self.global_t, self.max_neg_cc)) + "\n")
		hnd.close()

	def output_solution(self, cc):
		# create path to file if necessary (it shouldn't be, the path should've been created by the training program
		if not os.path.exists(os.path.dirname(self._outputFilePath)):
			try:
				os.makedirs(os.path.dirname(self._outputFilePath))
			except OSError as exc:  # Guard against race condition
				if exc.errno != errno.EEXIST:
					raise
		   
		# update the file
		hnd = open(self._outputFilePath, 'a+')
		hnd.write("s " + str((self.process_idx, self.global_t, cc, self.state)) + "\n")
		hnd.close()
	
	def set_output_file_path(self, path):
		self._outputFilePath = path + "/output.txt"

	def set_global_t(self, global_t):
		self.global_t = global_t

	def set_process_idx(self, idx):
		self.process_idx = idx

	###
	# MCTS Methods

	def recursive_build_tree(self, state, actions, accumulated_reward, depth, agent, always_use_predictions):
		# max recursion depth reached, update tree and return
		if depth >= self.tree_depth:
			self.tree[tuple(actions)] = [state, accumulated_reward]
			return

		# get tree_width many best actions for a given state
		if always_use_predictions:
			best_actions = self.get_best_actions(state, agent)
		else:
			if depth == 0:
				best_actions = self.get_best_actions(state, agent)
			else:
				best_actions = np.random.choice(self.action_space.n, self.tree_width)

		# get all keys corresponding to pre-computed states at this tree level
		nth_leaves = [k for k in self.tree if len(k) == depth + 1]
		# loop over all actions
		for a in best_actions:
			new_action = actions + [a]
			if tuple(new_action) in nth_leaves:
				state, reward = self.tree[tuple(new_action)]
			else:  # compute new contributions
				state, reward, _ = self.do_step(a, state)

			self.tree[tuple(new_action)] = [state, accumulated_reward + reward]
			self.recursive_build_tree(state, new_action, accumulated_reward + reward, depth + 1, agent, always_use_predictions)

	def get_best_actions(self, state, agent):
		statevar = agent.batch_states(np.array([state]), np, agent.phi)
		pout, vout = agent.model.pi_and_v(statevar)
		pout_probs = pout.all_prob.data[0]
		pout_top_action_probs = sorted(pout_probs, reverse=True)
		pout_top_action_probs = pout_top_action_probs[0:self.tree_width]
		pout_top_actions = []
		found_enough_actions = False
		for ap in pout_top_action_probs:
			if found_enough_actions:
				break
			pos = np.where(pout_probs == ap)[0]
			if len(pos) + len(pout_top_actions) > self.tree_width:
				for i in range(len(pos)):
					pout_top_actions.append(pos[i])
					if len(pout_top_actions) == self.tree_width:
						found_enough_actions = True
						break
			else:
				pout_top_actions.append(pos[0])
		return pout_top_actions

	def update_tree(self, new_tree_root):
		# reshuffle the tree by making new_tree_root the new root
		new_tree = {}
		root_reward = self.tree[(new_tree_root,)][1]
		for acts, vals in self.tree.items():
			if (acts[0] != new_tree_root) or (len(acts) == 1 and acts[0] == new_tree_root):  # throw away old branches and leave that is now root
				continue

			# now shift the keys one to the left, i.e. rebuild the tree w.r.t. to new root
			new_tree[tuple([acts[i] for i in range(1, len(acts))])] = [vals[0], vals[1] - root_reward]

		self.tree = new_tree
		return

	def get_best_action_and_update_tree(self, agent):
		# return random action to keep exploring:
		if np.random.random() > 0.9:
			return np.array([np.random.randint(0, self.action_space.n)])
		# fill tree first
		self.recursive_build_tree(self.state, [], 0, 0, agent, True)

		# now find the best action, i.e. the one that accumulates the largest reward
		last_leaves = {reward[-1]: acts[0] for acts, reward in self.tree.items() if len(acts) == self.tree_depth}
		best_action = last_leaves[max(last_leaves.keys())]

		# make this selected best action the new tree root
		self.update_tree(best_action)
		return np.array([best_action])
