from __future__ import print_function
import math
import random
import numpy as np
import time
import sys
from simanneal import Annealer

def random_metric(N,sigma): # pos def metric
		A = np.random.normal(size=(N,N), scale = sigma)
		#A = np.array([[sigma*mpm.sqrt(mpf(2))*mpm.erfinv(mpf(2)*mpm.rand()-mpf(1)) for i in range(self.nmod)] for j in range(self.nmod)])
		#mpm addition
		#A = np.array([[mpm.npdf(0,self.sigma) for i in range(self.nmod)] for j in range(self.nmod)])
		#print "metric test\n", A
		return np.dot(A,A.transpose())

class BP(Annealer):

	# pass extra data (the distance matrix) into the constructor
	def __init__(self, N, sig, eps, metric):
		self.metric = metric
		self.sigma = sig
		self.N = N
		self.eps = eps
		self.count = 0
		self.origin = None
		
		### 
		# shift origin in largest eigenvector direction
		eig_vals, eig_vecs = np.linalg.eig(np.array([[np.float(ii) for ii in jj] for jj in self.metric]))
		eig_vecs = np.transpose(eig_vecs)
		new_eig_vecs = []
		for i in range(len(eig_vecs)):
			new_eig_vecs.append(eig_vecs[i]/eig_vals[i]**.5)
		rounded_evecs = [[int(round(entry)) for entry in ev] for ev in new_eig_vecs]
		state = rounded_evecs[np.argmin(eig_vals)]


		super(BP, self).__init__(state)  # important!

	def move(self):
		action = np.random.randint(self.N)
		mag = np.random.randint(1)
		mag += 1
		idx, sign = (action-action%2)/2, (-1)**(action%2)
		self.state[idx] += sign*mag

	def energy(self):
		self.cc = -1 + np.dot(np.dot(self.metric,self.state),self.state)
		dist = abs(self.cc-self.eps)
		self.count += 1
		return -1/dist

if __name__ == '__main__':

	N = 10
	SIG = .0001
	EPS = .000000000001
	STEPS = 1e6

	np.random.seed(10)
	metric = random_metric(N,SIG)

	bp = BP(N,SIG,EPS,metric)
	bp.steps = STEPS
	bp.copy_stragtegy = "slice"

	state, e = bp.anneal()

	print("\n\nBest found")
	print(state)
	print(e)
	print(str(-1 + np.dot(np.dot(metric,state),state)))
