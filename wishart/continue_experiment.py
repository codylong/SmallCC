import os
import operator
import subprocess
import os.path
import time
from notebooks.sigthresh import *

class ChainerRLExperiment():
	def __init__(self):
		self.stuff = Null
	def equals(self,exp):
		raise NotImplementedError

class WishartExperiment(ChainerRLExperiment):
	def __init__(self,sig=1,eps=1,n=1,gam=1,bet=1,max=1,pow=1):
		self.sigma = sig
		self.eps = eps
		self.nmod = n
		self.gamma = gam
		self.beta = bet
		self.maxsteps = max
		self.power = pow
	
	def print_sigma(self):
		print self.sigma
	
	def equals(self,sigma,epsilon,nmod,gamma,beta,maxsteps,power):
		if self.sigma !=sigma:
			return False
		if self.epsilon != epsilon:
			return False
		if self.nmod != nmod:
			return False
		if self.gamma != gamma:
			return False
		if self.beta != beta:
			return False
		if self.maxsteps != maxsteps:
			return False
		if step.power != power:
			return False

class ChainerRLExperiments():
	def __init__(self,sig=1,eps=1,n=1,gam=1,bet=1,max=1,pow=1,outdir = 'output',mtotal=1000):
		self.sigma = sig
		self.eps = eps
		self.nmod = n
		self.gamma = gam
		self.beta = bet
		self.maxsteps = max
		self.power = pow
		self.outdir = outdir
		self.maxtotal = mtotal
		self.dirstring = "sig" + str(self.sigma) + "eps" + str(self.eps) + "nmod" + str(self.nmod) + "gamma" + str(self.gamma) + "beta" + str(self.beta) + "maxsteps" + str(self.maxsteps) + "dpower" + str(self.power)

	
	def train(self):
		#create the new eperiment instance
		#new_experiment = WishartExperiment(sig,eps,n,gam,bet,max,pow)
		#now we need to check if such an experiment has been run before. first, define filename to look for
		#get all filenames in outdir
		#filenames = [name for name in os.listdir(self.outdir) if os.path.isdir(name)]
		#found = False
		#for filename in filenames:
		#	print filename
		#	if self.dirstring == filename:
		#		found = True
		if not os.path.isdir(self.outdir + '/' + self.dirstring):
			os.system('mkdir ' + self.outdir + '/' + self.dirstring)
			self.submit_new_job()
		else:
			self.continue_training()

	def continue_training(self):
		self.nndir = self.get_current_neural_directory()
		if self.nndir != '':
			self.submit_cont_job()
		else:
			self.submit_new_job()
			

	def submit_cont_job(self):
		substring = "train_a3c_gym.py 2 --steps " + str(self.maxtotal) + " --env cc-"+str(self.maxsteps)+"-v0 --outdir " + self.outdir + '/' + self.dirstring + " --gamma " + str(self.gamma) + " --eps " + str(self.eps) + " --nmod " + str(self.nmod) + " --sigma " + str(self.sigma) + " --beta " + str(self.beta)+ " --reward-d-pow " + str(self.power) + " --load " + str(self.nndir)
		self.substring = substring


	def submit_new_job(self):
		substring = "train_a3c_gym.py 2 --steps " + str(self.maxtotal) + " --env cc-"+str(self.maxsteps)+"-v0 --outdir " + self.outdir + '/' + self.dirstring + " --gamma " + str(self.gamma) + " --eps " + str(self.eps) + " --nmod " + str(self.nmod) + " --sigma " + str(self.sigma) + " --beta " + str(self.beta)+ " --reward-d-pow " + str(self.power)
		self.substring = substring



	def get_current_neural_directory(self):
		timestamps = [name for name in os.listdir(str(self.outdir+'/'+self.dirstring)) if os.path.isdir(str(self.outdir + '/' + self.dirstring) + '/' + name)]
		years = [stamp[0:4] for stamp in timestamps]
		months = [stamp[4:6] for stamp in timestamps]
		days = [stamp[6:8] for stamp in timestamps]
		hours = [stamp[8:10] for stamp in timestamps]
		minutes = [stamp[10:12] for stamp in timestamps]
		seconds = [stamp[12:] for stamp in timestamps]
		tosort =  [[timestamps[i],years[i],months[i],days[i],hours[i],minutes[i],seconds[i]] for i in range(len(timestamps))]
		sortedstamps = sorted(tosort, key = operator.itemgetter(1, 2, 3, 4, 5, 6))
		toreverse = [s[0] for s in sortedstamps]
		orderedstamps = toreverse[::-1]
		for i in orderedstamps:
			finishdirs = [f for f in os.listdir(self.outdir + '/' + self.dirstring + '/' + i + '/') if f.endswith('finish')]
			if len(finishdirs) > 0:
				return self.outdir + '/' + self.dirstring + '/' + i + '/' + finishdirs[0]
		return ''

class WishartExperiments(ChainerRLExperiments):

	def __init__(self,sig=1,eps=1,n=1,gam=1,bet=1,max=1,pow=1,outdir = 'output',mtotal=1000):
		self.sigma = sig
		self.eps = eps
		self.nmod = n
		self.gamma = gam
		self.beta = bet
		self.maxsteps = max
		self.power = pow
		self.outdir = outdir
		self.maxtotal = mtotal
		self.dirstring = "sig" + str(self.sigma) + "eps" + str(self.eps) + "nmod" + str(self.nmod) + "gamma" + str(self.gamma) + "beta" + str(self.beta) + "maxsteps" + str(self.maxsteps) + "dpower" + str(self.power)
		from continue_experiment import WishartExperiments












		
	
