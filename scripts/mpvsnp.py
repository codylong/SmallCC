import mpmath as mpm
import numpy as np
import time
import random

def test(nmod,sig,num_samples,dps):
	mpm.dps = dps

	ratios = []
	for k in range(num_samples):
		m1 = np.random.rand(nmod,1)
		m2 = [[mpm.npdf(0,sig) for i in range(nmod)] for j in range(1)]
		m2 = np.array(m2)

		t1 = time.time()
		p1 = np.dot(m1,m1)
		t2 = time.time()
		p2 = np.dot(m2,m2)
		t3 = time.time()

		ratios.append((t3-t2)/(t2-t1))
	return np.mean(ratios)

def testspeedmpm(nmod,sig,num_samples,dps):
	mpm.dps = dps

	times = []
	cc = -1
	nvec = np.array([[random.randint(0,5) for i in range(nmod)] for j in range(1)])
	m2 = np.array([[mpm.npdf(0,sig) for i in range(nmod)] for j in range(nmod)])
	print 'm2'
	print m2
	print 'nvec'
	print nvec
	nvecm2 = np.dot(nvec,m2)
	print 'nvecm2'
	print nvecm2
	nvecm20 = nvecm2[0][0]
	print 'nvecm20'
	print nvecm20
	print 'whoa'
	print nvecm2
	nvecm2list = [nvecm2[0][i] for i in range(nmod)]
	print 'nvecm2list'
	print nvecm2list
	for k in range(num_samples):
		sign, idx = (-1)**random.randint(0,nmod-1),random.randint(0,nmod-1)
		t2 = time.time()
		cc = cc +2*sign*nvecm2list[idx]+m2[idx][idx]
		nvecm2listnew = [nvecm2list[i]+sign*m2[idx][i] for i in range(len(nvecm2list))]
		nvecm2list = nvecm2listnew
		t3 = time.time()
		print 'puts',nvecm20
		print 'nvecm2list'
		print nvecm2list

		times.append((t3-t2))
	return np.mean(times)

def testspeednpmatrix(nmod,sig,num_samples,dps):
	mpm.dps = dps

	times = []
	for k in range(num_samples):
		m1 = np.random.rand(nmod,nmod)
		t2 = time.time()
		p2 = np.dot(np.transpose(m1),m1)
		t3 = time.time()

		times.append((t3-t2))
	return np.mean(times)

def testspeednp(nmod,sig,num_samples,dps):
	mpm.dps = dps

	times = []
	for k in range(num_samples):
		m1 = np.random.rand(nmod,1)
		t2 = time.time()
		p2 = np.dot(np.transpose(m1),m1)
		t3 = time.time()

		times.append((t3-t2))
	return np.mean(times)

for d in range(1,50,5):
	print 'begin',d
	print testspeedmpm(100,1e-4,10,d)* 20000000 / 60 / 60
	print testspeednp(100,1e-4,10,d)* 20000000 / 60 / 60
	print testspeednpmatrix(100,1e-4,10,d)* 20000000 / 60 / 60
