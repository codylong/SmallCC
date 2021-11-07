# literally taught in a course
# https://www.cs.dartmouth.edu/~ac/Teach/CS105-Winter05/Notes/nanda-scribe-3.pdf

def merge_lists(L,LP):
	ret = list(L[0:])
	for x in LP: 
		if x not in ret:
			ret.append(x)
	ret.sort()
	return ret

def trim(L,delta):
	y = L # y_i are elements from the L that is passed
	m, LP, last = len(L), [0], y[0]
	for i in range(1,len(y)):
		if y[i] > last*(1+delta):
			LP.append(y[i])
			last = y[i]
	return LP

def approx_subset_sum(S,t,eps):
	n, L = len(S), [[0]]
	for i in range(1,n+1):
		temp = merge_lists(L[i-1],[l + S[i-1] for l in L[i-1]])
		temp = trim(temp,eps/2.0/n)
		# print 't1', temp
		L.append([l for l in temp if l <= t])
	return max(L[n])


# print merge_lists([1,3,5],[2,6,4])


from mpmath import *
import numpy as np
mp.dps = 500

print "Below has a subset that sums to 15, so should return 15"
print approx_subset_sum([mpf(k) for k in [1,2,4,5,6]],mpf(15),mpf(.1))

def doctored_list(my_list,eps): # because of BP, want to doctor list to make small CC in there
	new_list = my_list[0:]
	new_list.extend([mpf(.25),mpf(.75),mpf(eps)])
	new_list.sort()
	return new_list

def doctor_gauss(N):
	return [mpf(random.gauss(.5,.5)) for i in range(N-3)]

def gauss(N):
	return [mpf(random.gauss(.5,.5)) for i in range(N)]

import random
from datetime import datetime

# print "\n\nNow play with doctered lists"
# for acc in range(3,15):
# 	target = mpf(1+1e-10+1e-15)
# 	eps_acc = mpf(10**(-acc))
# 	result = approx_subset_sum(doctored_list(doctor_gauss(10),mpf(1e-10)),target,eps_acc)
# 	print acc, target, "ratios:", target / result, result / target, "eps_acc:", eps_acc




# # For real CC
# print "\n\nReal CC, Doctored"
# LAMBDA = mpf(1e-120)
# N=10
# target = mpf(1+LAMBDA+LAMBDA/10)
# eps_acc = mpf(10**(-130))
# input_list = doctored_list(doctor_gauss(N),LAMBDA)
# t1 = datetime.now()
# result = approx_subset_sum(input_list,target,eps_acc)
# t2 = datetime.now()
# print "N:", N, "log10(1-result):", np.log10(float(result-1))

# For real CC
print "\n\nReal CC, undoctored"
LAMBDA = mpf(1e-120)
N=10
target = mpf(1+LAMBDA+LAMBDA/10)
eps_acc = mpf(10**(-120))
input_list = gauss(N)
t1 = datetime.now()
result = approx_subset_sum(input_list,target,eps_acc)
t2 = datetime.now()
print "N:", N, "log10(1-result):", np.log10(float(result-1))
print "result", result

print "\n\nNow test scaling with N"
for N in range(5,100,5):
	LAMBDA = mpf(1e-120)
	target = mpf(1+LAMBDA+LAMBDA/10)
	eps_acc = mpf(10**(-130))
	input_list = doctored_list(doctor_gauss(N),LAMBDA)
	t1 = datetime.now()
	result = approx_subset_sum(input_list,target,eps_acc)
	t2 = datetime.now()
	# print result
	print "N:", N, "log10(1-result):", np.log10(float(result-1)), "time:", (t2-t1).total_seconds()



# print "\n\nFix accuracy, increase N"
# time_inc_N = {}
# Ns = []
# for N in range(5,100,5):
# 	target = mpf(1+1e-10+1e-15)
# 	input_list = doctored_list(doctor_gauss(N),mpf(1e-10))
# 	eps_acc = mpf(10**(-2)) # to within a factor of (1-1/100), will test acc scaling soon
	
# 	### run approx
# 	t1 = datetime.now()
# 	result = approx_subset_sum(input_list,target,eps_acc)
# 	t2 = datetime.now()
# 	time_inc_N[N] = (t2 - t1).total_seconds()*1000
	
# 	### some prints, for eval
# 	print "\nN", N
# 	if Ns != []:
# 		print (N,np.log(time_inc_N[N]/time_inc_N[Ns[-1]]))
# 	print N, target, "time (ms):", time_inc_N[N], "ratios:", target / result, result / target, "eps_acc:", eps_acc
	
# 	Ns.append(N)