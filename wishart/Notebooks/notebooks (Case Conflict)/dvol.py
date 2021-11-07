from scipy import special
import numpy as np
import math

def digammap(k,z):
	res = 0
	for i in range(1,k+1):
		res += special.polygamma(0,[z/1.0+(1-i)/2.0])
	return res

def sigthresh(k,Q,eps):
	return math.exp((2.0*math.log(.5*k*math.pi**(.5*k)/math.gamma(.5*k+1.0)*eps)-digammap(k,.5*Q)-k*math.log(2.))*.5/k)

#test = [[0 for j in range(1,11)] for k in range(1,11)]
#keps = range(10,110,10)
#for i in range(len(keps)):
#        for j in range(len(keps)):
#                test[i][j] = sigthresh(keps[i],keps[i],10**(-keps[j]*1.0))
#                print i,j,sigthresh(keps[i],keps[i],10**(-keps[j]*1.0))
#print np.array(test)
