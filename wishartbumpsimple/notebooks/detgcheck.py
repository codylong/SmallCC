import numpy as np

def random_metric(sigma,nmod): # pos def metric
    A = np.random.normal(size=(nmod,nmod), scale = sigma)
    return np.dot(A,A.transpose())

# below is hard to understand because multiple k-dep in log det g.
#metdetsk = [[]]
#sig = 1e-2
#for k in range(1,101):
#    metdets = [np.linalg.det(random_metric(sig,k)) for p in range(1000)]
#    metdetsk.append(metdets)
#for k in range(2,100):
#    print np.mean(metdetsk[k]) / np.mean(metdetsk[k-1])

# sigma only appears one place in log det g.

###
# In the next bit we fix k to a few different values and see how ratios
# of average determinants change as sigma is changed.
#

metdetssig = [[]]
k = 1
print 'k',k
for sig in range(1,10):
    metdets = [np.linalg.det(random_metric(sig*1e-5,k)) for p in range(100000)]
    metdetssig.append(metdets)
for i in range(2,10):
    print np.mean(metdetssig[i-1])/np.mean(metdetssig[i]), (1.0*(i-1)/i)**(2*k)
    #print np.mean(metdetssig[i-1]),np.mean(metdetssig[i])

metdetssig = [[]]
k = 10
print 'k',k
for sig in range(1,10):
    metdets = [np.linalg.det(random_metric(sig*1e-5,k)) for p in range(100000)]
    metdetssig.append(metdets)
for i in range(2,10):
    print np.mean(metdetssig[i-1])/np.mean(metdetssig[i]), (1.0*(i-1)/i)**(2*k)
    #print np.mean(metdetssig[i-1]),np.mean(metdetssig[i])

metdetssig = [[]]
k = 50
print 'k',k
for sig in range(1,10):
    metdets = [np.linalg.det(random_metric(sig*1e-5,k)) for p in range(10000)]
    metdetssig.append(metdets)
for i in range(2,10):
    print np.mean(metdetssig[i-1])/np.mean(metdetssig[i]), (1.0*(i-1)/i)**(2*k)
    #print np.mean(metdetssig[i-1]),np.mean(metdetssig[i])
