import mpmath as mpm
import numpy as np
from matplotlib import pyplot as plt

mu = 1.5
sigma = 1e-10

k = [sigma*np.sqrt(2)*mpm.erfinv(2*mpm.rand()-1)+mu for n in range(10000)]
k = [float(n) for n in k]
#print k

print 'mean', np.mean(k)
print 'std', np.std(k)

plt.hist(k,bins=50)
plt.show()
