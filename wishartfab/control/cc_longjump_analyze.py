import sys
import os
import cPickle as pickle
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from pandas.plotting import scatter_matrix

# size is the number of parallel processes used
nmod, sigma, num_analyses, size = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])

ls_as = []
for rank in range(size):
    filename = 'pickles/eigk'+str(nmod)+'s'+str(sigma)+'numanalyses'+str(num_analyses)+"rank"+str(rank)+".pickle"
    if os.path.isfile(filename):
        print filename
        f = open(filename,'r')
        ls_as.extend(pickle.load(f))
        f.close()

concat = []
for i in range(len(ls_as)):
    for j in range(len(ls_as[i])): 
        concat.append(ls_as[i][j])

df = pd.DataFrame(concat)
#df = df.replace([np.inf,-np.inf],np.nan)
#df = df.dropna('index')

print df

for c in concat:
    print c['eigval']

#some of these are better of logged
df['log10eigval']= np.log10(df['eigval'])
df['log10smallestpos']= np.log10(df['smallestpos'])
df['log10o_cc']= np.log10(df['o_cc'])
df['log10meandiffs']= np.log10(df['meandiffs'])
df['log10meanccs']= np.log10(df['meanccs'])
df2 = df
#df2 = df.drop(columns=['box_size','eigval','smallestpos','meandiffs'])

df2.plot(kind='box', subplots=True, layout=(15,1), sharex=False, sharey=False)
plt.show()

df2.hist(bins=50)
plt.show()

scatter_matrix(df2)
plt.show()
