import numpy as np

def random_metric(k,sigma):
    A = np.random.normal(size=(k,k), scale = sigma)
    return np.dot(A,A.transpose())

def normalized_eigensystem(g):
    eig_vals, eig_vecs = np.linalg.eig(g)
    eig_vecs = np.transpose(eig_vecs)
    new_eig_vecs = []
    for i in range(len(eig_vecs)):
        new_eig_vecs.append(eig_vecs[i]/eig_vals[i]**.5)
        #print np.dot(np.dot(evec,g),evec)
    return (eig_vals, new_eig_vecs)

def compare_cc_distances(k,sigma):
    g = random_metric(k,sigma)
    eig_vals, eig_vecs = normalized_eigensystem(g)
    cc_diff = []
    for evec in eig_vecs:
        r = [round(v) for v in evec]
        #print np.dot(np.dot(evec,g),evec), np.dot(np.dot(r,g),r)
        cc_diff.append(np.dot(np.dot(evec,g),evec)- np.dot(np.dot(r,g),r))
    return cc_diff

sigs = [10**(-3),10**(-5),10**(-7)]
ks = [5,10,15,20,25,30,40,50,100]

for k in ks:
    for sig in sigs:
        print "\nBegin:", k,sig
        diffs = []
        for l in range(100):
            diffs.extend(compare_cc_distances(k,sig))
        print 'mean abs diff', np.mean([abs(d) for d in diffs]), 'and it over sig', np.mean([abs(d) for d in diffs])/sig

