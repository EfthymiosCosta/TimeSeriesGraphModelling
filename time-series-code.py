#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 18:11:26 2023

@author: hutchings
"""

import numpy as np
import os
from scipy.stats import gamma
import pandas as pd

# to change current directory
class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)
############################## read data #####################################
#with cd("Documents/PhD-year-1/time-series-course/time_series-walden"):
with open('SDM_3.dat') as f:
    first_line = f.readline().strip('\n').split(' ')
    r, B, K, nf = np.loadtxt(first_line, unpack=True)
    lines = [line for line in f]
r, K, nf = int(r), int(K), int(nf)

# extract lines i1:i2 and convert to floats
def f(i1, i2):
    return np.array([float(x) for x in lines[i1:i2]])
frequencies = f(1, nf+1)

Sf = np.zeros((r,r,nf), dtype=complex) 
idx_start = nf+2 #starting index
jks=[]
for r_idx in range((r+1)*r//2):
    idx_ij = idx_start #occur every 2nf + 2 blank lines + 1ij lines
    i, j = [int(x) for x in lines[idx_ij].split()]
    jks += [(i, j)] # list of edges
    i1 = idx_ij + 1 #index of start of real part
    i2 = i1+nf #index at end of real part
    real = f(i1,i2) 
    idx_start = i2+1 +nf
    im = f(i2+1, idx_start)
    Sf[i-1, j-1, :] = real + 1j*im
    if i!=j:
        Sf[j-1, i-1,:] = np.conj(Sf[i-1, j-1, :])
 ##############################################################################       

### calculate indexes if f' values
assert(np.all(np.sort(frequencies) == frequencies))
f_prime_idx = [np.argmax(frequencies >= B/2)]
for i, freq, in enumerate(frequencies):
    tmp = f_prime_idx[-1]
    if i <= tmp:
        continue
    else:
        tmp2 = np.argmax(frequencies >= (B+frequencies[tmp]))
        if tmp2!=0:
            f_prime_idx += [ tmp2 ]
f_prime_idx = np.array(f_prime_idx)

#Efthymios index
f_prime_idx = np.array([98,274,490,686,882])

### calculate S^-1(f) for all f given
invSf = np.empty_like(Sf)
for i in range(nf):
    invSf[:,:,i] = np.linalg.inv(Sf[:,:,i])
    #assert(np.allclose(invSf[:,:,i]@Sf[:,:,i], np.eye(5)))


# calculate gamma(f) for all f given
jks2 = tuple([ (j, k) for j,k in jks if j!=k]) #j,k without j,j
gammaf = {'columns': jks2} #freq = row, (j,k) = col
gammaf['rows'] = frequencies
tmp = np.empty((nf, len(jks2)))
for i, jk in enumerate(jks2):
    j, k = jk
    num = invSf[j-1,k-1,:] * invSf[j-1,k-1,:].conj()
    denom = invSf[j-1,j-1,:]*invSf[k-1,k-1,:]
    assert(np.max(abs((num/denom).imag))<0.0001)
    tmp[:,i] = num/denom
gammaf['gamma'] = tmp

# selecting nly the fa (independent) needed
gammaf['gammafa'] = gammaf['gamma'][f_prime_idx, :]

W = -2*K* np.sum(np.log(gammaf['gammafa']), axis=0)

A = len(f_prime_idx)
q = r-1

alpha = A
beta = (K-q)/(2*K)

scale = 1/beta

L = (r**2-r)//2
Ci = np.arange(1, L+1)
alpha_hypothesis = 0.01
Ci = gamma.ppf((1-(alpha_hypothesis/Ci)), A, scale=scale)

tmp = 'Ci(' + str(alpha_hypothesis) + ')'
d = {'W':W, tmp:Ci, '(j, k)': jks2}
df = pd.DataFrame(data=d).set_index('(j, k)').sort_values('W', ascending=False, inplace=True)
print(df)
 
    
    
    


