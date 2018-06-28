import numpy as np
import matplotlib.pyplot as plt

# Load data
exp = np.genfromtxt('exp.data', delimiter=',')
MD = np.genfromtxt('../2-python/IR-data.txt', usecols=(0,5))

# Get normalization factors, which can really be any constant
norm_exp = exp[:,1].max() - exp[:,1].min()
norm_MD  = MD[:,1].max()  - MD[:,1].min()

# Plot
fig=plt.figure(figsize=[8.0,6.0])
plt.plot(MD[:,0] ,  ((MD[:,1]  - MD[:,1].min() ) / norm_MD))
plt.plot(exp[:,0], -((exp[:,1] - exp[:,1].min()) / norm_exp))
plt.xlabel(r'$\nu$ (cm$^{-1}$)')
plt.xlim([300,3600])
plt.gca().invert_xaxis()
plt.tick_params(axis='y', labelleft='off')
plt.legend(['MD', 'Exp.'], loc='best')
plt.savefig('IR-spectra-comparison.png')
plt.close()
