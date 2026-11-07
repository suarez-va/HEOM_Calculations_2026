import ctypes
_libc = ctypes.CDLL("libc.so.6")
_libc.mallopt(-3, 131072)   # M_MMAP_THRESHOLD - also disables glibc's dynamic adjustment
_libc.mallopt(-1, 131072)   # M_TRIM_THRESHOLD

import cmath
import sys
import cmath
import math
import os
import h5py

import numpy as np
import matplotlib.pyplot as plt

if sys.platform=="cygwin":
    from cyglibra_core import *
elif sys.platform=="linux" or sys.platform=="linux2":
    from liblibra_core import *
import util.libutil as comn
from libra_py import units
import libra_py.dynamics.heom.compute as compute
from libra_py import ft

from scipy.linalg import inv, pinv, lstsq, svd, eig, hankel
from scipy.integrate import trapezoid, quad
from scipy.special import iv # modified Bessel function of the first kind
from scipy.fft import fft, ifft, fftfreq, fftshift

##############################
### Hamiltonian Parameters ###
##############################
T = 315774.6670514614
beta = units.hartree / (units.boltzmann * T)
Omega = 2.00 / beta
Lambda = 12.0 / beta
epsilon = 6.0 / beta
Gamma = 2.00 / beta
Delta = 10**(-1.4) / beta
P1eq = np.exp(-0.5*beta*epsilon)/(np.exp(-0.5*beta*epsilon)+np.exp(0.5*beta*epsilon))
P2eq = np.exp(0.5*beta*epsilon)/(np.exp(-0.5*beta*epsilon)+np.exp(0.5*beta*epsilon))

# Thermalization Hamiltonian
Ham0 = CMATRIX(2,2)
Ham0.set(0,0,0.5*epsilon); Ham0.set(0,1,0.0);
Ham0.set(1,0,0.0); Ham0.set(1, 1,-0.5*epsilon)

# Full Hamiltonian
Ham = CMATRIX(2,2)
Ham.set(0,0,0.5*epsilon); Ham.set(0,1,Delta);
Ham.set(1,0,Delta); Ham.set(1, 1,-0.5*epsilon)

# Default HEOM Parameters
params = {
    # hierarchy parameters "KK":1, "LL":15,
    "KK":1, "LL":16,

    # bath parameters
    "spectral_density":1,
    "gamma":Gamma,
    "eta":Lambda,
    "omega":Omega,
    "temperature":T,
    "el_phon_couplings":initialize_el_phonon_couplings(2),

    # dynamics parameters
    "dt":0.1, "nsteps":201, "nprint":1,
    "verbosity":-1, "progress_frequency":0.2,

    # computational efficiency parameters
    "truncation_scheme":4, "do_scale":1,
    "adm_tolerance":0.0, "adm_deriv_tolerance":0.0,
    "filter_after_steps":1,"do_zeroing":0,
    "num_threads":1,

    # data management parameters 
    "prefix":"heom_therm",
    "hdf5_output_level":0, "txt_output_level":0, "mem_output_level":4,
    "properties_to_save": ["timestep", "time", "denmat", "auxdenmat"],
    "use_compression":0, "compression_level":[0,0,0]
    }

# Thermalization HEOM
rho0 = CMATRIX(2,2); rho0.set(0,0,1.0+0.0j)
compute.run_dynamics(params, Ham0, rho0)

# Dynamics HEOM
params.update({"dt":0.01*beta, "nsteps":1001, "nprint":1})
params.update({"prefix":"heom", "mem_output_level":3, "properties_to_save": ["timestep", "time", "denmat"]})

rho, adm = None, None
with h5py.File(F"heom_therm/mem_data.hdf", 'r') as f:
    rho = list(f["denmat/data"][-1,:,:])
    adm = list(f["auxdenmat/data"][-1,:,:,:])
rho = np.array(rho)
adm = np.array(adm)

rho_init = CMATRIX(2,2);
rho_init.set(0,0,rho[0,0]); rho_init.set(0,1,rho[0,1]);
rho_init.set(1,0,rho[1,0]); rho_init.set(1,1,rho[1,1])

adm_init = CMATRIXList()
for n in range(adm.shape[0]):
    rhon_init = CMATRIX(2,2);
    rhon_init.set(0,0,adm[n,0,0]); rhon_init.set(0,1,adm[n,0,1]);
    rhon_init.set(1,0,adm[n,1,0]); rhon_init.set(1,1,adm[n,1,1])
    adm_init.append(rhon_init)

compute.run_dynamics(params, Ham, rho_init, adm_init)

P1eq = np.exp(-0.5*beta*epsilon)/(np.exp(-0.5*beta*epsilon)+np.exp(0.5*beta*epsilon))
P2eq = np.exp(0.5*beta*epsilon)/(np.exp(-0.5*beta*epsilon)+np.exp(0.5*beta*epsilon))
time, rhot = None, None

with h5py.File(F"heom/mem_data.hdf", 'r') as f:
    time = list(f["time/data"][:])
    rhot = list(f["denmat/data"][:,:,:])
    
time = np.array(time)
rhot = np.array(rhot)

P2t = rhot[:,1,1].real
kt = np.gradient(P2t, time)/(1 - P2t/P2eq)
k_HEOM = kt[-1]

print(fr'log(beta hbar k_HEOM) = {float(np.log(k_HEOM)/np.log(10))}')
exit()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 4), sharex=True)
ax1.set_ylabel('$\mathbf{\\rho_{0}}(t)$')
ax1.plot(time / beta, rho0[:,0,0].real, label='$\\rho_{00}$', color = 'r') 
#ax1.plot(time / beta, rho0[:,1,1].real, label='$\\rho_{11}$', color = 'b') 
#ax1.plot(time / beta, rho0[:,0,1].real, label='$\mathrm{Re}[\\rho_{01}]$', color = 'g')
#ax1.plot(time / beta, rho0[:,0,1].imag, label='$\mathrm{Im}[\\rho_{01}]$', color = 'purple')
ax1.plot(time / beta, P1MTt, label='Marcus', color = 'k')
ax1.plot(time / beta, P1ZUSt, label='Zusman', color = 'darkorange')
ax1.plot(time / beta, P1NADt, label='Quantum', color = 'darkgreen')
ax1.legend(ncol=2, fontsize=9, loc='best', handletextpad=0.2, columnspacing=0.9)
ax2.set_xlabel('$t/\\beta\hbar$')
ax2.set_ylabel('$k(t)\\beta\hbar$')
ax2.plot(time / beta, kt * beta, color = 'k')#, label='$k(t)$' 
#ax2.legend(ncol=2, fontsize=9, loc='best', handletextpad=0.2, columnspacing=0.9)
plt.subplots_adjust(hspace=0.1)
plt.show()
plt.close()

print(f'k_HEOM = {float(np.log(kt[-1]*beta)/np.log(10))}')
print(f'k_MT = {float(np.log(kMT12*beta)/np.log(10))}')
print(f'k_ZUS = {float(np.log(kZUS12*beta)/np.log(10))}')
print(f'k_Quantum = {float(np.log(kNAD12*beta)/np.log(10))}')


#############################
### Rate Theory Benchmark ###
#############################
P1eq = np.exp(-0.5*beta*epsilon)/(np.exp(-0.5*beta*epsilon)+np.exp(0.5*beta*epsilon))
P2eq = np.exp(0.5*beta*epsilon)/(np.exp(-0.5*beta*epsilon)+np.exp(0.5*beta*epsilon))

kMT12 = 2*np.pi*Delta**2*np.sqrt(beta/(4*np.pi*Lambda))*np.exp(-beta*(Lambda-epsilon)**2/(4*Lambda))
kNAD12 = 2*np.pi/Omega*Delta**2*np.exp(0.5*beta*epsilon)*np.exp(-Lambda/(Omega*np.tanh(0.5*beta*Omega)))*iv(epsilon/Omega, 2*Lambda*np.exp(-0.5*beta*Omega)/(Omega*(1-np.exp(-beta*Omega))))
kZUS12 = kMT12 / (1 + 4*np.pi*Delta**2*Gamma/(Lambda*Omega**2))

kMT21 = P1eq/P2eq*kMT12
kNAD21 = P1eq/P2eq*kNAD12
kZUS21 = P1eq/P2eq*kZUS12

plt.rcParams.update({
    'figure.figsize': (6.0, 4.0),
    'figure.dpi': 150,
    'figure.facecolor': 'white',
    'figure.edgecolor': 'white',
    'lines.linewidth': 2,
    'axes.linewidth': 3,
    'axes.labelsize': 15,
    'axes.titlesize': 15,
    'xtick.direction': 'in',
    'xtick.top': True,
    'ytick.direction': 'in',
    'ytick.right': True,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'legend.frameon': False,
})

