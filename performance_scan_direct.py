 #! /usr/bin/env python3

import sys
import numpy as np
import filtering as flt
from time import time

kappa_z_array = np.array([0.01,0.03,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3,4,5,6,7,8,9,10])

# choose a kappa_y according to system variable
kappa_z = kappa_z_array[int(sys.argv[1])]

# pass the number of iterations
nIter = int(sys.argv[2])

# where to save
filename = "data_raw/performance_kappaz="+str(kappa_z)

# seeed the run with a value related to kappa_y
np.random.seed(int(kappa_z*10))

# model parameters
T = 10 # simulation time
dt = 0.01 # step size
t = np.arange(0,T,dt)
alpha = flt.xi_fun_inv(kappa_z * dt)
timesteps = int(T/dt)
kappa_phi = 1 # inverse diffusion constant
phi_0 = 0 # initial mean
kappa_0 = 20 # initial certainty
kappa_y = 1 # certainty of increment observations

# run the simulations and read out first and second order statistics for each time step
phi_final = np.zeros([nIter,timesteps])
vonMises = np.zeros([nIter,2,timesteps])
GaussADF = np.zeros([nIter,2,timesteps])
PF = np.zeros([nIter,2,timesteps])
start = time()

for i in range(0, nIter): # run for nIter iterations

    # generate data
    phi, dy, z = flt.generateData(T,kappa_phi,kappa_y=kappa_y,dt=dt,phi_0=phi_0,kappa_0=kappa_0,alpha=alpha)

    # Gauss ADF
    mu_G, kappa_G = flt.GaussADF_run(T,kappa_phi,dy=dy,kappa_y=kappa_y,z=z,alpha=alpha,
                            phi_0=phi_0,kappa_0=kappa_0,dt=dt)

    # Particle Filter
    N = 1000
    mu_PF, r_PF = flt.PF_run(T,N,kappa_phi,dy=dy,z=z,alpha=alpha,
                            kappa_y=kappa_y,phi_0=phi_0,kappa_0=kappa_0,dt=dt)

    # von Mises projection filter
    mu_VM, kappa_VM = flt.vM_Projection_Run(T,kappa_phi,dy=dy,kappa_y=kappa_y,z=z,alpha=alpha,
                            phi_0=phi_0,kappa_0=kappa_0,dt=dt)

    # read out statistics
    phi_final[i] = phi
    vonMises[i] = np.array([mu_VM,kappa_VM])
    GaussADF[i] = np.array([mu_G,kappa_G])
    PF[i] = np.array([mu_PF,r_PF])

np.savez(filename,phi_final=phi_final,vonMises=vonMises,GaussADF=GaussADF,PF=PF,
            kappa_phi=kappa_phi,kappa_y=kappa_y,kappa_z=kappa_z,T=T,dt=dt)

print('kappa_z = '+str(kappa_z)+' done \n')

end = time()
print(f'It took {end - start} seconds!')