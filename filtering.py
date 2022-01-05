
"""
Created on Tue Aug 27 11:46:49 2019

@author: Anna Kutschireiter
"""

import numpy as np
from scipy.stats import vonmises
from scipy.special import i0, i1
from scipy.optimize import root_scalar


##### Helper functions #####

def A_Bessel(kappa):
    """Computes the ratio of Bessel functions."""
    r = i1(kappa)/i0(kappa)
    return r

def A_Bessel_inv(r):
    """Computes the inverse of the ratio of Bessel functions by root-finding."""
    f = lambda kappa: A_Bessel(kappa) - r
    sol = root_scalar(f,bracket=[0,50],method='brentq')
    kappa = sol.root
    return kappa

def f_kappa(kappa):
    """ Computes the precision decay function in the circKF. """
    f = A_Bessel(kappa)/(kappa-A_Bessel(kappa)-kappa*A_Bessel(kappa)**2)
    return f

def polar_to_euclidean(r,phi):
    """ Converts a polar coordinate with radius r and angle phi to Cartesian coordinates. """
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return x, y

def euclidean_to_polar(x,y):
    """ Converts a Cartesian to polar coordinates. """
    r = np.sqrt( x**2 + y**2 )
    phi = np.arctan2(y,x)
    return phi,r

def xi(alpha):
    """ Xi function. Used to compute the Fisher information for a single observation with precision alpha. """
    info = alpha * i1(alpha)/i0(alpha)
    return info

def xi_fun_inv(dt):
    """ Inverse of xi function, determined by root-finding."""
    f = lambda alpha: alpha * A_Bessel(alpha) - dt
    sol = root_scalar(f,bracket=[0.001,50],method='brentq')
    alpha = sol.root
    return alpha

def circular_mean(phi,w=None):
    """ Computes a (weighted) circular mean of the vector of angles phi.

    Input:
        phi - angular positions of the particles
        w - weights

    Output:
        phi_hat - estimated angle
        r_hat - estimated precision in [0,1] """
    x = np.cos(phi)
    y = np.sin(phi)
    X = np.average(x,weights=w)
    Y = np.average(y,weights=w)
    
    # convert average back to polar coordinates
    phi_hat = np.arctan2(Y,X)
    r_hat = np.sqrt( X**2 + Y**2 )
    
    return phi_hat, r_hat


def backToCirc(phi):
    """Makes sure the angle phi is in [-pi,pi]."""
    phi = ( (phi+np.pi) % (2*np.pi) ) - np.pi
    return phi



##### Generate the data #####

def generateData(T,kappa_phi,kappa_y=0,alpha=0,dt=0.01,phi_0=0,kappa_0=0):
    """ Generates artifical data from the generative model for circular inference. 
    The hidden trajectory is a diffusion on a circle. Draws increment and direct 
    observations.

    Input:
    t - simulation length
    dt - time step
    kappa_phi - inverse diffusion constant
    kappa_y - reliability of increment observations
    kappa_z - reliability of direct observations
    phi_0 - initial mean 
    kappa_0 - initial certainty 

    Output:
    phi - trajectory of hidden process
    dy - increment observations
    z - direct observations
    """

    # hidden state
    phi = np.zeros(int(T/dt)) 
    #init
    if kappa_0 == 0:
        phi[0] = (phi_0 + np.pi ) 
    else:
        phi_0 = np.random.vonmises(phi_0,kappa_0)
        phi[0] = (phi_0 + np.pi )
    # generate sequence
    for i in range(1,int(T/dt)):
        phi[i] = np.random.normal(phi[i-1],1/np.sqrt(kappa_phi) * np.sqrt(dt))

    # increment observations
    dy = np.zeros(int(T/dt)) 
    if kappa_y != 0:
        for i in range(1,int(T/dt)):
            dy[i] = np.random.normal(phi[i]-phi[i-1],1/np.sqrt(kappa_y) * np.sqrt(dt)) 
    
    # correct range for hidden state
    phi = (phi % (2*np.pi) ) - np.pi # range [-pi,pi]

    # direct observations
    z = np.zeros(int(T/dt))
    if alpha != 0:
        z = np.random.vonmises(phi,alpha)
    
    return phi, dy, z




##### Filtering algorithms #####

## Von Mises Projection Filter (projecting the propagation on von Mises distribution), 
# mu kappa parametrization
def vM_Projection(mu,kappa,kappa_phi,z=None,alpha=0,dy=None,kappa_y=0,dt=0.01):
    """" A single step of the circular Kalman filter (vM projection filter), using Euler-Maruyama.
    
    Input:
    mu          - mean estimate before update
    kappa       - certainty estimate before update
    kappa_phi   - inverse diffusion constant of hidden state process
    z           - direct observation
    alpha     - precision of direct observation
    dy          - increment observation
    kappa_y     - precision of increment observation
    dt          - time step
    
    Output:
    mu_out      - mean estimate after update
    kappa_out   - certainty estimate after update """
    
    # update
    if alpha != 0:
        dmu_z = alpha/kappa * np.sin( z - mu ) #* dt
        dkappa_z = alpha * np.cos( z - mu )# * dt
    else:
        dmu_z = 0
        dkappa_z = 0

    # only update kappa if result isn't negative 
    if kappa + dkappa_z > 0:
        mu = mu + dmu_z
        kappa = kappa + dkappa_z

    # prediction (include increment observations)
    if kappa_y != 0:
        dmu_pred = kappa_y/(kappa_phi+kappa_y) * dy
    else:
        dmu_pred = 0
    dkappa_pred = - 1/2 * 1/(kappa_phi + kappa_y) * kappa * f_kappa(kappa) * dt

    mu_out = mu + dmu_pred
    mu_out = ((mu_out + np.pi) % (2*np.pi) ) - np.pi # mu in[-pi,pi]
    kappa_out = kappa + dkappa_pred
    
    return (mu_out,kappa_out)


def vM_Projection_Run(T,kappa_phi,z=None,alpha=0,dy=None,kappa_y=0,
                        phi_0=0,kappa_0=10,dt=0.01):
    """ Runs the circular Kalman filter for a sequence of observations.
    Input:
    T           - time at simulation end
    kappa_phi   - inverse diffusion constant of hidden state process
    z           - direct observation sequence
    alpha       - precision of direct observations
    kappa       - certainty estimate before update
    dy          - increment observation sequence
    kappa_y     - precision of increment observations
    phi_0       - initial mean estimate
    kappa_0     - initial precision estimate
    dt          - time step
    
    Output:
    mu_out      - mean estimate after update
    kappa_out   - certainty estimate after update 
    """

    mu = np.zeros(int(T/dt))
    mu[0] = phi_0
    kappa = np.zeros(int(T/dt))
    kappa[0] = kappa_0
    if kappa_y == 0:
        dy = np.zeros(int(T/dt))
    for i in range(1,int(T/dt)):
        [mu[i],kappa[i]] = vM_Projection(mu[i-1],kappa[i-1],
                                        kappa_phi, #diffusion
                                        z=z[i],alpha=alpha, # direct obs
                                        dy=dy[i],kappa_y=kappa_y, #relative heading info
                                        dt=dt)
    return mu, kappa



## Circular particle filter
def PF_effectiveDiff(x_in,w,kappa_phi,z=None,alpha=0,dy=None,kappa_y=0,dt=0.01):
    """" A single step in the particle filter, reliazed by propagating the particles
    through an effective diffusion (modulated by increment observations).
    
    Input:
    x_in        - initial particle positions
    w           - particle importance weights
    kappa_phi   - inverse diffusion constant of hidden state process
    z           - direct observation
    alpha       - precision of direct observation
    dy          - increment observation
    kappa_y     - precision of increment observation
    dt          - time step
    
    Output:
    x_out       - particle positions
    w           - particle weights
    
    """

    n = x_in.shape[0]
    kappa_eff = kappa_y + kappa_phi
    
    # particle diffusion
    x_in = x_in + np.pi # range between 0 and 2pi
    dx = np.random.normal(kappa_y/kappa_eff*dy,1/np.sqrt(kappa_eff) * np.sqrt(dt),n)
    x_out = x_in + dx
    x_out = (x_out % (2*np.pi) ) - np.pi # range between -pi and pi
    
    # compute weights
    if alpha != 0:       # only compute weights and resample if there is an observation
        w = w * vonmises.pdf(x_out, alpha,loc=z)
        w = w/np.sum(w)
        #resampling
        N_eff = 1/np.sum(w**2)
        if N_eff/n < 0.5:
            x_out = np.random.choice(x_out,n,p=w)
            w = 1/n * np.ones(n)
    
    return x_out, w

def PF_run(T,N,kappa_phi,z=None,alpha=0,dy=None,kappa_y=0,
                        phi_0=0,kappa_0=10,dt=0.01):
    """ Runs the particle filter for a sequence of observations.
    Input:
    T           - time at simulation end
    N           - number of particles
    kappa_phi   - inverse diffusion constant of hidden state process
    z           - direct observation sequence
    alpha       - precision of direct observations
    dy          - increment observation sequence
    kappa_y     - precision of increment observations
    phi_0       - initial mean estimate
    kappa_0     - initial precision estimate
    dt          - time step
    
    Output:
    mu          - mean estimate after update
    r           - certainty estimate after update, in [0,1]
    """

    mu = np.zeros(int(T/dt))
    mu[0] = phi_0
    r = np.zeros(int(T/dt))
    r[0] = A_Bessel(kappa_0)
    phi_PF = np.random.vonmises(phi_0,kappa_0,N)
    w = 1/N * np.ones(N)
    if kappa_y == 0:
        dy = np.zeros(int(T/dt))
    # propagate particles and compute statistics
    for i in range(1,int(T/dt)):
        phi_PF,w = PF_effectiveDiff(phi_PF,w,kappa_phi,z=z[i],alpha=alpha,dy=dy[i],kappa_y=kappa_y,dt=dt)
        mu[i],r[i] = circular_mean(phi_PF,w) 
    return mu, r


## Gauss ADF, vM parameters
def GaussADF(mu,kappa,kappa_phi,z=None,alpha=0,dy=None,kappa_y=0,dt=0.01):
    """" A single step in the Gauss ADF, using Euler-Maruyama.
    
    Input:
    mu          - mean estimate before update
    kappa       - certainty estimate before update
    kappa_phi   - inverse diffusion constant of hidden state process
    z           - direct observation
    alpha     - precision of direct observation
    dy          - increment observation
    kappa_y     - precision of increment observation
    dt          - time step
    
    Output:
    mu_out      - mean estimate after update
    kappa_out   - certainty estimate after update 
    """
    
    kappa_eff = kappa_phi + kappa_y

     # update
    if alpha != 0:
        diff = backToCirc(z - mu)
        dmu_z = alpha/kappa * np.sin( diff ) #* dt
        dkappa_z = alpha * np.cos( diff )# * dt
    else:
        dmu_z = 0
        dkappa_z = 0

    
    # only update kappa if result isn't negative 
    if kappa + dkappa_z > 0:
        mu = mu + dmu_z
        kappa = kappa + dkappa_z

    # prediction
    # (pretending to be a diffusion on the line)
    dmu_y = kappa_y/kappa_eff * dy
    dkappa_y = - 1/kappa_eff * kappa**2 * dt

    
    
    mu_out = mu + dmu_y 
    kappa_out = kappa + dkappa_y 
    mu_out = ((mu_out + np.pi) % (2*np.pi) ) - np.pi # mu in [-pi,pi]
    return (mu_out,kappa_out)

def GaussADF_run(T,kappa_phi,z=None,alpha=0,dy=None,kappa_y=0,
                        phi_0=0,kappa_0=10,dt=0.01):
    """ Runs the Gauss ADF for a sequence of observations.
    Input:
    T           - time at simulation end
    kappa_phi   - inverse diffusion constant of hidden state process
    z           - direct observation sequence
    alpha       - precision of direct observations
    kappa       - certainty estimate before update
    dy          - increment observation sequence
    kappa_y     - precision of increment observations
    phi_0       - initial mean estimate
    kappa_0     - initial precision estimate
    dt          - time step
    
    Output:
    mu_G        - mean estimate after update
    kappa_G     - certainty estimate after update 
    """
    mu_G = np.zeros(int(T/dt))
    mu_G[0] = phi_0
    kappa_G = np.zeros(int(T/dt))
    kappa_G[0] = kappa_0
    for i in range(1,int(T/dt)):
        [mu_G[i],kappa_G[i]] = GaussADF(mu_G[i-1],kappa_G[i-1],
                                        kappa_phi, #diffusion
                                        z=z[i],alpha=alpha, # direct obs
                                        dy=dy[i],kappa_y=kappa_y, #relative heading info
                                        dt=dt)
    return mu_G,kappa_G



##### Plotting #####

def circplot(t,phi):
    """ Stiches t and phi to make unwrapped circular plot. """
    
    phi_minus = phi - 2*np.pi
    phi_plus = phi + 2*np.pi

    phi_array = np.array((phi_plus , phi , phi_minus))
    difference = np.abs(phi_array[:,1:] - phi[0:-1])
    ind_up = np.where(np.argmin(difference,axis=0)==0)[0]
    ind_down = np.where(np.argmin(difference,axis=0)==2)[0]
    ind = np.union1d(ind_up,ind_down)

    phi_stiched = np.copy(phi)
    t_stiched = np.copy(t)
    for i in np.flip(np.arange(ind.size)):
        idx = ind[i]
        if np.isin(idx,ind_up):
            phi_stiched = np.concatenate((phi_stiched[0:idx+1],[phi_plus[idx+1]],
                                          [np.nan],[phi_minus[idx]],phi_stiched[(idx+1):]))
        else:
            phi_stiched = np.concatenate((phi_stiched[0:idx+1],[phi_minus[idx+1]],
                                          [np.nan],[phi_plus[idx]],phi_stiched[(idx+1):]))
        t_stiched = np.concatenate((t_stiched[0:idx+2],[np.nan],t_stiched[idx:]))

    
    return t_stiched,phi_stiched

