# Circular Kalman filter
This repo contains the code for simulations and figures in 

Kutschireiter, A., Rast, L., Drugowitsch, J., 2022. Projection Filtering with Observed State Increments with Applications in Continuous-Time Circular Filtering. *IEEE Transactions on Signal Processing*.


## Code structure and environment setup
* `root` - python scripts and jupyter notebooks containing the main code
* `/data_processed` - analysed artificial data
* `/data_raw` - raw data, empty (see below)

### Python environment

Our code uses Python 3.9.1. 
Packages we used are listed in the file `environment.yml`.
These can either be installed by hand, or alternatively installed in a virtual environment with name `cfltenv` by running
```
conda env create -f environment.yml
```
To activate this virtual environment, run
```
conda activate cfltenv
```
before running the code below.


### filtering.py
The file `filtering.py` includes the core implementation of circular filtering filtering algorithms we used.

Artificial data can be generated with the function `generateData()`. 
This will result in a single trajectory of a diffusion on the circule ("ground-truth heading direction"), as well as increment observations and/or direct angular observations.

These observations can be fed into the filter implementations, i.e., `vM_Projection_Run()` for the circular Kalman filter, `PF_run()` for the particle filter, and `GaussADF_run()` for the Gaussian assumed-density filter.

This file also a function `circplot()`, which is useful for plotting a time series of angular data, as well as further helper functions.

## Reproduce figures
Figures in the manuscript can be reproduced by running the Jupyter notebooks of the same name, i.e., `Figure_1a.ipynb` for Figure 1a etc. 
`Figure_1a.ipynb` and `Figure_2.ipynb` contain code that simulates the data and runs the filtering algorithms directly in the notebook.
Unaltered, `Figure_1b_1c.ipynb` and `Figure_3b_3c.ipynb` plot preprocessed (artificial) data we provide in the folder `/data_preprocessed`.

## Run simulations
To run the simulations that underlie the data for `Figure_1b_1c.ipynb` and `Figure_3b_3c.ipynb`, run `performance_scan.py` (for filtering with increment observations) or `performance_scan_direct.py` (for filtering with direct observations), thereby specifying the observation reliability and the number of iterations.

For instance,
```
performance_scan.py 6.0 500
```
will simulate 500 trajectories of the hidden process and of increment increment observations with reliability $\kappa_y = 6.0$.
For our manuscript we ran this script with reliabilities ranging from $\kappa_y = 0.1$ to $\kappa_y = 100$, with 5000 trajectories each.

Similarly
```
performance_scan_direct.py 2.5 1000
```
will simulate 1000 trajectories of the hidden process and direct observations with reliability $\kappa_z = 2.5$ at each time step.
For our manuscript we ran this script with reliabilities ranging from $\kappa_z = 0.1$ to $\kappa_z = 100$, with 5000 trajectories each.

Running such a simulation will result in an `.npz` archive in the folder `/data-raw`. To assess, process and plot this data, set
```
preprocess = True
```
in `Figure_1b_1c.ipynb` (for increment observations) and `Figure_3b_3c.ipynb` (for direct observations), and run the corresponding cell for preprocessing in these notebooks.
