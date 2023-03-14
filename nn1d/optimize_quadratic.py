import numpy as np 
import matplotlib.pyplot as plt
from steepestDescent import steepestDescent

a1 = 1.0
a2 = 0.5 

def f_fun(x):
    return a1 * x[0]**2 + a2 * x[1]**2

def g_fun(x):
    return a1 * x[0]**2 + a2 * x[1]**2

# Begin optimization here. Can change all of these around to see how it affects the optimization 
x0 = np.array([1.0, 1.0])
step_size = 0.1 
max_iter = 100 
f_tol = 1e-8 
g_tol = 1e-8 

# Run the optimization algorithm 

