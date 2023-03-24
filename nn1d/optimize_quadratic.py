import numpy as np 
import matplotlib.pyplot as plt
from steepestDescent import steepestDescent

a0 = 8.0
a1 = 1.0

def f_fun(x):
    return a0 * x[0]**2 + a1 * x[1]**2

def g_fun(x):
    return np.array([2*a0 * x[0], 2*a1 * x[1]])

# These create uniformly spaced arrays in x and y direction 
n_points = 100
xgv = np.linspace(-2, 2, n_points)
ygv = np.linspace(-2, 2, n_points)

# This creates a 2D grid of x and y values i.e. X has dimensions 100 by 100 
X, Y = np.meshgrid(xgv, ygv)

# This will evaluate the quadratic at each pair of x and y values on the grid 
Q = np.zeros(X.shape)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Q[i,j] = f_fun([X[i,j], Y[i,j]])

# We can plot grid values using matplotlib 
plt.figure()
plt.contour(X, Y, Q)
plt.title("Contour")


# Begin optimization here. Can change all of these around to see how it affects the optimization 
x0 = np.array([-1.0, 1.0])
step_size = 0.1
max_iter = 100
f_tol = 1e-6
g_tol = 1e-6

# Run the optimization algorithm 
x_opt = steepestDescent(f_fun, g_fun, x0, step_size, max_iter=max_iter, f_tol=f_tol, g_tol=g_tol)
print(x_opt.shape[0])
x0_all = x_opt[:,0]
x1_all = x_opt[:,1]

plt.plot(x0_all, x1_all, '-o')
plt.axis("equal")
plt.show()