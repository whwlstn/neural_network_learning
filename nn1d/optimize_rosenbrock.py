import numpy as np 
import matplotlib.pyplot as plt
from steepestDescent import steepestDescent


def rosenbrock(x):
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


def rosenbrockGradient(x):
    return np.array([-2 + 2 * x[0] + 200 * (-2 * x[0] * x[1] + 2 * x[0] ** 3), 200 * (x[1]- x[0] ** 2)])


# These create uniformly spaced arrays in x and y direction 
n_points = 200
xgv = np.linspace(-2, 2, n_points)
ygv = np.linspace(-1, 3, n_points)

# This creates a 2D grid of x and y values i.e. X has dimensions 100 by 100 
X, Y = np.meshgrid(xgv, ygv)

# This will evaluate the quadratic at each pair of x and y values on the grid 
Q = np.zeros(X.shape)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Q[i,j] = rosenbrock([X[i,j], Y[i,j]])

# We can plot grid values using matplotlib 
plt.figure()
plt.pcolor(X, Y, Q)
plt.title("Contour")


# Begin optimization here. Can change all of these around to see how it affects the optimization 
x0 = np.array([-1.5, 2.0])
step_size = 0.001
max_iter = 10000
f_tol = 1e-10
g_tol = 1e-10

# Run the optimization algorithm 
x_opt = steepestDescent(rosenbrock, rosenbrockGradient, x0, step_size, max_iter=max_iter, f_tol=f_tol, g_tol=g_tol)
print(x_opt.shape[0])
x0_all = x_opt[:,0]
x1_all = x_opt[:,1]

plt.plot(x0_all, x1_all, '-or', markersize=1)
plt.axis("equal")
plt.show()