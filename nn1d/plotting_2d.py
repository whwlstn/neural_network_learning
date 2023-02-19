import numpy as np 
import matplotlib.pyplot as plt 

# We will try plotting contours of functions of 2D 
def quadratic(x, y):
    return x**2 + y**2 

# These create uniformly spaced arrays in x and y direction 
n_points = 100
xgv = np.linspace(-1, 1, n_points)
ygv = np.linspace(-1, 1, n_points)

# This creates a 2D grid of x and y values i.e. X has dimensions 100 by 100 
X, Y = np.meshgrid(xgv, ygv)

# This will evaluate the quadratic at each pair of x and y values on the grid 
Q = quadratic(X, Y) 

# We can plot grid values using matplotlib 
plt.figure()
plt.contour(X, Y, Q)
plt.title("Contour")
plt.show()

# We can also plot this as a 3D surface 
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Q)
plt.title("Surface")
plt.show()

# You can also color the surface with a colormap depending on the height of the surface
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Q, cmap='coolwarm')
plt.title("Colored surface")
plt.show()


