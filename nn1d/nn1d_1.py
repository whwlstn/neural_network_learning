import numpy as np 
import matplotlib.pyplot as plt 

# y = x^p 
x = np.arange(0, 100, 0.1)
y = x ** 5

plt.plot(x, y)
plt.title('y = x^p')
plt.show()


# sine graph
x = np.arange(-2 * np.pi, 2 * np.pi, 0.1)
y = np.sin(2 * x)

plt.plot(x, y)
plt.title('sine graph')
plt.show()

# cosine graph
x = np.arange(-2 * np.pi, 2 * np.pi, 0.1)
y = np.cos(3 * x)

plt.plot(x, y)
plt.title('cosine graph')
plt.show()


# tangent graph
x = np.arange(-2 * np.pi, 2 * np.pi, 0.1)
y = np.tan(x)

plt.plot(x, y)
plt.title('tangent graph')
plt.show()

# ln
x = np.arange(0, 1, 0.1)
y = np.ln(x)

plt.plot(x, y)
plt.title('y = ln(x)')
plt.show()

