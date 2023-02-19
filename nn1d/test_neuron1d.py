import numpy as np 
import matplotlib.pyplot as plt 
from neuron1d import Neuron

activation = "sigmoid"
x = np.linspace(-10, 10, 100)
wb1 = np.random.randn(2)
neuron1 = Neuron(wb1[0], wb1[1], activation=activation)

wb2 = np.random.randn(2)
neuron2 = Neuron(wb2[0], wb2[1], activation=activation)

wb3 = np.random.randn(2)
neuron3 = Neuron(wb3[0], wb3[1], activation=activation)

v = np.random.randn(3)

plt.figure()
y = v[0] * neuron1.forward(x) +  v[1] * neuron2.forward(x) + v[2] * neuron3.forward(x)
plt.plot(x,y)
plt.show()


# Dot product 
w = np.random.randn(4)
v = np.random.randn(4)
vw = np.inner(v, w)