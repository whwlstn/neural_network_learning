import numpy as np 
import matplotlib.pyplot as plt 
from neuron1d import Neuron
num_neurons = 2000
neurons = [] 

for i in range(num_neurons):
    w = np.random.randn() 
    b = np.random.randn() 
    activation = "sigmoid"
    neurons.append(Neuron(w, b, activation))

x = np.linspace(-10, 10)
y_neuron = 0
output_weights = np.random.randn(num_neurons)
for neuron, output_weight in zip(neurons, output_weights):
    y_neuron += neuron.forward(x) * output_weight / num_neurons

plt.plot(x, y_neuron)
plt.show()


