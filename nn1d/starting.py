import numpy as np 
import matplotlib.pyplot as plt 


class Neuron:
    def __init__(self, weight, bias, activation="sigmoid"):
        self.weight = weight 
        self.bias = bias 
        self.activation = activation 

    def forward(self, x):
        if self.activation == "sigmoid":
            return sigmoid(self.weight * x + self.bias)
        if self.activation == "relu":
            return relu(self.weight * x + self.bias)
        else:
            return self.weight * x + self.bias


    def update(self, w_new, b_new):
        pass 

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def relu(x):
    return np.maximum(x, 0)

# generating an array 
x = np.linspace(-10, 10, 100)

# making a neuron
w = np.random.randn()
b = np.random.rand()
activation = "relu"
neuron = Neuron(w, b, activation) 

# Function of array 
y = neuron.forward(x) 

# plotting 
plt.figure()
plt.plot(x, y)
plt.show()

# Random numbers in python 
random_normal_array = np.random.randn(1000)
random_uniform_array = np.random.rand(1000)
plt.figure()
plt.hist(random_normal_array)
plt.show()

plt.figure()
plt.hist(random_uniform_array)
plt.show()


