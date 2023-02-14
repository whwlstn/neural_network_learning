import numpy as np 
import matplotlib.pyplot as plt

class SingleLayerNetwork1D:
    def __init__(self, weight_input, bias_input, weights_output, activation="sigmoid"):
        self.weight_input = weight_input
        self.bias_input = bias_input
        self.weights_output = weights_output
        self.activation = activation 

    def forward(self, x):
        """
        Given input x, return the output of the network 
        """
        if self.activation == "sigmoid":
            return sigmoid(self.weight_input * x + self.bias_input)
        if self.activation == "relu":
            return relu(self.weight_input * x + self.bias_input)
        else:
            return self.weight_input * x + self.bias_input
    
    def update(self, w_new, b_new):
        pass

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def relu(x):
    return np.maximum(x, 0)

if __name__ == "__main__":
    # generating an array
    x = np.linspace(-10, 10, 100)

    # making a neuron
    w = np.random.randn()
    b = np.random.randn()
    activation = "relu"
    neuron = SingleLayerNetwork1D(w, b, activation=activation)

    # Function of array
    y = SingleLayerNetwork1D.foward(x)

    # plotting
    plt.figure()
    plt.plot(x, y)
    plt.show()
