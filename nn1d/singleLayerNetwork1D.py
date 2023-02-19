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
            a = sigmoid(self.weight_input * x + self.bias_input)
            y = np.inner(self.weights_output, a)
            return y 

        if self.activation == "relu":
            a = relu(self.weight_input * x + self.bias_input)
            y = np.inner(self.weights_output, a)
            return y 
        else:
            a = self.weight_input * x + self.bias_input
            y = np.inner(self.weights_output, a)
            return y 
    
    def update(self, w_new, b_new):
        pass

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def relu(x):
    return np.maximum(x, 0)

if __name__ == "__main__":
    # generating an array
    x = np.linspace(-1, 1, 100)
    # making a neuron
    width = 100
    w = np.random.randn(width)
    b = np.random.randn(width)
    v = np.random.randn(width)
    activation = "sigmoid"
    layer = SingleLayerNetwork1D(w, b, v, activation=activation)

    # Function of array
    y = np.zeros(x.shape)
    for i in range(len(x)):
        y[i] = layer.forward(x[i])

    # y = layer.forward(x)
    # plotting
    plt.figure()
    plt.plot(x, y)
    plt.show()
