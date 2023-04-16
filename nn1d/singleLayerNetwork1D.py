import numpy as np 
import matplotlib.pyplot as plt

class SingleLayerNetwork1D:
    def __init__(self, weight_input, bias_input, weight_output, activation="sigmoid"):
        self.weight_input = weight_input
        self.bias_input = bias_input
        self.weight_output = weight_output
        self.activation = activation 

    def forward(self, x):
        """
        Given input x, return the output of the network 
        """
        if self.activation == "sigmoid":
            a = sigmoid(self.weight_input * x + self.bias_input)
            y = np.inner(self.weight_output, a)
            return y 

        if self.activation == "relu":
            a = relu(self.weight_input * x + self.bias_input)
            y = np.inner(self.weight_output, a)
            return y 
        else:
            a = self.weight_input * x + self.bias_input
            y = np.inner(self.weight_output, a)
            return y 
    
    def gradient(self, x_array, y_array):
        assert self.activation == "sigmoid" 
        assert len(x_array) == len(y_array)

        input_weight_gradient = np.zeros(len(self.weight_input))
        output_weight_gradient = np.zeros(len(self.weight_output))
        bias_gradient = np.zeros(len(self.bias_input))

        n_data = len(x_array)
        for i in range(n_data):
            misfit = y_array[i] - self.forward(x_array[i])

            output_weight_gradient -= misfit \
                * sigmoid(self.weight_input * x_array[i] + self.bias_input) * 2 / n_data
            
            input_weight_gradient -= misfit \
                * sigmoid_derivative(self.weight_input * x_array[i] + self.bias_input) \
                * x_array[i] * 2 / n_data
            
            bias_gradient -= misfit \
                * sigmoid_derivative(self.weight_input * x_array[i] + self.bias_input) \
                * 2 / n_data

        return input_weight_gradient, bias_gradient, output_weight_gradient


def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_derivative(x):
    return np.exp(-x)/(1 + np.exp(-x))**2

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
