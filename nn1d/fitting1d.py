import numpy as np 
import matplotlib.pyplot as plt 
from singleLayerNetwork1D import SingleLayerNetwork1D

def ballHeight(t):
    STARTING_HEIGHT = 10
    GRAVITATIONAL_ACCELERATION = 9.81
    return STARTING_HEIGHT - 0.5 * GRAVITATIONAL_ACCELERATION * t**2 

def generateBallHeightData(t_min, t_max, n_points, noise_stdev):
    t_measured = 0
    h_measured = 0
    return t_measured, h_measured

# def ballHeight(t):
#     STARTING_HEIGHT = 10
#     GRAVITATIONAL_ACCELERATION = 9.81
#     return STARTING_HEIGHT - 0.5 * GRAVITATIONAL_ACCELERATION * t**2 + np.sin(20*t)


T_MIN = 0.0
T_MAX = 1.0
NUMBER_OF_POINTS = 5
NOISE_STDEV = 1.0
t_measured = np.linspace(T_MIN, T_MAX, NUMBER_OF_POINTS)
h_ball = ballHeight(t_measured)
h_noise = NOISE_STDEV * np.random.randn(NUMBER_OF_POINTS)
h_measured = h_ball + h_noise

WIDTH = 100
weight_input = np.random.randn(WIDTH)
bias_input = np.random.randn(WIDTH)
weight_output = np.random.randn(WIDTH)

# Construct NN
myNeuralNetwork = SingleLayerNetwork1D(weight_input, bias_input, weight_output)

# Make prediction with NN (before training)
h_predict_before = np.zeros(len(t_measured))
for i in range(len(t_measured)):
    h_predict_before[i] = myNeuralNetwork.forward(t_measured[i])

# Steepest descent training
STEPSIZE = 0.01 
N_ITER = 50000
for i in range(N_ITER):
    input_weight_gradient, bias_gradient, output_weight_gradient = myNeuralNetwork.gradient(t_measured, h_measured)
    myNeuralNetwork.weight_input -= STEPSIZE * input_weight_gradient
    myNeuralNetwork.weight_output -= STEPSIZE * output_weight_gradient
    myNeuralNetwork.bias_input -= STEPSIZE * bias_gradient
    if i % 100 == 0:
        print("Iteration ", i)


# Make prediction with NN (after training)
h_predict_after = np.zeros(len(t_measured))
for i in range(len(t_measured)):
    h_predict_after[i] = myNeuralNetwork.forward(t_measured[i])

t_fine = np.linspace(T_MIN, T_MAX, 100)
h_predict_fine = np.zeros(len(t_fine))
for i in range(len(t_fine)):
    h_predict_fine[i] = myNeuralNetwork.forward(t_fine[i])


plt.figure()
plt.plot(t_measured, h_ball, 'o', label="True")
plt.plot(t_measured, h_measured, 'o', label="Noisy data")
plt.plot(t_measured, h_predict_before, 'o', label="NN prediction before training")
plt.plot(t_measured, h_predict_after, 'o', label="NN prediction after training")
plt.plot(t_fine, h_predict_fine, '-', label="NN prediction after training")
plt.legend()
plt.show()

