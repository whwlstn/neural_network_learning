import numpy as np 
import matplotlib.pyplot as plt 

# def ballHeight(t):
#     STARTING_HEIGHT = 10
#     GRAVITATIONAL_ACCELERATION = 9.81
#     return STARTING_HEIGHT - 0.5 * GRAVITATIONAL_ACCELERATION * t**2 


def ballHeight(t):
    STARTING_HEIGHT = 10
    GRAVITATIONAL_ACCELERATION = 9.81
    return STARTING_HEIGHT - 0.5 * GRAVITATIONAL_ACCELERATION * t**2 + np.sin(20*t)


T_MIN = 0.0
T_MAX = 1.0
NUMBER_OF_POINTS = 20
NOISE_STDEV = 0.5 
t_measured = np.linspace(T_MIN, T_MAX, NUMBER_OF_POINTS)
h_ball = ballHeight(t_measured)
h_noise = NOISE_STDEV * np.random.randn(NUMBER_OF_POINTS)
h_measured = h_ball + h_noise

plt.figure()
plt.plot(t_measured, h_ball, 'o')
plt.plot(t_measured, h_measured, 'o')

plt.show()

def generateBallHeightData(t_min, t_max, n_points, noise_stdev):
    t_measured = 0
    h_measured = 0
    return t_measured, h_measured