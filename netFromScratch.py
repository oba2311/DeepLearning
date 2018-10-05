# numpy used for mathematical purposes:
import numpy as np
# matplotlib used for plotting:
import matplotlib.pyplot as plt

# activation func:
def sigmoid(x):
    activationResult = 1/(1+np.e**(-x))
    return activationResult

def sigmoid_derivative(x):
    sigmoid(x) * (1- sigmoid(x))

# Plot Activation Results to understand the
# Sigmoids behavior:
plottingData = np.linspace(-10,10, num=100)
plt.plot(sigmoid(plottingData))
plt.show()
