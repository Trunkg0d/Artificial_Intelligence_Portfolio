import numpy as np
import warnings

#suppress warnings
warnings.filterwarnings('ignore')

def sigmoiz(x):
    return 1 / (1 + np.exp(-x))
def classify(x, theta):
    return sigmoiz(np.dot(x, theta.T))

def gradient_descent(x, theta, y, num_iter, learning_rate = 0.001):
    m = x.shape[0]
    for i in range(num_iter):
        y_pred = classify(x, theta)
        grad = np.dot(x.T, y_pred - y) * learning_rate / m
        theta -= grad.T
    return theta

x_train = np.array([
    [1, 1000, 400],
    [1, 200, 860],
    [1, 100, 4500],
    [1, 300, 4600],
    [1, 800, 110],
    [1, 47, 550],
    [1, 500, 3000],
    [1, 120, 5000],
    [1, 6000, 500],
    [1, 7000, 350],
    [1, 900, 50]
], dtype=float)

y_train = np.array([
    [1],
    [0],
    [0],
    [0],
    [1],
    [0],
    [0],
    [0],
    [1],
    [1],
    [1]
], dtype=float)

x_test = np.array([
    [1, 6000, 4500],
    [1, 400, 890],
    [1, 6450, 4500],
    [1, 381, 4689],
    [1, 890, 115],
    [1, 470, 150],
    [1, 545, 3080],
    [1, 125, 5070],
    [1, 600, 5200],
    [1, 7034, 3150],
    [1, 900, 550]
], dtype=float)

# Initialize
theta = np.array([[0, 0, 0]], dtype=float)

new_theta = gradient_descent(x_train, theta, y_train, 100)

print(classify(x_test, new_theta))

