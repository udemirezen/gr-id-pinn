#!/usr/bin/env python
"""
Testing out the PINN method on Burger's equation.
(Just a re-creation of work from the original PINN paper).
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

seed = 84
np.random.seed(seed)
tf.compat.v1.set_random_seed(seed)

# Lets be crazy and not do this with classes or any fanciness.
# Time for some bare-bones code to better elucidate the core ideas!

# Mysterious ML sage wisdom,
# Apparently this is a "good" initialization for symmetric saturating activation functions:
def xavier_init(size):
    in_dim = size[0]
    out_dim = size[1]
    xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
    return tf.Variable(tf.random.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

# First, construct & initialize the network:
layers = [2, 100, 100, 1] # Layer sizes
weights = []
biases = []
for l in range(0, len(layers)-1):
    W = xavier_init(size=[layers[l], layers[l+1]])
    b = tf.Variable(tf.zeros([1, layers[l+1]], dtype=tf.float32), dtype=tf.float32) # ???
    weights.append(W)
    biases.append(b)

# Now, define function for a forward pass
# (Isn't there soddme tf short-hand pre-implemented function for this sort of thing?)
def forward_pass(X, weights, biases):
    H = X
    for l in range(0, len(weights)-1):
        W = weights[l]
        b = biases[l]
        H = tf.tanh(tf.add(tf.matmul(H, W), b))
    W = weights[-1]
    b = biases[-1]
    Y = tf.add(tf.matmul(H, W), b) # No activation on final layer I guess
    return Y

# Define domain, bounds, initial conditions
x_min = -1.0
x_max = +1.0
t_min = 0.
t_max = 1.

u_left = 0.
u_right = 0.

def u_0(x):
    return np.sin(np.pi*((2.*(x-x_min)/(x_max-x_min))-1.))

# Generate training / regularization data
#To-do

# Define solution evaluation function, and equation satisfaction constraint
def u(t, x):
    T_norm = (2.*(t - t_min)/(t_max - t_min)) - 1.
    X_norm = (2.*(x - x_min)/(x_max - x_min)) - 1.
    u = forward_pass(tf.stack([T_norm, X_norm], axis=1), weights, biases)
    return tf.squeeze(u)

def f(t, x):
    u_val = u(t, x)
    u_t = tf.gradients(u_val, t)[0]
    u_x = tf.gradients(u_val, x)[0]
    u_xx = tf.gradients(u_x, x)[0]
    f_val = u_t + u_val*u_x - (0.01/np.pi)*u_xx # Viscosity coefficient hard-coded here
    return f_val

# Define loss function and optimization algorithm.
#To-do

# Initialize variables, create session:
sess = tf.Session()
init = tf.compat.v1.global_variables_initializer()
sess.run(init)

# Create place-holders for input & temporary variables
with tf.name_scope("Placeholders"):
    f_t = tf.placeholder(tf.float32, shape=[None])
    f_x = tf.placeholder(tf.float32, shape=[None])
    u_i_t = tf.placeholder(tf.float32, shape=[None])
    u_i_x = tf.placeholder(tf.float32, shape=[None])
    u_i = tf.placeholder(tf.float32, shape=[None])
    u_b_t = tf.placeholder(tf.float32, shape=[None])
    u_b_x = tf.placeholder(tf.float32, shape=[None])
    u_b = tf.placeholder(tf.float32, shape=[None])

# Create tf graphs for calculating stuff
u_init_pred = u(u_i_t, u_i_x)

# Train
#To-do

# Display

# Lets test that the network is built correctly
print("Weights:")
print(weights)
print("Biases:")
print(biases)
# Test that forward-pass works
t_arr = tf.constant([0.5,])
x_arr = tf.constant([0.3,])

print(u(t_arr, x_arr))
print(f(t_arr, x_arr))
