#!/usr/bin/env python
"""
Testing out the PINN method on Burger's equation.
(Just a re-creation of work from the original PINN paper).
"""

import tensorflow as tf
import numpy as np
import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt
from pyDOE import lhs

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
layers = [2, 20, 20, 1] # Layer sizes
weights = []
biases = []
for l in range(0, len(layers)-1):
    W = xavier_init(size=[layers[l], layers[l+1]])
    b = tf.Variable(tf.zeros([1, layers[l+1]], dtype=tf.float32), dtype=tf.float32) # ???
    weights.append(W)
    biases.append(b)

# Now, define function for a forward pass
# (Isn't there some tf short-hand pre-implemented function for this sort of thing?)
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

# Define domain
x_min = -1.0
x_max = +1.0
t_min = 0.
t_max = 1.

# Define solution evaluation function, and equation satisfaction constraint
def u(t, x):
    T_norm = (2.*(t - t_min)/(t_max - t_min)) - 1.
    X_norm = (2.*(x - x_min)/(x_max - x_min)) - 1.
    u = forward_pass(tf.stack([T_norm, X_norm], axis=1), weights, biases)
    return tf.squeeze(u)

def u_x(t, x):
    u_val = u(t, x)
    u_x = tf.gradients(u_val, x)[0]
    return u_x

def u_t(t, x):
    u_val = u(t, x)
    u_t = tf.gradients(u_val, t)[0]
    return u_t

def f(t, x):
    u_val = u(t, x)
    u_t = tf.gradients(u_val, t)[0]
    u_x = tf.gradients(u_val, x)[0]
    u_xx = tf.gradients(u_x, x)[0]
    f_val = u_t + u_val*u_x - (0.01/np.pi)*u_xx # Viscosity coefficient hard-coded here
    return f_val

# Create place-holders for input & temporary variables
with tf.name_scope("Placeholders"):
    t_int = tf.compat.v1.placeholder(tf.float32, shape=[None])
    x_int = tf.compat.v1.placeholder(tf.float32, shape=[None])
    t_init = tf.compat.v1.placeholder(tf.float32, shape=[None])
    x_init = tf.compat.v1.placeholder(tf.float32, shape=[None])
    u_init = tf.compat.v1.placeholder(tf.float32, shape=[None])
    t_boundary_xmin = tf.compat.v1.placeholder(tf.float32, shape=[None])
    x_boundary_xmin = tf.compat.v1.placeholder(tf.float32, shape=[None])
    t_boundary_xmax = tf.compat.v1.placeholder(tf.float32, shape=[None])
    x_boundary_xmax = tf.compat.v1.placeholder(tf.float32, shape=[None])


# Create tf graphs for calculating stuff
u_init_pred = u(t_init, x_init)
u_boundary_xmin_pred = u(t_boundary_xmin, x_boundary_xmin)
u_boundary_xmax_pred = u(t_boundary_xmax, x_boundary_xmax)
f_pred = f(t_int, x_int)


# Define initial conditions
def u_0(x):
    return -np.sin(np.pi*((2.*(x-x_min)/(x_max-x_min))-1.))

# Define loss function and optimization algorithm.
#To-do

initial_loss = tf.reduce_mean(tf.square(u_init_pred - u_init))
interior_loss = tf.reduce_mean(tf.square(f_pred))
boundary_loss = tf.reduce_mean(tf.square(u_boundary_xmin_pred - u_boundary_xmax_pred))

loss = initial_loss + interior_loss + boundary_loss

# optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss,
#                                                    method='L-BFGS-B',
#                                                    options={'maxiter':40000,
#                                                             'maxfun':100000,
#                                                             'maxcor': 50,
#                                                             'maxls': 50,
#                                                             'ftol': 1.0*np.finfo(float).eps})
optimizer = tf.train.AdamOptimizer(learning_rate=1e-1)
opt = optimizer.minimize(loss, var_list=tf.trainable_variables())

# Generate training / regularization data
#To-do
N_t = 100
N_x = 100
N_consistency = 1000
t = np.linspace(t_min, t_max, N_t)
x = np.linspace(x_min, x_max, N_x)

#T, X = np.meshgrid(t, x)
## What is X_star???
## Has shape (N_pred ** 3, 3)
#X_star = np.hstack((T.flatten()[:, None], X.flatten()[:, None]))
## Domain bounds for t, x, and v. Equivelant to taking +/- of largest {t,x,v}_max value (I think)
#lb = X_star.min(0)
#ub = X_star.max(0)

lb = np.array([t_min, x_min])
ub = np.array([t_max, x_max])

# Train

# stack of t, x for each sample
X_initial_train = np.stack((np.zeros(N_x), # t for each sample
                            np.linspace(x_min, x_max, N_x))) # x for each sample
Y_initial_train = u_0(X_initial_train[1,:])

X_interior_train = lb + (ub - lb)*lhs(2, N_consistency)

tf_dict = {t_init: X_initial_train[0,:],
           x_init: X_initial_train[1,:],
           u_init: Y_initial_train,
           t_int: X_interior_train[:, 0],
           x_int: X_interior_train[:, 1],
           t_boundary_xmin: t,
           x_boundary_xmin: x_min*np.ones(N_t),
           t_boundary_xmax: t,
           x_boundary_xmax: x_max*np.ones(N_t)}

# Initialize variables, create session:
sess = tf.compat.v1.Session()
init = tf.compat.v1.global_variables_initializer()
sess.run(init)

for i in range(1000):
    _, loss_num = sess.run([opt, loss], feed_dict=tf_dict)
    if i % 10 == 0:
        print("Epoch:{} Loss: {}".format(i, loss_num))
    # optimizer.minimize(sess, feed_dict=tf_dict,
    #                    fetches=[loss, interior_loss, initial_loss, boundary_loss])


# Display

# Do a simple forward pass
u_init_pred_result = sess.run(u_init_pred, {x_init:x, t_init:np.zeros(N_x)})
u_pred_result = sess.run(u_init_pred, {x_init:x, t_init:0.3*np.ones(N_x)})

plt.plot(x, u_init_pred_result, label="NN")
plt.plot(x, u_0(x), label="Initial Condition")
plt.plot(x, u_pred_result, label="NN at t=0.3")
plt.legend()
plt.show()

