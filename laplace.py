# Solution of the Laplace equation using physics-informed NNs.
# Problem setup: no charges; a unit box with a single plate at a finite potential
#   and the others at 0.

import tensorflow as tf  # Using TF2
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# If GPU, set it so TensorFlow doesn't suck up all the memory from the get-go
physical_devices = tf.config.list_physical_devices('GPU')
try:
    for gpu in physical_devices:
        tf.config.experimental.set_memory_growth(gpu, True)
except:
    pass


# ---------- Hyperparams ---------- #
# These are just guesses for reasonable values
# Data params
num_interior_pts = 2000  # Number of points to sample in the interior of the box
num_bound_pts = 500  # Number of points to sample on each boundary
# Training params
num_epochs = 10001  # Number of training iterations to go through.
learning_rate = 1e-3  # Uhhhhh learning rate.
solution_weight = 1.0  # The weight of the error of the Laplace equation.
bound_weight = 1.0  # The weight of the error of the boundary conditions.


# ---------- Make sample points and values ---------- #
# Box bounds are 0 to 1
# Points to sample at given as x,y,z coordinates
interior_pts = tf.Variable(np.random.rand(200, 3), dtype=tf.float32)

# Boundary condition sample points. Positive is towards positive inf and negative is the opposite.
# Fix one component at 0 or 1, set the rest of the components to random values.
bound_x_neg_pts = tf.Variable(np.insert(np.random.rand(num_bound_pts, 2), 0, np.zeros((num_bound_pts)), axis=1))
bound_x_pos_pts = tf.Variable(np.insert(np.random.rand(num_bound_pts, 2), 0, np.ones((num_bound_pts)), axis=1))
bound_y_neg_pts = tf.Variable(np.insert(np.random.rand(num_bound_pts, 2), 1, np.zeros((num_bound_pts)), axis=1))
bound_y_pos_pts = tf.Variable(np.insert(np.random.rand(num_bound_pts, 2), 1, np.ones((num_bound_pts)), axis=1))
bound_z_neg_pts = tf.Variable(np.insert(np.random.rand(num_bound_pts, 2), 2, np.zeros((num_bound_pts)), axis=1))
bound_z_pos_pts = tf.Variable(np.insert(np.random.rand(num_bound_pts, 2), 2, np.ones((num_bound_pts)), axis=1))

# Negative-side X plate (x=0) is at +1V, 0 on all other plates.
bound_x_neg = 1
bound_x_pos = 0
bound_y_neg = 0
bound_y_pos = 0
bound_z_neg = 0
bound_z_pos = 0


# ---------- Build the model/solution to Laplace's equation ---------- #
optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

# We parameterize the solution to the Laplace equation, f, as a dense neural network.
inputs = keras.Input(shape=(3,))
dense1 = keras.layers.Dense(32, activation=tf.nn.elu)  # I like elu :)
dense2 = keras.layers.Dense(32, activation=tf.nn.elu)
dense3 = keras.layers.Dense(1)
output = dense3(dense2(dense1(inputs)))
# This "model" is 'f' because it takes in x, y, z and spits out a value.
model = keras.Model(inputs=inputs, outputs=output)


# ---------- Train ---------- #
# Fancy pants python decorator magic that speeds up computation by like an order of magnitude.
@tf.function
def trainfunc():
    # The gradient tape basically keeps tracks of the variables so it can reverse-autodiff
    # I don't really quite fully understand it this gradient tape business yet, but it works.
    with tf.GradientTape() as lossTape:
        # These are the interior sample points. Explicitly state end of slice to keep dimensions.
        x = interior_pts[:, 0:1]
        y = interior_pts[:, 1:2]
        z = interior_pts[:, 2:3]
        # Need persistent = true on this one so the gradients aren't wiped after the first
        #   call to div_tape.gradient (I think? Something like that)
        # I think TF caches the graph defined here for performance reasons. I don't
        #   think it constructs it each run of the loop.
        with tf.GradientTape(persistent=True) as div_tape:
            # Need to "watch" these tensors so the gradient tape remembers to keep track of them.
            #   Otherwise (I think) they'll be crunched down for optimization purposes. You don't
            #   need to watch variables though -- those are kept track of automagically.
            div_tape.watch([x, y, z])
            with tf.GradientTape() as grad_tape:
                # Each gradient tape needs it's own watch function from what I gather.
                grad_tape.watch([x, y, z])
                # Actually evaluate f / the model on the interior points.
                f = model(tf.concat([x, y, z], axis=1))
            df_dx, df_dy, df_dz = grad_tape.gradient(f, [x, y, z])
        d2f_dx2 = div_tape.gradient(df_dx, x)
        d2f_dy2 = div_tape.gradient(df_dy, y)
        d2f_dz2 = div_tape.gradient(df_dz, z)
        laplacian = d2f_dx2 + d2f_dy2 + d2f_dz2

        # The Laplacian of the potential (f, in this case) should equal 0 because there is 0 charge
        #   in the interior of the box.
        solution_loss = tf.math.reduce_mean(tf.math.square(laplacian))

        # Calculate the error of the calculated potential at the box boundaries.
        bound_x_neg_loss = tf.math.reduce_mean(tf.math.square(model(bound_x_neg_pts) - bound_x_neg))
        bound_x_pos_loss = tf.math.reduce_mean(tf.math.square(model(bound_x_pos_pts) - bound_x_pos))
        bound_y_neg_loss = tf.math.reduce_mean(tf.math.square(model(bound_y_neg_pts) - bound_y_neg))
        bound_y_pos_loss = tf.math.reduce_mean(tf.math.square(model(bound_y_pos_pts) - bound_y_pos))
        bound_z_neg_loss = tf.math.reduce_mean(tf.math.square(model(bound_z_neg_pts) - bound_z_neg))
        bound_z_pos_loss = tf.math.reduce_mean(tf.math.square(model(bound_z_pos_pts) - bound_z_pos))
        bound_total_loss = (bound_x_pos_loss + bound_x_neg_loss +
                            bound_y_pos_loss + bound_y_neg_loss +
                            bound_z_pos_loss + bound_z_neg_loss)

        # Behold, my PhD, a
        total_loss = solution_loss * solution_weight + bound_total_loss * bound_weight

    # Calculate the gradients of the loss with respect to all trainable parameters in here (this
    #   includes backpropagating through the Laplace operator too).
    grads = lossTape.gradient(total_loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    return total_loss


for i in range(num_epochs):
    total_loss = trainfunc()

    if i % 100 == 0:
        print("Epoch: {}\tLoss: {}".format(i, total_loss))

gridsize = 100
# Save plots of final solution to the Laplace equation.
# x-y plane along z=0.5:
mesh = np.meshgrid(np.linspace(0, 1, gridsize), np.linspace(0, 1, gridsize))
coords = np.concatenate((mesh[0].reshape(-1, 1), mesh[1].reshape(-1, 1)), axis=1)
coords = np.concatenate((coords, np.ones((gridsize * gridsize, 1)) * 0.5), axis=1)
img = model(coords)
plt.figure()
plt.title("x-y plane at z=0.5")
plt.imshow(img.numpy().reshape(gridsize, gridsize), vmin=0, vmax=1, extent=(0, 1, 0, 1))
plt.colorbar()
plt.savefig("output-xy.png")

# y-z plane along x=0.5:
coords = np.concatenate((mesh[0].reshape(-1, 1), mesh[1].reshape(-1, 1)), axis=1)
coords = np.concatenate((np.ones((gridsize * gridsize, 1)) * 0.5, coords), axis=1)
img = model(coords)
plt.figure()
plt.title("y-z plane at x=0.5")
plt.imshow(img.numpy().reshape(gridsize, gridsize), vmin=0, vmax=1, extent=(0, 1, 0, 1))
plt.colorbar()
plt.savefig("output-yz.png")
