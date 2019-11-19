#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
"""
Filename: Brock_Mirman_1972.py
Author(s): Takafumi Usui
E-mail: u.takafumi@gmail.com
Description:
Brock and Mirman (1972) in JET
"""
import sys
import tensorflow as tf
from dnn_utils import random_mini_batches_X
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
# Use TeX font
rc('font', **{'family': 'sans-serif', 'serif': ['Helvetica']})
rc('text', usetex=True)
plt.rcParams["font.size"] = 15

print(r"Version of Tensorflow is {}".format(tf.__version__))

# --------------------------------------------------------------------------- #
# Parameter setting
# --------------------------------------------------------------------------- #
A = 1  # Technology level
alpha = 0.35  # Capital share in the Cobb-Douglas production function
beta = 0.98  # Discount factor


# --------------------------------------------------------------------------- #
# Analytical solution
# --------------------------------------------------------------------------- #
def k_compute_infty(alpha, beta, A):
    """ Return the stationary point (or steady state) """
    return (1 / (beta * alpha * A))**(1/(alpha - 1))


k_infty = k_compute_infty(alpha, beta, A)
print("Stationary point is {:5f}".format(k_infty))


def k_plus_analytic(k, alpha, beta, A):
    """ Analytical solution
    Return the optimal capital stock in the next period """
    return alpha * beta * A * k**alpha


def consum(k, kplus, alpha, beta, A):
    """ Return the optimal consumption policy """
    return A*k**alpha - kplus


# sys.exit(0)
# --------------------------------------------------------------------------- #
# Hyper parameters
# --------------------------------------------------------------------------- #
# Layer setting
num_input = 1
num_hidden1 = 100
num_hidden2 = 100
num_output = 2
layers_dim = [num_input, num_hidden1, num_hidden2, num_output]
print("Dimensions of each layer are {}".format(layers_dim))

learning_rate = 0.001  # Leargning rate
t_length = 10000  # Simulation length

# Number of episodes, epochs and minibatch size
num_episodes = 100
num_epochs = 10
minibatch_size = 64


# sys.exit(0)
# --------------------------------------------------------------------------- #
# Create placeholders, initialize parameters, forward propagation
# --------------------------------------------------------------------------- #
def create_placeholders(num_x):
    """ Create the placeholders.
    The column dimention is None that represents the length of the simulated
    path."""
    X = tf.placeholder(tf.float32, shape=[num_x, None], name='X')
    return X


def initialize_parameters(layers_dim):
    """ Initialize parameters to build a neural network
    1: [num_input, None] -> [num_hidden1, None]
    2: [num_hidden1, None] -> [num_hidden2, None]
    3: [num_hidden2, None] -> [num_output, None] """

    W1 = tf.get_variable('W1', [layers_dim[1], layers_dim[0]], tf.float32,
                         tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable('b1', [layers_dim[1], 1], tf.float32,
                         tf.zeros_initializer())
    W2 = tf.get_variable('W2', [layers_dim[2], layers_dim[1]], tf.float32,
                         tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable('b2', [layers_dim[2], 1], tf.float32,
                         tf.zeros_initializer())
    W3 = tf.get_variable('W3', [layers_dim[3], layers_dim[2]], tf.float32,
                         tf.contrib.layers.xavier_initializer())
    b3 = tf.get_variable('b3', [layers_dim[3], 1], tf.float32,
                         tf.zeros_initializer())

    parameters = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'W3': W3, 'b3': b3}
    return parameters


def forward_propagation(X, parameters):
    """ Implement the forward propagation for the model
    Linear -> Relu -> Linear -> Relu -> Linear -> Softplus """

    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    Z1 = tf.add(tf.matmul(W1, X), b1)  # Linear combination
    A1 = tf.nn.relu(Z1)  # Activate with Relu
    Z2 = tf.add(tf.matmul(W2, A1), b2)  # Linear combination
    A2 = tf.nn.relu(Z2)  # Activate with Relu
    Z3 = tf.add(tf.matmul(W3, A2), b3)  # Linear combination
    A3 = tf.nn.softplus(Z3)  # Activate with Relu

    return A3


# --------------------------------------------------------------------------- #
# Cost function
# --------------------------------------------------------------------------- #
def compute_cost(X, A3, parameters, beta, A, alpha):
    """ Compute the mean squared error
    errREE: relative Euler error
    errKKT: error in the KKT complementarity equation """

    # Retlieve the current capital state
    k = X

    # Retlieve the simulation path
    # Need to have a dimension of [num_x, None] to have compativility with the
    # placeholder X
    k_plus = tf.expand_dims(A3[0, :], axis=0)
    lambd = tf.expand_dims(A3[1, :], axis=0)

    # Retlieve the optimal actions in the next period
    action_plus = forward_propagation(k_plus, parameters)
    # Need to have a dimension of [num_x, None] to have compativility with the
    # placeholder X
    k_plusplus = tf.expand_dims(action_plus[0, :], axis=0)
    lambd_plus = tf.expand_dims(action_plus[1, :], axis=0)

    # Define the relative Euler error
    errREE = (beta * lambd_plus * A * alpha * k_plus**(alpha-1)) / lambd - 1

    # Define the KKT error
    errKKT = lambd * (A * k**alpha - k_plus - 1/lambd)

    # Stack two approximation errors
    err = tf.stack([errREE[0], errKKT[0]])

    # When the two constraints are exactly binding
    err_optimal = tf.zeros_like(err)

    # Define the cost function
    cost = tf.losses.mean_squared_error(err, err_optimal)

    return cost


# --------------------------------------------------------------------------- #
# Define DNN
# --------------------------------------------------------------------------- #
def model(train_X, layers_dim, learning_rate, num_epochs, minibatch_size,
          print_span, print_cost=True):
    """ Train DNN with the provided training data """

    tf.reset_default_graph()  # Reset DNN
    (num_input, t_length) = train_X.shape
    costs = []  # Keep track of the cost

    # Create placeholder
    X = create_placeholders(num_input)
    # Initialize parameters
    parameters = initialize_parameters(layers_dim)
    # One step of the forward propagation
    A3 = forward_propagation(X, parameters)
    # Compute the current cost
    cost = compute_cost(X, A3, parameters, beta, A, alpha)
    # Set the optimizer in the backward propagation
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
        cost)
    # Initialize all the variables
    init = tf.global_variables_initializer()

    # ----------------------------------------------------------------------- #
    # Start to train DNN with random minibatches
    # ----------------------------------------------------------------------- #
    with tf.Session() as sess:
        sess.run(init)  # Initialization

        for epoch in range(1, num_epochs+1):
            epoch_cost = 0  # Initialize and track the cost for each epoch
            # Number of minibatch size
            num_minibatches = int(t_length / minibatch_size)
            minibatches = random_mini_batches_X(train_X, minibatch_size)

            for minibatch in minibatches:
                minibatch_X = minibatch
                _, minibatch_cost = sess.run(
                    [optimizer, cost], feed_dict={X: minibatch_X})
                epoch_cost += minibatch_cost / num_minibatches

            # Track the cost
            costs.append(epoch_cost)

            # Print the cost every epoch
            if print_cost is True and epoch % print_span == 0:
                print(r'Cost after iteration {} is {:.3e}'.format(
                    epoch, epoch_cost))

        # Save the parameters in a variable
        parameters = sess.run(parameters)
        print("Parameters have been trained")

        return costs, parameters


# --------------------------------------------------------------------------- #
# Training data
# --------------------------------------------------------------------------- #
kbeg, kend, ksize = 1e-6, 0.25, 250
kinit = (kbeg + kend) / 2
kgrid = np.linspace(kbeg, kend, ksize, dtype=np.float32)
train_X = np.random.uniform(kbeg, kend, (num_input, t_length))
print("train_X has a shape of {}".format(train_X.shape))

# --------------------------------------------------------------------------- #
# Plot the optimal policy function, cost etc.
# --------------------------------------------------------------------------- #
costs, parameters_star = model(
    train_X, layers_dim, learning_rate, num_epochs=1000,
    minibatch_size=minibatch_size, print_span=50, print_cost=True)

with tf.Session() as sess:
    # One step of the forward propagation
    policy_star = sess.run(forward_propagation(
        kgrid.reshape(1, ksize), parameters_star))

# Capital stock in the next period
k_plus_analytic_path = k_plus_analytic(kgrid, alpha, beta, A)
# Consumption policy
c_analytic = consum(kgrid, k_plus_analytic_path, alpha, beta, A)
c_dnn = consum(kgrid, policy_star[0, :], alpha, beta, A)

# Capital stock tomorrow
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
axes[0].plot(kgrid, policy_star[0, :], 'k-', label="DNN")
axes[0].plot(kgrid, k_plus_analytic_path, 'k--', label="Analytic")
axes[0].plot(kgrid, kgrid, 'k:')
axes[0].set_xlabel(r"$k_{t}$")
axes[0].set_ylabel(r"$k_{t+1}$")
axes[0].set_xlim([kbeg, kend])
axes[0].set_ylim([0, kend])
axes[0].legend(loc='best')

axes[1].plot(kgrid, c_dnn, 'k-', label="DNN")
axes[1].plot(kgrid, c_analytic, 'k--', label="Analytic")
axes[1].set_xlabel(r"$k_{t}$")
axes[1].set_ylabel(r"$c_{t}$")
axes[1].set_xlim([kbeg, kend])
axes[1].set_ylim([0, None])
axes[1].legend(loc='best')

plt.savefig('policies_training.pdf')
plt.close()

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(np.log10(costs), 'k-')
ax.set_xlabel("Number of epochs")
ax.set_ylabel("$log_{10}$ of the cost")
ax.set_xlim([0, None])

plt.savefig('cost_training.pdf')
plt.close()


# sys.exit(0)
# --------------------------------------------------------------------------- #
# Define DNN with sampling the most relevant states
# --------------------------------------------------------------------------- #
def model_sampling(
        init_state, num_input, t_length, layers_dim, learning_rate,
        num_episodes, num_epochs, minibatch_size, print_span, print_cost=True):
    """ Train DNN with sampling the most relevant states """

    tf.reset_default_graph()  # Reset DNN

    costs = []  # Keep track of the cost

    episodic_costs = []  # Keep track of the episodic cost

    # Create placeholder
    X = create_placeholders(num_input)
    # Initialize parameters
    parameters = initialize_parameters(layers_dim)
    # One step of the forward propagation
    A3 = forward_propagation(X, parameters)
    # Compute the current cost
    cost = compute_cost(X, A3, parameters, beta, A, alpha)
    # Set the optimizer in the backward propagation
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
        cost)
    # Initialize all the variables
    init = tf.global_variables_initializer()

    # ----------------------------------------------------------------------- #
    # Start to train DNN with sampling and random minibatches
    # ----------------------------------------------------------------------- #
    with tf.Session() as sess:
        sess.run(init)  # Initialization

        for episode in range(1, num_episodes+1):
            episodic_cost = 0
            # --------------------------------------------------------------- #
            # Simulate one training path
            # --------------------------------------------------------------- #
            # Training set, which samples the most relevent states
            # Keep compativility with placeholder X that has [num_input. None]
            train_X = np.empty((num_input, t_length), dtype=np.float32)
            # Retlieve the initial state from the previous episode
            train_X[0, 0] = init_state
            for t in range(1, t_length):
                # Keep the same shape with X ([num_input, None])
                x_previous = train_X[0, t-1].reshape((num_input, 1))
                train_X[0, t] = sess.run(A3, feed_dict={X: x_previous})[0]

            # --------------------------------------------------------------- #
            # Train DNN with the sampled state path
            # --------------------------------------------------------------- #
            for epoch in range(1, num_epochs+1):
                epoch_cost = 0  # Initialize and trach the cost for each epoch
                # Number of minibatch size
                num_minibatches = int(t_length / minibatch_size)
                minibatches = random_mini_batches_X(train_X, minibatch_size)

                for minibatch in minibatches:
                    minibatch_X = minibatch
                    _, minibatch_cost = sess.run(
                        [optimizer, cost], feed_dict={X: minibatch_X})
                    epoch_cost += minibatch_cost / num_minibatches

                # Track the cost
                costs.append(epoch_cost)

                # Print the cost every epoch
                if print_cost is True and epoch % print_span == 0:
                    print(
                        r'Cost after epoch {} in episode {} is {:.3e}'.format(
                            epoch, episode, epoch_cost))
                episodic_cost += epoch_cost / num_epochs
            # Track the episodic cost
            episodic_costs.append(episodic_cost)

            # Trach the last state for the next episode
            init_state = train_X[0, -1]
            train_X_previous = train_X

        # Save the parameters in a variable
        parameters = sess.run(parameters)

        return costs, episodic_costs, parameters, train_X, train_X_previous


# --------------------------------------------------------------------------- #
# Simulate the model with sampling
# --------------------------------------------------------------------------- #
costs_sampling, episodic_costs, parameters_sampling, train_X, \
    train_X_previous = model_sampling(
        kinit, num_input, t_length, layers_dim, learning_rate, num_episodes,
        num_epochs, minibatch_size, print_span=5, print_cost=True)


fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(train_X_previous[0, :], train_X[0, :], 'o', markersize=3, label='DNN')
ax.plot(k_infty, k_infty, 'o', c='red', markersize=5, label='Stationary')
ax.set_xlim([0.19, 0.20])
ax.set_ylim([0.19, 0.20])
ax.set_xlabel(r"$k_{t}$")
ax.set_ylabel(r"$k_{t+1}$")

ax.legend(loc='best')
plt.savefig('k_kplus_sampling.pdf')
plt.close()

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(np.log10(episodic_costs), 'k-')
ax.set_xlabel("Number of episodes")
ax.set_ylabel("$log_{10}$ of the cost")
ax.set_xlim([0, None])

plt.savefig('episodic_costs_sampling.pdf')
plt.close()
