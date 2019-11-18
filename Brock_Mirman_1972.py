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

print(r"Version of Tensorflow is {}".format(tf.__version__))

# --------------------------------------------------------------------------- #
# Parameter setting
# --------------------------------------------------------------------------- #
A = 1  # Technology level
alpha = 0.35 # Capital share in the Cobb-Douglas production function
beta = 0.98  # Discount factor


# --------------------------------------------------------------------------- #
# Analytical solution
# --------------------------------------------------------------------------- #
def k_compute_infty(alpha, beta, A):
    """ Return the stationary point (or steady state) """
    return (1 / (beta * alpha * A))**(1/(alpha - 1))

k_infty = k_compute_infty(alpha, beta, A)
print("Stationary point is {}".format(k_infty))


def k_plus_analytic(k, alpha, beta, A):
    """ Analytical solution
    Return the optimal capital stock in the next period """
    return alpha * beta * A * k**alpha


# Setup the capital grid
kbeg, kend, ksize = 1e-3, 3, 250
kgrid = np.linspace(kbeg, kend, ksize, dtype=np.float32)

# Capital stock in the next period
k_plus = k_plus_analytic(kgrid, alpha, beta, A)

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
t_lentgh = 10000  # Simulation length

num_epochs = 2500
minibatch_size = 64
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
    Linear -> RELU -> Linear -> RELU -> Linear -> Softplus """

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
    
    # Current capital state
    k = X
    
    # Retlieve the simulation path
    k_plus = tf.expand_dims(A3[0, :], axis=0)
    lambd = tf.expand_dims(A3[1, :], axis=0)
    
    # Retlieve the optimal actions in the next period
    action_plus = forward_propagation(k_plus, parameters)
    k_plusplus = tf.expand_dims(action_plus[0, :], axis=0)
    lambd_plus = tf.expand_dims(action_plus[1, :], axis=0)
    
    # Define the relative Euler error
    errREE = (beta * lambd_plus * A * alpha * k_plus**(alpha-1)) / lambd - 1

    # Define the KKT error
    errKKT = lambd * (A * k**alpha - k_plus - 1/lambd)

    # Stack two approximation errors
    err = tf.stack([errREE[0], errKKT[0]])

    # The two constraints are exactly binding
    err_optimal = tf.zeros_like(err)
    
    # Define the cost function
    cost = tf.losses.mean_squared_error(err, err_optimal)
    
    return cost

# --------------------------------------------------------------------------- #
# Training data
# --------------------------------------------------------------------------- #
train_X = np.random.uniform(kbeg, kend, (num_input, t_lentgh))
print("train_X has a shape of {}".format(train_X.shape))

# sys.exit(0)
# --------------------------------------------------------------------------- #
# Define DNN
# --------------------------------------------------------------------------- #
def model(train_X, simulate_X, layers_dim, learning_rate, num_epochs,
          minibatch_size, print_span, print_cost=True):
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

    # ---------------------------------------------------------------------- #
    # Start to train DNN with random minibatches
    # ---------------------------------------------------------------------- #   
    with tf.Session() as sess:
        sess.run(init)  # Initialization

        for epoch in range(1, num_epochs+1):
            epoch_cost = 0  # Initialize and trach the cost for each epoch
            # Number of minibatch size
            num_minibatches = int(t_length / minibatch_size)
            minibatches = random_mini_batches_X(train_X, minibatch_size)

            for minibatch in minibatches:
                minibatch_X = minibatch
                _, minibatch_cost = sess.run([optimizer, cost],
                                             feed_dict={X: minibatch_X})
                epoch_cost += minibatch_cost / num_minibatches

            # Track the cost
            costs.append(epoch_cost)

            # Print the cost every epoch
            if print_cost is True and epoch % print_span == 0:
                print(r'Cost after iteration {} is {:5f}'.format(
                    epoch, epoch_cost))
                
        # Save the parameters in a variable
        parameters = sess.run(parameters)
        print("Parameters have been trained")

        action_star = sess.run(
            forward_propagation(simulate_X.reshape(1, 250), parameters))
        
        return costs, parameters, action_star


# --------------------------------------------------------------------------- #
# Simulate the model
# --------------------------------------------------------------------------- #
costs, parameters_star, action_star = model(
    train_X, kgrid, layers_dim, learning_rate=learning_rate,
    num_epochs=num_epochs, minibatch_size=minibatch_size,
    print_span=10, print_cost=True)


# --------------------------------------------------------------------------- #
# Simulate the model
# --------------------------------------------------------------------------- #
fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(kgrid, k_plus, 'k:', label="Analytic")
ax.plot(kgrid, action_star[0, :], 'k-', label="DNN")
ax.set_xlabel(r'$k_{t}$')
ax.set_ylabel(r'$k_{t+1}$')
ax.set_xlim([kbeg, kend])
ax.set_ylim([0, None])
plt.legend(loc='best')
plt.savefig('k_plus.pdf')
plt.close()

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(costs, 'k-')
ax.set_xlabel(r'Number of iterations')
ax.set_ylabel(r'Cost')
ax.set_xlim([0, None])
ax.set_ylim([0, None])
plt.savefig('cost.pdf')
plt.close()
