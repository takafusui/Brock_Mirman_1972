#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
"""
Filename: BM1972.py
Author(s): Takafumi Usui
E-mail: u.takafumi@gmail.com
Description:
Brock and Mirman (1972) in JET
"""
import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print(r"Version of Tensorflow is {}".format(tf.__version__))

# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #
_alpha = 0.4  # Capital share
_A = 1  # Technology level
_beta = 0.96  # Discount factor

with tf.name_scope('econ_params'):
    alpha = tf.constant(_alpha, dtype=tf.float32, name='alpha')
    A = tf.constant(_A, dtype=tf.float32, name='_A')
    beta = tf.constant(_beta, dtype=tf.float32, name='beta')


# --------------------------------------------------------------------------- #
# DNN structure
# --------------------------------------------------------------------------- #
with tf.name_scope('neural_net'):
    num_input = 1  # There is only one state variable: capital stock
    # Two outputs: capital stock tomorrow and the Lagrange multiplier
    num_output = 2
# --------------------------------------------------------------------------- #
# Create placeholder
# --------------------------------------------------------------------------- #
def create_placeholders(n_x):
    "Create the placeholders for the tensorflow session"
    X = tf.placeholder(tf.float32, shape=[n_x, None], name='X')
    return X

# --------------------------------------------------------------------------- #
# Initialize the NN parameters
# --------------------------------------------------------------------------- #
def initialize_NN_parameters(layers_dim):
    "Initialize parameters to build a neural network with tensorflow"

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

# Test
layers_dim = [1, 100, 100, 2]
tf.reset_default_graph()
with tf.Session() as sess:
    parameters = initialize_NN_parameters(layers_dim)
print(r'Parameter W1 is {}'.format(parameters['W1']))
print(r'Parameter b1 is {}'.format(parameters['b1']))
print(r'Parameter W2 is {}'.format(parameters['W2']))
print(r'Parameter b2 is {}'.format(parameters['b2']))
print(r'Parameter W3 is {}'.format(parameters['W3']))
print(r'Parameter b3 is {}'.format(parameters['b3']))

# --------------------------------------------------------------------------- #
# Forward propagation in tensorflow
# --------------------------------------------------------------------------- #
def forward_propagation(X, parameters):
    """ Implement the forward propagation for the model
    Linear -> RELU -> Linear -> RELU -> Linear -> Softplus """

    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    Z1 = tf.add(tf.matmul(W1, X), b1)
    A1 = tf.nn.relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)
    A2 = tf.nn.relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)
    A3 = tf.nn.softplus(Z3)

    return A3

# Test
tf.reset_default_graph()
with tf.Session() as sess:
    X = create_placeholders(layers_dim[0])
    parameters = initialize_NN_parameters(layers_dim)
    A3 = forward_propagation(X, parameters)
    print("A3 = {}".format(A3))


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #
def simulate(k0, T_length, parameters):
    """ Simulate T periods economic paths
    k0: Initial state
    T_length: Length of the simulated time period
    parameters: DNN parameters """

    path = np.empty((2, T), dtype=np.float32)
    path[0, 0] = k0  # Initialize the given capital stock

    for t in range(0, T_length):
        path[1, ]
sys.exit(0)
# --------------------------------------------------------------------------- #
# Cost function
# --------------------------------------------------------------------------- #
# def compute_cost():
#     """ Loss function, which is the mean of the sum of the squared relative 
#     Euler error and the squared KKT error
#     err_REE: relative Euler error
#     err_KKT: KKT error """




# # --------------------------------------------------------------------------- #
# # Build the DNN model
# # --------------------------------------------------------------------------- #
# def model(learning_rate=0.0001, num_epochs=10, epsilon_tol=1e-3,
#           print_cost=True)
# --------------------------------------------------------------------------- #
# Analytical solution is available
# --------------------------------------------------------------------------- #
# krange = np.linspace(1e-3, 10, 250)
# krange = tf.convert_to_tensor(krange, dtype=tf.float32)

# def kplus_analytic(k, beta, alpha, A):
#     """ Optimal capital stock in the next period following the analytically
#     derived rule"""
#     _k = tf.placeholder(tf.float32)
#     _kplus = tf.placeholder(tf.float32)
#     _beta = tf.placeholder(tf.float32)
#     _alpha = tf.placeholder(tf.float32)
#     _A = tf.placeholder(tf.float32)
#     with tf.Session() as sess:
#         _kplus = sess.run(
#             tf.math.multiply(
#                 tf.math.multiply(_beta, _alpha),
#                 tf.math.multiply(_A, tf.math.pow(_k, _alpha))),
#             feed_dict={_k: k, _beta: beta, _alpha: alpha})
#     return _kplus


# plt.plot(krange, kplus_analytic(krange, beta, alpha, A))
# plt.show()
    
