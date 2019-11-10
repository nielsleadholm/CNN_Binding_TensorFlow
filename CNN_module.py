#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.python import debug as tf_debug
import numpy as np
import math
from keras.datasets import mnist, fashion_mnist, cifar10
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import mltest
from PIL import Image
import os
import tfCore_adversarial_attacks as atk


# #Temporarily disable deprecation warnings (using tf 1.14)
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 


def data_setup(params):
    #Note the shape of the images required by the custom CNNs is 2D, rather than flattened as for the Madry model
    if params['dataset'] == 'mnist':
        (training_data, training_labels), (testing_data, testing_labels) = mnist.load_data()
        training_data = np.reshape(training_data, [np.shape(training_data)[0], 28, 28, 1])
        testing_data = np.reshape(testing_data, [np.shape(testing_data)[0], 28, 28, 1])

    elif params['dataset'] == 'fashion_mnist':
        (training_data, training_labels), (testing_data, testing_labels) = fashion_mnist.load_data()
        training_data = np.reshape(training_data, [np.shape(training_data)[0], 28, 28, 1])
        testing_data = np.reshape(testing_data, [np.shape(testing_data)[0], 28, 28, 1])

    elif params['dataset'] == 'cifar10':
        (training_data, training_labels), (testing_data, testing_labels) = cifar10.load_data()
    
    #Rescale images to values between 0:1 and reshape so each image is 28x28
    training_data = training_data/255
    testing_data = testing_data/255

    #Transform the labels into one-hot encoding
    training_labels = to_categorical(training_labels)
    testing_labels = to_categorical(testing_labels)

    if params['crossval_bool'] == True:
        crossval_data = training_data[-10000:]
        crossval_labels = training_labels[-10000:]
        training_data = training_data[0:-10000]
        training_labels = training_labels[0:-10000]

    else:
        crossval_data = None
        crossval_labels = None

    return (training_data, training_labels, testing_data, testing_labels, crossval_data, crossval_labels)

#Define a summary variables function for later visualisation of the network
def var_summaries(variable):
    with tf.name_scope('Summaries'):
        mean = tf.reduce_mean(variable)
        tf.compat.v1.summary.scalar('Mean', mean) #The tf.summary operation determines which graph node you would like to annotate, and scalar or histogram the type of summary

        with tf.name_scope('STD'):
            std = tf.sqrt(tf.reduce_mean(tf.square(variable - mean)))

        tf.compat.v1.summary.scalar('STD', std)
        tf.compat.v1.summary.scalar('Max', tf.reduce_max(variable))
        tf.compat.v1.summary.scalar('Min', tf.reduce_min(variable))
        tf.compat.v1.summary.histogram('Histogram', variable)

def initializer_fun(params, training_data, training_labels):

    tf.compat.v1.reset_default_graph() #Re-set the default graph to clear previous e.g. variable assignments

    dropout_rate_placeholder = tf.compat.v1.placeholder(tf.float32)
    initializer = tf.contrib.layers.variance_scaling_initializer()
    y = tf.compat.v1.placeholder(training_labels.dtype, [None, 10], name='y-input')

    if (params['dataset'] == 'mnist') or (params['dataset'] == 'fashion_mnist'): #Define core variables for a LeNet-5 architecture for MNIST/Fashion

        x = tf.compat.v1.placeholder(training_data.dtype, [None, 28, 28, 1], name='x-input')
        
        with tf.compat.v1.variable_scope(params['architecture']):
        #Note for example that the first convolutional weights layer has a 5x5 filter with 1 input channel, and 6 output channels
        #tf.compat.v1.get_variable will either get an existing variable with these parameters, or otherwise create a new one
            weights = {
            'conv_W1' : tf.compat.v1.get_variable('CW1', shape=(5, 5, 1, 6), initializer=initializer),
            'conv_W2' : tf.compat.v1.get_variable('CW2', shape=(5, 5, 6, 16), initializer=initializer),
            'dense_W1' : tf.compat.v1.get_variable('DW1', shape=(400, 120), initializer=initializer),
            'dense_W2' : tf.compat.v1.get_variable('DW2', shape=(120, 84), initializer=initializer),
            'output_W' : tf.compat.v1.get_variable('OW', shape=(84, 10), initializer=initializer)
            }
            if (params['architecture'] == 'BindingCNN') or (params['architecture'] == 'controlCNN'):
                weights['course_bindingW1'] = tf.compat.v1.get_variable('courseW1', shape=(1600, 120), initializer=initializer)
                weights['finegrained_bindingW1'] = tf.compat.v1.get_variable('fineW1', shape=(1176, 120), initializer=initializer)

            #Add summaries for each weightseight variable in the dictionary, for later use in TensorBoard
            for weights_var in weights.values():
                var_summaries(weights_var)

            biases = {
            'conv_b1' : tf.compat.v1.get_variable('Cb1', shape=(6), initializer=initializer),
            'conv_b2' : tf.compat.v1.get_variable('Cb2', shape=(16), initializer=initializer),
            'dense_b1' : tf.compat.v1.get_variable('Db1', shape=(120), initializer=initializer),
            'dense_b2' : tf.compat.v1.get_variable('Db2', shape=(84), initializer=initializer),
            'output_b' : tf.compat.v1.get_variable('Ob', shape=(10), initializer=initializer)
            }

            for biases_var in biases.values():
                var_summaries(biases_var)

        var_list = [weights['conv_W1'], weights['conv_W2'], weights['dense_W1'], weights['dense_W2'], 
        weights['output_W'], biases['conv_b1'], biases['conv_b2'], biases['dense_b1'], 
        biases['dense_b2'], biases['output_b']]

        if (params['architecture'] == 'BindingCNN') or (params['architecture'] == 'controlCNN'):
                var_list.append(weights['course_bindingW1'])
                var_list.append(weights['finegrained_bindingW1'])

    if (params['dataset'] == 'cifar10'): #Define core variables for a VGG architecture for CIFAR-10

        x = tf.compat.v1.placeholder(training_data.dtype, [None, 32, 32, 3], name='x-input')

        with tf.compat.v1.variable_scope(params['architecture']):
        #Note for example that the first convolutional weights layer has a 5x5 filter with 1 input channel, and 6 output channels
        #tf.compat.v1.get_variable will either get an existing variable with these parameters, or otherwise create a new one
            weights = {
            'conv_W1' : tf.compat.v1.get_variable('CW1', shape=(3, 3, 3, 32), initializer=initializer),
            'conv_W2' : tf.compat.v1.get_variable('CW2', shape=(3, 3, 32, 32), initializer=initializer),
            'conv_W3' : tf.compat.v1.get_variable('CW3', shape=(3, 3, 32, 64), initializer=initializer),
            'conv_W4' : tf.compat.v1.get_variable('CW4', shape=(3, 3, 64, 64), initializer=initializer),
            'conv_W5' : tf.compat.v1.get_variable('CW5', shape=(3, 3, 64, 128), initializer=initializer),
            'conv_W6' : tf.compat.v1.get_variable('CW6', shape=(3, 3, 128, 128), initializer=initializer),
            'dense_W1' : tf.compat.v1.get_variable('DW1', shape=(4*4*128, 120), initializer=initializer),
            'dense_W2' : tf.compat.v1.get_variable('DW2', shape=(120, 84), initializer=initializer),
            'output_W' : tf.compat.v1.get_variable('OW', shape=(84, 10), initializer=initializer)
            }
            if (params['architecture'] == 'BindingVGG') or (params['architecture'] == 'controlVGG'):
                weights['course_bindingW1'] = tf.compat.v1.get_variable('courseW1', shape=(16*16*64, 120), initializer=initializer)
                weights['finegrained_bindingW1'] = tf.compat.v1.get_variable('fineW1', shape=(16*16*32, 120), initializer=initializer)
                weights['course_bindingW2'] = tf.compat.v1.get_variable('courseW2', shape=(8*8*128, 120), initializer=initializer)
                weights['finegrained_bindingW2'] = tf.compat.v1.get_variable('fineW2', shape=(8*8*64, 120), initializer=initializer)

            #Add summaries for each weightseight variable in the dictionary, for later use in TensorBoard
            for weights_var in weights.values():
                var_summaries(weights_var)

            biases = {
            'conv_b1' : tf.compat.v1.get_variable('Cb1', shape=(32), initializer=initializer),
            'conv_b2' : tf.compat.v1.get_variable('Cb2', shape=(32), initializer=initializer),
            'conv_b3' : tf.compat.v1.get_variable('Cb3', shape=(64), initializer=initializer),
            'conv_b4' : tf.compat.v1.get_variable('Cb4', shape=(64), initializer=initializer),
            'conv_b5' : tf.compat.v1.get_variable('Cb5', shape=(128), initializer=initializer),
            'conv_b6' : tf.compat.v1.get_variable('Cb6', shape=(128), initializer=initializer),
            'dense_b1' : tf.compat.v1.get_variable('Db1', shape=(120), initializer=initializer),
            'dense_b2' : tf.compat.v1.get_variable('Db2', shape=(84), initializer=initializer),
            'output_b' : tf.compat.v1.get_variable('Ob', shape=(10), initializer=initializer)
            }

            for biases_var in biases.values():
                var_summaries(biases_var)
 
        var_list = [weights['conv_W1'], weights['conv_W2'], weights['conv_W3'], weights['conv_W4'], 
        weights['conv_W5'], weights['conv_W6'], weights['dense_W1'], weights['dense_W2'], 
        weights['output_W'], biases['conv_b1'], biases['conv_b2'], biases['conv_b3'], biases['conv_b4'], 
        biases['conv_b5'], biases['conv_b6'], biases['dense_b1'], biases['dense_b2'], biases['output_b']]

        if (params['architecture'] == 'BindingVGG') or (params['architecture'] == 'controlVGG'):
                var_list.append(weights['course_bindingW1'])
                var_list.append(weights['finegrained_bindingW1'])
                var_list.append(weights['course_bindingW2'])
                var_list.append(weights['finegrained_bindingW2'])
    
    return x, y, dropout_rate_placeholder, var_list, weights, biases


def LeNet_predictions(features, dropout_rate_placeholder, weights, biases, dynamic_var):

    print("Building standard LeNet CNN")

    sparsity_dic = {} #Store the sparsity of layer activations for later analysis
    pool1_drop, _, sparsity_dic = conv1_sequence(features, dropout_rate_placeholder, weights, biases, sparsity_dic)

    pool2_drop, _, _, sparsity_dic = conv2_sequence(pool1_drop, dropout_rate_placeholder, weights, biases, sparsity_dic)

    #Operations distinct from other networks:
    pool2_flat = tf.reshape(pool2_drop, [-1, 5 * 5 * 16])
    dense1 = tf.nn.bias_add(tf.matmul(pool2_flat, weights['dense_W1']), biases['dense_b1'])

    logits, sparsity_dic, dense2_drop = fc_sequence(dense1, dropout_rate_placeholder, weights, biases, sparsity_dic)

    l1_reg_activations1 = tf.norm(dense2_drop, ord=1, axis=None)
    l1_reg_activations2 = 0

    return logits, sparsity_dic, l1_reg_activations1, l1_reg_activations2

def BindingCNN_predictions(features, dropout_rate_placeholder, weights, biases, dynamic_var):

    print("Building Binding CNN")

    sparsity_dic = {}
    pool1_drop, pool1_indices, sparsity_dic = conv1_sequence(features, dropout_rate_placeholder, weights, biases, sparsity_dic)

    pool2_drop, pool2_indices, relu2, sparsity_dic = conv2_sequence(pool1_drop, dropout_rate_placeholder, weights, biases, sparsity_dic)
    pool2_flat = tf.reshape(pool2_drop, [-1, 5 * 5 * 16])

    #Operations distinct from other networks:
    #'Course' unpooling binding information
    unpool_binding_activations, sparsity_dic = unpooling_sequence(pool_drop=pool2_drop, 
        pool_indices=pool2_indices, relu=relu2, relu_flat_shape=[-1, 10 * 10 * 16], 
        dropout_rate_placeholder=dropout_rate_placeholder, sparsity_dic=sparsity_dic)

    #'Fine-grained' binding information from gradient unpooling
    gradient_unpool_binding_activations, sparsity_dic = gradient_unpooling_sequence(high_level=pool2_drop, 
        low_level=pool1_drop, low_flat_shape=[-1, 14 * 14 * 6], dropout_rate_placeholder=dropout_rate_placeholder, 
        sparsity_dic=sparsity_dic)

    if dynamic_var == "Ablate_unpooling":
        print("Ablating unpooling activations...")
        unpool_binding_activations = tf.zeros(shape=tf.shape(unpool_binding_activations))
    elif dynamic_var == "Ablate_gradient_unpooling":
        print("Ablating gradient unpooling activations...")
        gradient_unpool_binding_activations = tf.zeros(shape=tf.shape(gradient_unpool_binding_activations))
    elif dynamic_var == "Ablate_binding":
        print("Ablating unpooling and gradient unpooling activations...")
        unpool_binding_activations = tf.zeros(shape=tf.shape(unpool_binding_activations))
        gradient_unpool_binding_activations = tf.zeros(shape=tf.shape(gradient_unpool_binding_activations))
    elif dynamic_var == "Ablate_maxpooling":
        pool2_flat = tf.zeros(shape=tf.shape(pool2_flat))

    dense1 = tf.nn.bias_add(tf.add(tf.add(tf.matmul(pool2_flat, weights['dense_W1']),
        tf.matmul(unpool_binding_activations, weights['course_bindingW1'])),
        tf.matmul(gradient_unpool_binding_activations, weights['finegrained_bindingW1'])),
        biases['dense_b1'])

    l1_reg_activations1 = 0
    l1_reg_activations2 = 0

    logits, sparsity_dic, _ = fc_sequence(dense1, dropout_rate_placeholder, weights, biases, sparsity_dic)

    return logits, sparsity_dic, l1_reg_activations1, l1_reg_activations2


def controlCNN_predictions(features, dropout_rate_placeholder, weights, biases, dynamic_var):

    print("Building control-version of Binding CNN")

    sparsity_dic = {}
    pool1_drop, pool1_indices, sparsity_dic = conv1_sequence(features, dropout_rate_placeholder, weights, biases, sparsity_dic)

    pool2_drop, pool2_indices, relu2, sparsity_dic = conv2_sequence(pool1_drop, dropout_rate_placeholder, weights, biases, sparsity_dic)
    pool2_flat = tf.reshape(pool2_drop, [-1, 5 * 5 * 16])

    #Operations distinct from other networks:
    unpool_binding_activations = tf.reshape(relu2, [-1, 10*10*16])
    gradient_unpool_binding_activations = tf.reshape(pool1_drop, [-1, 14*14*6])

    sparsity_dic['gradient_unpool_sparsity'] = tf.math.zero_fraction(gradient_unpool_binding_activations)

    dense1 = tf.nn.bias_add(tf.add(tf.add(tf.matmul(pool2_flat, weights['dense_W1']),
        tf.matmul(unpool_binding_activations, weights['course_bindingW1'])),
        tf.matmul(gradient_unpool_binding_activations, weights['finegrained_bindingW1'])),
        biases['dense_b1'])

    logits, sparsity_dic, _ = fc_sequence(dense1, dropout_rate_placeholder, weights, biases, sparsity_dic)

    l1_reg_activations1 = tf.norm(unpool_binding_activations, ord=1, axis=None)
    l1_reg_activations2 = tf.norm(gradient_unpool_binding_activations, ord=1, axis=None)

    if dynamic_var == 'add_logit_noise':
        print("Adding noise to logits")
        #Add noise as a control for Boundary attack resistance being related to e.g. numerical imprecision
        logits = logits + tf.random.normal(tf.shape(logits), mean=0.0, stddev=5.0)


    return logits, sparsity_dic, l1_reg_activations1, l1_reg_activations2


def VGG_predictions(features, dropout_rate_placeholder, weights, biases, dynamic_var):

    print("Building standard VGG CNN")

    sparsity_dic = {} #Store the sparsity of layer activations for later analysis
    
    #First VGG block
    pool1_drop, _, _, _, sparsity_dic = VGG_conv_sequence(inputs=tf.dtypes.cast(features, dtype=tf.float32), dropout_rate_placeholder=dropout_rate_placeholder, 
        conv_weights=[weights['conv_W1'], weights['conv_W2']], conv_biases=[biases['conv_b1'], biases['conv_b2']], 
        sparsity_dic=sparsity_dic, VGG_block=1)

    #Second VGG block
    pool2_drop, _, _, _, sparsity_dic = VGG_conv_sequence(inputs=pool1_drop, dropout_rate_placeholder=dropout_rate_placeholder, 
        conv_weights=[weights['conv_W3'], weights['conv_W4']], conv_biases=[biases['conv_b3'], biases['conv_b4']], 
        sparsity_dic=sparsity_dic, VGG_block=2)

    #Third VGG block
    pool3_drop, _, _, _, sparsity_dic = VGG_conv_sequence(inputs=pool2_drop, dropout_rate_placeholder=dropout_rate_placeholder, 
        conv_weights=[weights['conv_W5'], weights['conv_W6']], conv_biases=[biases['conv_b5'], biases['conv_b6']], 
        sparsity_dic=sparsity_dic, VGG_block=3)

    #Operations distinct from other networks:
    pool3_flat = tf.reshape(pool3_drop, [-1, 4 * 4 * 128])

    dense1 = tf.nn.bias_add(tf.matmul(pool3_flat, weights['dense_W1']), biases['dense_b1'])

    logits, sparsity_dic, _ = fc_sequence(dense1, dropout_rate_placeholder, weights, biases, sparsity_dic)

    #We do not regularize activations with L1 norm in the VGG networks, so pass 0
    l1_reg_activations1 = 0
    l1_reg_activations2 = 0

    return logits, sparsity_dic, l1_reg_activations1, l1_reg_activations2

def BindingVGG_predictions(features, dropout_rate_placeholder, weights, biases, dynamic_var):

    print("Building a binding VGG-like CNN")

    sparsity_dic = {} #Store the sparsity of layer activations for later analysis
    
    #First VGG block
    pool1_drop, pool1_indices, relu1B, _, sparsity_dic = VGG_conv_sequence(inputs=tf.dtypes.cast(features, dtype=tf.float32), dropout_rate_placeholder=dropout_rate_placeholder, 
        conv_weights=[weights['conv_W1'], weights['conv_W2']], conv_biases=[biases['conv_b1'], biases['conv_b2']], 
        sparsity_dic=sparsity_dic, VGG_block=1)

    #Second VGG block
    pool2_drop, pool2_indices, relu2B, _, sparsity_dic = VGG_conv_sequence(inputs=pool1_drop, dropout_rate_placeholder=dropout_rate_placeholder, 
        conv_weights=[weights['conv_W3'], weights['conv_W4']], conv_biases=[biases['conv_b3'], biases['conv_b4']], 
        sparsity_dic=sparsity_dic, VGG_block=2)

    unpool_binding_activations1, sparsity_dic = unpooling_sequence(pool_drop=pool2_drop, 
        pool_indices=pool2_indices, relu=relu2B, relu_flat_shape=[-1, 16*16*64], 
        dropout_rate_placeholder=dropout_rate_placeholder, sparsity_dic=sparsity_dic)

    #Third VGG block
    pool3_drop, pool3_indices, relu3B, relu3A, sparsity_dic = VGG_conv_sequence(inputs=pool2_drop, dropout_rate_placeholder=dropout_rate_placeholder, 
        conv_weights=[weights['conv_W5'], weights['conv_W6']], conv_biases=[biases['conv_b5'], biases['conv_b6']], 
        sparsity_dic=sparsity_dic, VGG_block=3)

    unpool_binding_activations2, sparsity_dic = unpooling_sequence(pool_drop=pool3_drop, 
        pool_indices=pool3_indices, relu=relu3B, relu_flat_shape=[-1, 8*8*128], 
        dropout_rate_placeholder=dropout_rate_placeholder, sparsity_dic=sparsity_dic)

    #Gradient unpooling
    gradient_unpool_binding_activations1, sparsity_dic = gradient_unpooling_sequence(high_level=pool3_drop, 
        low_level=pool1_drop, low_flat_shape=[-1,16*16*32], dropout_rate_placeholder=dropout_rate_placeholder, 
        sparsity_dic=sparsity_dic)

    gradient_unpool_binding_activations2, sparsity_dic = gradient_unpooling_sequence(high_level=pool3_drop, 
        low_level=pool2_drop, low_flat_shape=[-1,8*8*64], dropout_rate_placeholder=dropout_rate_placeholder, 
        sparsity_dic=sparsity_dic)


    pool3_flat = tf.reshape(pool3_drop, [-1, 4 * 4 * 128])

    dense1 = tf.nn.bias_add(tf.add(tf.add(tf.add(tf.add(
        tf.matmul(pool3_flat, weights['dense_W1']), 
        tf.matmul(unpool_binding_activations1, weights['course_bindingW1'])),
        tf.matmul(gradient_unpool_binding_activations1, weights['finegrained_bindingW1'])),
        tf.matmul(unpool_binding_activations2, weights['course_bindingW2'])),
        tf.matmul(gradient_unpool_binding_activations2, weights['finegrained_bindingW2'])), 
        biases['dense_b1'])

    logits, sparsity_dic, _ = fc_sequence(dense1, dropout_rate_placeholder, weights, biases, sparsity_dic)

    #We do not regularize activations with L1 norm in the VGG networks, so pass 0
    l1_reg_activations1 = 0
    l1_reg_activations2 = 0

    return logits, sparsity_dic, l1_reg_activations1, l1_reg_activations2

def unpooling_sequence(pool_drop, pool_indices, relu, relu_flat_shape, dropout_rate_placeholder, sparsity_dic):
    
    #Extract binding information for mid-level neurons that are driving the max-pooled (spatially invariant) representations
    unpool_binding_activations = max_unpool(pool_drop, pool_indices, relu)
    unpool_binding_activations_flat = tf.reshape(unpool_binding_activations, relu_flat_shape)

    sparsity_dic['unpool_sparsity'] = tf.math.zero_fraction(unpool_binding_activations_flat)

    return unpool_binding_activations_flat, sparsity_dic

def gradient_unpooling_sequence(high_level, low_level, low_flat_shape, dropout_rate_placeholder, sparsity_dic):

    #Extract binding information for low-level neurons that are driving critical (i.e. max-pooled) mid-level neurons
    binding_grad = tf.squeeze(tf.gradients(high_level, low_level, unconnected_gradients=tf.UnconnectedGradients.ZERO), 0) #Squeeze removes the dimension of the gradient tensor that stores dtype
    binding_grad_flat = tf.reshape(binding_grad, low_flat_shape)

    #Use k-th largest value as a threshold for getting a boolean mask
    #K is roughly selected for 85% sparsity
    values, _ = tf.math.top_k(binding_grad_flat, k=round(low_flat_shape[1]*0.15))
    kth = tf.reduce_min(values, axis=1)
    mask = tf.greater_equal(binding_grad_flat, tf.expand_dims(kth, -1))
    low_level_flat = tf.reshape(low_level, low_flat_shape) 
    gradient_unpool_binding_activations = tf.multiply(low_level_flat, tf.dtypes.cast(mask, dtype=tf.float32)) #Apply the Boolean mask element-wise to just the pool1 activations (i.e. not including conv2 transformation)

    sparsity_dic['gradient_unpool_sparsity'] = tf.math.zero_fraction(gradient_unpool_binding_activations)

    return gradient_unpool_binding_activations, sparsity_dic

def conv1_sequence(features, dropout_rate_placeholder, weights, biases, sparsity_dic):

    conv1 = tf.nn.conv2d(input=tf.dtypes.cast(features, dtype=tf.float32), filter=weights['conv_W1'], 
                         strides=[1, 1, 1, 1], padding="SAME")
    conv1 = tf.nn.bias_add(conv1, biases['conv_b1'])
    conv1_drop = tf.nn.dropout(conv1, rate=dropout_rate_placeholder)
    relu1 = tf.nn.relu(conv1_drop)
    sparsity_dic['relu1_sparsity'] = tf.math.zero_fraction(relu1)
    pool1, pool1_indices = tf.nn.max_pool_with_argmax(relu1, ksize=(1,2,2,1), strides=(1,2,2,1), padding="VALID")
    pool1_drop = tf.nn.dropout(pool1, rate=dropout_rate_placeholder)

    return pool1_drop, pool1_indices, sparsity_dic

def conv2_sequence(pool1_drop, dropout_rate_placeholder, weights, biases, sparsity_dic):

    conv2 = tf.nn.conv2d(pool1_drop, weights['conv_W2'], strides=[1,1,1,1], padding="VALID")
    conv2 = tf.nn.bias_add(conv2, biases['conv_b2'])
    conv2_drop = tf.nn.dropout(conv2, rate=dropout_rate_placeholder)
    relu2 = tf.nn.relu(conv2_drop)
    sparsity_dic['relu2_sparsity'] = tf.math.zero_fraction(relu2)
    pool2, pool2_indices = tf.nn.max_pool_with_argmax(relu2, ksize=(1,2,2,1), strides=(1,2,2,1), padding="VALID")
    pool2_drop = tf.nn.dropout(pool2, rate=dropout_rate_placeholder)

    return pool2_drop, pool2_indices, relu2, sparsity_dic

def fc_sequence(dense1, dropout_rate_placeholder, weights, biases, sparsity_dic):

    dense1_drop = tf.nn.dropout(dense1, rate=dropout_rate_placeholder)
    dense1_drop = tf.nn.relu(dense1_drop)
    sparsity_dic['dense1_sparsity'] = tf.math.zero_fraction(dense1_drop)
    dense2 = tf.nn.bias_add(tf.matmul(dense1_drop, weights['dense_W2']), biases['dense_b2'])
    dense2_drop = tf.nn.dropout(dense2, rate=dropout_rate_placeholder)
    dense2_drop = tf.nn.relu(dense2_drop)
    sparsity_dic['dense2_sparsity'] = tf.math.zero_fraction(dense2_drop)
    logits = tf.nn.bias_add(tf.matmul(dense2_drop, weights['output_W']), biases['output_b'])

    return logits, sparsity_dic, dense2_drop

def VGG_conv_sequence(inputs, dropout_rate_placeholder, conv_weights, conv_biases, sparsity_dic, VGG_block):

    #First set of convolutions
    convA = tf.nn.conv2d(input=inputs, filter=conv_weights[0], strides=[1, 1, 1, 1], padding="SAME")
    convA = tf.nn.bias_add(convA, conv_biases[0])
    convA_drop = tf.nn.dropout(convA, rate=dropout_rate_placeholder)
    reluA = tf.nn.relu(convA_drop)
    sparsity_dic['reluA' + str(VGG_block) + '_sparsity'] = tf.math.zero_fraction(reluA)

    #Second set of convolutions
    convB = tf.nn.conv2d(input=reluA, filter=conv_weights[1], strides=[1, 1, 1, 1], padding="SAME")
    convB = tf.nn.bias_add(convB, conv_biases[1])
    convB_drop = tf.nn.dropout(convB, rate=dropout_rate_placeholder)
    reluB = tf.nn.relu(convB_drop)
    sparsity_dic['reluB' + str(VGG_block) + '_sparsity'] = tf.math.zero_fraction(reluB)

    #Max-pooling
    pool, pool_indices = tf.nn.max_pool_with_argmax(reluB, ksize=(1,2,2,1), strides=(1,2,2,1), padding="VALID")
    pool_drop = tf.nn.dropout(pool, rate=dropout_rate_placeholder)

    return pool_drop, pool_indices, reluB, reluA, sparsity_dic


#Define max_unpool function, used in the binding CNN - note credit below for this code
def max_unpool(pool, ind, prev_tensor, scope='unpool_2d'):
    """
    Code credit of 'Twice22' from thread https://github.com/tensorflow/tensorflow/issues/2169
    
    Implement the unpooling operation, as explained here:
    https://stackoverflow.com/questions/36548736/tensorflow-unpooling

    Args:
        pool (tensor): Input tensor of shape (N, H, W, C)
        ind (tensor): Input tensor of shape (N, H, W, C) containing the maximum
            flatten indices (see https://www.tensorflow.org/api_docs/python/tf.nn.max_pool_with_argmax)
        prev_tensor (tensor): previous tensor shape
        scope (str): scope in which to register the operations
    Return:
        ret (tensor): tensor same shape as prev_tensor that corresponds to the "invert" of the
            max pooling operation
    """
    with tf.compat.v1.variable_scope(scope):
        # input_shape = [N, H, W, C]
        input_shape = tf.shape(pool)
        o_shape = tf.shape(prev_tensor)

        output_shape = [input_shape[0], o_shape[1], o_shape[2], input_shape[3]]

        # N * H * W * C
        flat_input_size = tf.reduce_prod(input_shape)

        # flat output_shape = [N, 4 * H * W * C]
        flat_output_shape = [output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]]

        updates = tf.reshape(pool, [flat_input_size])

        # create the tensor [ [[[1]]], [[[0]]], ..., [[[N-1]]] ]
        batch_range = tf.reshape(
            tf.range(tf.cast(output_shape[0], tf.int64), dtype=ind.dtype),
            shape=[input_shape[0], 1, 1, 1])

        # b is a tensor of size (N, H, W, C) whose first element of the batch are 3D-array full of 0
        # second element of the batch are 3D-array full of 1, ...   
        b = tf.ones_like(ind) * batch_range
        b = tf.reshape(b, [flat_input_size, 1])

        # indices = [ [0, ind_1], [0, ind_2], ... [0, ind_k], ..., [N-1, ind_{N*H*W*C}], [N-1, ind_{N*H*W*C-1}] ]
        indices = tf.reshape(ind, [flat_input_size, 1])
        indices = tf.concat([b, indices], axis=-1)

        ret = tf.scatter_nd(indices, updates, shape=tf.cast(flat_output_shape, tf.int64))
        ret = tf.reshape(ret, output_shape)

        set_input_shape = pool.get_shape()
        prev_tensor_shape = prev_tensor.get_shape()

        #tf.set_shape() uses additional information to more precisely specify the shape of a tensor
        #(i.e. with information that cannot be inferred from the graph)
        #see https://www.tensorflow.org/api_docs/python/tf/Tensor#set_shape
        set_output_shape = [set_input_shape[0], prev_tensor_shape[1], prev_tensor_shape[2], set_input_shape[3]]
        ret.set_shape(set_output_shape)

        return ret

#Primary training function
def network_train(params, iter_num, var_list, training_data, training_labels, testing_data, testing_labels, 
    weights, biases, x_placeholder, y_placeholder, dropout_rate_placeholder):

    if params['meta_architecture'] == 'CNN':
        if params['architecture'] == 'LeNet':
            predictions, sparsity_dic, l1_reg_activations1, l1_reg_activations2 = LeNet_predictions(x_placeholder, dropout_rate_placeholder, weights, biases, params['dynamic_var']) 
        elif params['architecture'] == 'VGG':
            predictions, sparsity_dic, l1_reg_activations1, l1_reg_activations2 = VGG_predictions(x_placeholder, dropout_rate_placeholder, weights, biases, params['dynamic_var'])
        elif params['architecture'] == 'BindingCNN':
            predictions, sparsity_dic, l1_reg_activations1, l1_reg_activations2 = BindingCNN_predictions(x_placeholder, dropout_rate_placeholder, weights, biases, params['dynamic_var']) 
        elif params['architecture'] == 'BindingVGG':
            predictions, sparsity_dic, l1_reg_activations1, l1_reg_activations2 = BindingVGG_predictions(x_placeholder, dropout_rate_placeholder, weights, biases, params['dynamic_var']) 
        elif params['architecture'] == 'controlCNN':
            predictions, sparsity_dic, l1_reg_activations1, l1_reg_activations2 = controlCNN_predictions(x_placeholder, dropout_rate_placeholder, weights, biases, params['dynamic_var']) 

        cost = (tf.reduce_mean(tf.compat.v1.losses.softmax_cross_entropy(logits=predictions, onehot_labels=y_placeholder, label_smoothing=params['label_smoothing'])) + 
            params['L1_regularization_activations1']*l1_reg_activations1 + params['L1_regularization_activations2']*l1_reg_activations2)
        tf.compat.v1.summary.scalar('Softmax_cross_entropy', cost)

        correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(y_placeholder, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        total_accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.float32)) #Used to store batched accuracy for evaluating the test dataset
        
        tf.compat.v1.summary.scalar('Accuracy', accuracy)
        accuracy_summary = tf.compat.v1.summary.scalar(name="Accuracy_values", tensor=accuracy)

    #Create the chosen optimizer with tf.train.Adam..., then add it to the graph with .minimize
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=params['learning_rate']).minimize(cost)

    #Define values to be written with the summary method for later visualization
    loss_summary = tf.compat.v1.summary.scalar(name="Loss_values", tensor=cost)

    #Create a Saver object to enable later re-loading of the learned weights
    saver = tf.compat.v1.train.Saver(var_list)

    #Merge and provide directory for saving TF summaries
    merged = tf.compat.v1.summary.merge_all()


    #If using the cifar10 dataset, apply data-augmentation
    if params['dataset'] == 'cifar10':
        width_shift_range=0.1
        height_shift_range=0.1
        horizontal_flip=True
        print("\nApplying data augmentation\n")
    else:
        width_shift_range=0.0
        height_shift_range=0.0
        horizontal_flip=False

    with tf.compat.v1.Session() as sess:

        #Initialize variables; note the requirement for explicit initialization prevents expensive
        #initializers from being re-run when e.g. relaoding a model from a checkpoint
        sess.run(tf.compat.v1.global_variables_initializer())


        # #Run de-bugger
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

        network_name_str = str(iter_num) + params['architecture'] + '_adver_trained_' + str(params['adver_trained'])
        print("\n\nTraining " + network_name_str)
        training_writer = tf.compat.v1.summary.FileWriter('tensorboard_data/tb_' + network_name_str + '/training', sess.graph)
        testing_writer = tf.compat.v1.summary.FileWriter('tensorboard_data/tb_' + network_name_str + '/testing')

        for epoch in range(params['training_epochs']):

            batches=0
            #Perform data augmentation with small translations and horizontal flipping
            datagen = ImageDataGenerator(width_shift_range=width_shift_range, height_shift_range=height_shift_range, horizontal_flip=horizontal_flip)
            for batch_x, batch_y in datagen.flow(training_data, training_labels, batch_size=params['batch_size']):
                batches += 1
                if batches >= len(training_labels)/params['batch_size']:
                    break

                #Recall that tf.Session.run is the main method for running a tf.Operation or evaluation a tf.Tensor
                #By passing or more Tensors or Operations, TensorFlow will execute the operations needed
                run_optim = sess.run(optimizer, feed_dict = {x_placeholder: batch_x, y_placeholder: batch_y, dropout_rate_placeholder : params['dropout_rate']})

                loss, acc = sess.run([cost, accuracy], feed_dict = {x_placeholder: batch_x, y_placeholder: batch_y, dropout_rate_placeholder : 0.0})

            training_summ, training_acc = sess.run([merged, accuracy], feed_dict={x_placeholder: batch_x, y_placeholder: batch_y, dropout_rate_placeholder : 0.0})
            training_writer.add_summary(training_summ, epoch)

            #Find the accuracy on the test dataset using batches to avoid issues of memory capacity
            accuracy_total = 0
            for test_batch in range(math.ceil(len(testing_labels)/params['batch_size'])):

                test_batch_x = testing_data[test_batch*params['batch_size']:min((test_batch+1)*params['batch_size'], len(testing_labels))]
                test_batch_y = testing_labels[test_batch*params['batch_size']:min((test_batch+1)*params['batch_size'], len(testing_labels))]

                batch_testing_acc = sess.run(total_accuracy, feed_dict={x_placeholder: test_batch_x, y_placeholder: test_batch_y, dropout_rate_placeholder : 0.0})
                accuracy_total += batch_testing_acc

            testing_acc = accuracy_total/len(testing_labels)
            #testing_writer.add_summary(testing_acc, epoch)

            print("At iteration " + str(epoch) + ", Loss = " + \
                 "{:.4f}".format(loss) + ", Training Accuracy = " + \
                                "{:.4f}".format(training_acc) + ", Testing Accuracy = " + \
                                "{:.4f}".format(testing_acc))

        print("Training complete")

        save_path = saver.save(sess, "network_weights_data/" + network_name_str + ".ckpt")

        print("Final testing Accuracy:","{:.5f}".format(testing_acc))

        #On small-memory data-sets, check layer-wise sparsity
        if params['dataset']!='cifar10':
            testing_sparsity = sess.run(sparsity_dic, feed_dict={x_placeholder: testing_data, y_placeholder: testing_labels, dropout_rate_placeholder : 0.0})

            print("\nLayer-wise sparsity:")
            print(testing_sparsity)

            print("Mean sparsity is " + str(np.mean(np.fromiter(testing_sparsity.values(), dtype=float))))
        else:
            testing_sparsity = {'NA':-1.0}

        training_writer.close()
        testing_writer.close()

        return training_acc , testing_acc, network_name_str, testing_sparsity


if __name__ == '__main__':

    params = {'architecture':'BindingCNN',
    'dynamic_var':'None',
    'dataset':'mnist',
    'meta_architecture':'CNN',
    'training_epochs':30,
    'adver_trained':False,
    'crossval_bool':False,
    'dropout_rate':0.25,
    'label_smoothing':0.1,
    'L1_regularization_activations1':0.0,
    'L1_regularization_activations2':0.0,
    'learning_rate':0.001,
    'batch_size':128} #NB that drop-out 'rate' = 1 - 'keep probability'

    (training_data, training_labels, testing_data, testing_labels, _, _) = data_setup(params)

    x, y, dropout_rate_placeholder, var_list, weights, biases = initializer_fun(params, training_data, training_labels)
    iter_num = 0

    network_train(params, iter_num, var_list, training_data, training_labels, testing_data, testing_labels, weights, biases, x_placeholder=x, y_placeholder=y, dropout_rate_placeholder=dropout_rate_placeholder)


