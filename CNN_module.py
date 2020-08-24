#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import math
from keras.datasets import mnist, cifar10
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from skimage.util import random_noise
import os
import tfCore_adversarial_attacks as atk

# Suppress unecessary logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

def data_setup(params):

    if params['dataset'] == 'mnist':
        print("\nLoading MNIST data-set")
        (training_data, training_labels), (testing_data, testing_labels) = mnist.load_data()
        training_data = np.reshape(training_data, [np.shape(training_data)[0], 28, 28, 1])
        testing_data = np.reshape(testing_data, [np.shape(testing_data)[0], 28, 28, 1])

    elif params['dataset'] == 'cifar10':
        print("\nLoading CIFAR-10 data-set")
        (training_data, training_labels), (testing_data, testing_labels) = cifar10.load_data()
    
    training_data = training_data/255
    testing_data = testing_data/255

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

#Creates summary variables for later visualisation of the network
def var_summaries(variable, key):
    with tf.name_scope(name=(key + '_summ')):
        mean = tf.reduce_mean(variable)
        tf.compat.v1.summary.scalar('Mean', mean)

        with tf.name_scope('STD'):
            std = tf.sqrt(tf.reduce_mean(tf.square(variable - mean)))

        tf.compat.v1.summary.scalar('STD', std)
        tf.compat.v1.summary.scalar('Max', tf.reduce_max(variable))
        tf.compat.v1.summary.scalar('Min', tf.reduce_min(variable))
        tf.compat.v1.summary.histogram('Histogram', variable)

def initializer_fun(params, training_data, training_labels):

    tf.compat.v1.reset_default_graph() #Re-set the default graph to clear previous e.g. variable assignments

    dropout_rate_placeholder = tf.compat.v1.placeholder(tf.float32)
    #He-initialization; note the paper 'Delving Deep into Rectifiers' used a value of 2.0; params['He_modifier'] is by default set to 1.0
    initializer = tf.contrib.layers.variance_scaling_initializer(factor=2.0*params['He_modifier'])
    binding_regularizer_l2 = tf.contrib.layers.l2_regularizer(scale=params['L2_regularization_scale_binding'])
    maxpool_regularizer_l2 = tf.contrib.layers.l2_regularizer(scale=params['L2_regularization_scale_maxpool'])

    y = tf.compat.v1.placeholder(training_labels.dtype, [None, 10], name='y-input')

    if (params['dataset'] == 'cifar10'): #Define variables for a VGG-like architecture for CIFAR-10

        x = tf.compat.v1.placeholder(training_data.dtype, [None, 32, 32, 3], name='x-input')

        with tf.compat.v1.variable_scope(params['architecture']):
            weights = {
            'conv_W1' : tf.compat.v1.get_variable('CW1', shape=(3, 3, 3, 32), initializer=initializer),
            'conv_W2' : tf.compat.v1.get_variable('CW2', shape=(3, 3, 32, 32), initializer=initializer),
            'conv_W3' : tf.compat.v1.get_variable('CW3', shape=(3, 3, 32, 64), initializer=initializer),
            'conv_W4' : tf.compat.v1.get_variable('CW4', shape=(3, 3, 64, 64), initializer=initializer),
            'conv_W5' : tf.compat.v1.get_variable('CW5', shape=(3, 3, 64, 128), initializer=initializer),
            'conv_W6' : tf.compat.v1.get_variable('CW6', shape=(3, 3, 128, 128), initializer=initializer),
            'dense_W1' : tf.compat.v1.get_variable('DW1', shape=(4*4*128, params['MLP_layer_1_dim']), initializer=initializer, regularizer=maxpool_regularizer_l2),
            'dense_W2' : tf.compat.v1.get_variable('DW2', shape=(params['MLP_layer_1_dim'], params['MLP_layer_2_dim']), initializer=initializer),
            'output_W' : tf.compat.v1.get_variable('OW', shape=(params['MLP_layer_2_dim'], 10), initializer=initializer)
            }
            if (params['architecture'] == 'BindingVGG'):
                weights['course_bindingW1'] = tf.compat.v1.get_variable('courseW1', shape=(16*16*64, params['MLP_layer_1_dim']), initializer=initializer, regularizer=binding_regularizer_l2)
                weights['finegrained_bindingW1'] = tf.compat.v1.get_variable('fineW1', shape=(16*16*32, params['MLP_layer_1_dim']), initializer=initializer, regularizer=binding_regularizer_l2)
                weights['course_bindingW2'] = tf.compat.v1.get_variable('courseW2', shape=(8*8*128, params['MLP_layer_1_dim']), initializer=initializer, regularizer=binding_regularizer_l2)
                weights['finegrained_bindingW2'] = tf.compat.v1.get_variable('fineW2', shape=(8*8*64, params['MLP_layer_1_dim']), initializer=initializer, regularizer=binding_regularizer_l2)

            biases = {
            'conv_b1' : tf.compat.v1.get_variable('Cb1', shape=(32), initializer=initializer),
            'conv_b2' : tf.compat.v1.get_variable('Cb2', shape=(32), initializer=initializer),
            'conv_b3' : tf.compat.v1.get_variable('Cb3', shape=(64), initializer=initializer),
            'conv_b4' : tf.compat.v1.get_variable('Cb4', shape=(64), initializer=initializer),
            'conv_b5' : tf.compat.v1.get_variable('Cb5', shape=(128), initializer=initializer),
            'conv_b6' : tf.compat.v1.get_variable('Cb6', shape=(128), initializer=initializer),
            'dense_b1' : tf.compat.v1.get_variable('Db1', shape=(params['MLP_layer_1_dim']), initializer=initializer),
            'dense_b2' : tf.compat.v1.get_variable('Db2', shape=(params['MLP_layer_2_dim']), initializer=initializer),
            'output_b' : tf.compat.v1.get_variable('Ob', shape=(10), initializer=initializer)
            }

        var_list = [weights['conv_W1'], weights['conv_W2'], weights['conv_W3'], weights['conv_W4'], 
            weights['conv_W5'], weights['conv_W6'], weights['dense_W1'], weights['dense_W2'], 
            weights['output_W'], biases['conv_b1'], biases['conv_b2'], biases['conv_b3'], biases['conv_b4'], 
            biases['conv_b5'], biases['conv_b6'], biases['dense_b1'], biases['dense_b2'], biases['output_b']]

        if (params['architecture'] == 'BindingVGG'):
                var_list.append(weights['course_bindingW1'])
                var_list.append(weights['finegrained_bindingW1'])
                var_list.append(weights['course_bindingW2'])
                var_list.append(weights['finegrained_bindingW2'])

    else:

        x = tf.compat.v1.placeholder(training_data.dtype, [None, 28, 28, 1], name='x-input')

        if params['architecture'] == 'MadryCNN':
            
            with tf.compat.v1.variable_scope(params['architecture']):
                weights = {
                'conv_W1' : tf.compat.v1.get_variable('CW1', shape=(5, 5, 1, 16), initializer=initializer),
                'conv_W2' : tf.compat.v1.get_variable('CW2', shape=(5, 5, 16, 32), initializer=initializer),
                'dense_W1' : tf.compat.v1.get_variable('DW1', shape=(5*5*32, 512), initializer=initializer),
                'output_W' : tf.compat.v1.get_variable('OW', shape=(512, 10), initializer=initializer)
                }

                biases = {
                'conv_b1' : tf.compat.v1.get_variable('Cb1', shape=(16), initializer=initializer),
                'conv_b2' : tf.compat.v1.get_variable('Cb2', shape=(32), initializer=initializer),
                'dense_b1' : tf.compat.v1.get_variable('Db1', shape=(512), initializer=initializer),
                'output_b' : tf.compat.v1.get_variable('Ob', shape=(10), initializer=initializer)
                }

            var_list = [weights['conv_W1'], weights['conv_W2'], weights['dense_W1'], 
                weights['output_W'], biases['conv_b1'], biases['conv_b2'], biases['dense_b1'], 
                biases['output_b']]


        else:

            with tf.compat.v1.variable_scope(params['architecture']):
                weights = {
                'conv_W1' : tf.compat.v1.get_variable('CW1', shape=(5, 5, 1, 6), initializer=initializer),
                'conv_W2' : tf.compat.v1.get_variable('CW2', shape=(5, 5, 6, 16), initializer=initializer),
                'dense_W1' : tf.compat.v1.get_variable('DW1', shape=(400, params['MLP_layer_1_dim']), initializer=initializer, regularizer=maxpool_regularizer_l2),
                'dense_W2' : tf.compat.v1.get_variable('DW2', shape=(params['MLP_layer_1_dim'], params['MLP_layer_2_dim']), initializer=initializer),
                'output_W' : tf.compat.v1.get_variable('OW', shape=(params['MLP_layer_2_dim'], 10), initializer=initializer)
                }

                if (params['architecture'] == 'BindingCNN'):
                    weights['course_bindingW1'] = tf.compat.v1.get_variable('courseW1', shape=(1600, params['MLP_layer_1_dim']), initializer=initializer, regularizer=binding_regularizer_l2)
                    weights['finegrained_bindingW1'] = tf.compat.v1.get_variable('fineW1', shape=(1176, params['MLP_layer_1_dim']), initializer=initializer, regularizer=binding_regularizer_l2)

                biases = {
                'conv_b1' : tf.compat.v1.get_variable('Cb1', shape=(6), initializer=initializer),
                'conv_b2' : tf.compat.v1.get_variable('Cb2', shape=(16), initializer=initializer),
                'dense_b1' : tf.compat.v1.get_variable('Db1', shape=(params['MLP_layer_1_dim']), initializer=initializer),
                'dense_b2' : tf.compat.v1.get_variable('Db2', shape=(params['MLP_layer_2_dim']), initializer=initializer),
                'output_b' : tf.compat.v1.get_variable('Ob', shape=(10), initializer=initializer)
                }

            var_list = [weights['conv_W1'], weights['conv_W2'], weights['dense_W1'], weights['dense_W2'], 
                weights['output_W'], biases['conv_b1'], biases['conv_b2'], biases['dense_b1'], 
                biases['dense_b2'], biases['output_b']]

            if (params['architecture'] == 'BindingCNN'):
                var_list.append(weights['course_bindingW1'])
                var_list.append(weights['finegrained_bindingW1'])

    
    #Add summaries for each weight variable in the dictionary, for later use in TensorBoard
    for weights_key, weights_var in weights.items():
        var_summaries(weights_var, weights_key)

    for biases_key, biases_var in biases.items():
        var_summaries(biases_var, biases_key)

    return x, y, dropout_rate_placeholder, var_list, weights, biases


def LeNet_predictions(features, dropout_rate_placeholder, weights, biases, dynamic_dic):

    print("Building standard LeNet CNN")

    scalar_dic = {} #Store the sparsity of layer activations for later analysis
    pool1_drop, pool1_indices, relu1, scalar_dic['relu1_sparsity'], scalar_dic['pool1_sparsity'] = basic_conv_sequence(features, 
        dropout_rate_placeholder, weights['conv_W1'], biases['conv_b1'], 'SAME', scalar_dic)

    pool2_drop, pool2_indices, relu2, scalar_dic['relu2_sparsity'], scalar_dic['pool2_sparsity']  = basic_conv_sequence(pool1_drop, 
        dropout_rate_placeholder, weights['conv_W2'], biases['conv_b2'], 'VALID', scalar_dic)
    pool2_flat = tf.reshape(pool2_drop, [-1, 5 * 5 * 16])

    #Operations distinct from other networks:
    dense1 = tf.nn.bias_add(tf.matmul(pool2_flat, weights['dense_W1']), biases['dense_b1'])

    logits, scalar_dic, dense1_drop, dense2_drop = fc_sequence(dense1, dropout_rate_placeholder, weights, biases, scalar_dic)

    if dynamic_dic['dynamic_var'] == 'Add_logit_noise':
        print("Adding noise to logits")
        #Add noise as a control for Boundary attack resistance being related to e.g. numerical imprecision
        logits = logits + tf.random.normal(tf.shape(logits), mean=0.0, stddev=0.1)

    return logits, scalar_dic

#Larger model for MNIST with a similar basic architecture to that used in Madry et, 2017
def MadryCNN_predictions(features, dropout_rate_placeholder, weights, biases, dynamic_dic):

    print("Building a CNN architecture as used in Madry et al, 2017")

    scalar_dic = {} #Store the sparsity of layer activations for later analysis
    pool1_drop, pool1_indices, relu1, scalar_dic['relu1_sparsity'], scalar_dic['pool1_sparsity'] = basic_conv_sequence(features, 
        dropout_rate_placeholder, weights['conv_W1'], biases['conv_b1'], 'SAME', scalar_dic)

    pool2_drop, pool2_indices, relu2, scalar_dic['relu2_sparsity'], scalar_dic['pool2_sparsity']  = basic_conv_sequence(pool1_drop, 
        dropout_rate_placeholder, weights['conv_W2'], biases['conv_b2'], 'VALID', scalar_dic)

    #Operations distinct from other networks:
    pool2_flat = tf.reshape(pool2_drop, [-1, 5 * 5 * 32])
    dense1 = tf.nn.bias_add(tf.matmul(pool2_flat, weights['dense_W1']), biases['dense_b1'])

    #Note Madry-like architecture only has one fully connected layer
    dense1_drop = tf.nn.dropout(dense1, rate=dropout_rate_placeholder)
    dense1_drop = tf.nn.relu(dense1_drop)
    logits = tf.nn.bias_add(tf.matmul(dense1_drop, weights['output_W']), biases['output_b'])

    return logits, scalar_dic

def BindingCNN_predictions(features, dropout_rate_placeholder, weights, biases, dynamic_dic):

    print("Building Binding CNN")

    scalar_dic = {} #Store the sparsity of layer activations for later analysis
    pool1_drop, pool1_indices, relu1, scalar_dic['relu1_sparsity'], scalar_dic['pool1_sparsity'] = basic_conv_sequence(features, 
        dropout_rate_placeholder, weights['conv_W1'], biases['conv_b1'], 'SAME', scalar_dic)

    pool2_drop, pool2_indices, relu2, scalar_dic['relu2_sparsity'], scalar_dic['pool2_sparsity']  = basic_conv_sequence(pool1_drop, 
        dropout_rate_placeholder, weights['conv_W2'], biases['conv_b2'], 'VALID', scalar_dic)
    pool2_flat = tf.reshape(pool2_drop, [-1, 5 * 5 * 16])

    #Operations distinct from other networks:
    #'Course' binding information from unpooling
    unpool_binding_activations, scalar_dic = unpooling_sequence(pool_drop=pool2_drop, 
        pool_indices=pool2_indices, relu=relu2, relu_flat_shape=[-1, 10 * 10 * 16], 
        dropout_rate_placeholder=dropout_rate_placeholder, scalar_dic=scalar_dic)

    #'Fine-grained' binding information from gradient unpooling
    gradient_unpool_binding_activations, scalar_dic = gradient_unpooling_sequence(high_level=pool2_drop, 
        low_level=pool1_drop, low_flat_shape=[-1, 14 * 14 * 6], dropout_rate_placeholder=dropout_rate_placeholder, 
        scalar_dic=scalar_dic, dynamic_dic=dynamic_dic)

    if ((dynamic_dic['dynamic_var'] == "Ablate_unpooling") or 
        (dynamic_dic['dynamic_var'] == 'kloser_gradients')):
        print("Ablating unpooling activations...")
        unpool_binding_activations = tf.zeros(shape=tf.shape(unpool_binding_activations))
    elif dynamic_dic['dynamic_var'] == "Ablate_gradient_unpooling":
        print("Ablating gradient unpooling activations...")
        gradient_unpool_binding_activations = tf.zeros(shape=tf.shape(gradient_unpool_binding_activations))
    elif dynamic_dic['dynamic_var'] == "Ablate_binding":
        print("Ablating unpooling and gradient unpooling activations...")
        unpool_binding_activations = tf.zeros(shape=tf.shape(unpool_binding_activations))
        gradient_unpool_binding_activations = tf.zeros(shape=tf.shape(gradient_unpool_binding_activations))
    elif dynamic_dic['dynamic_var'] == "Ablate_maxpooling":
        print("Ablating max-pooling activations")
        pool2_flat = tf.zeros(shape=tf.shape(pool2_flat))

    dense1 = tf.nn.bias_add(tf.add(tf.add(tf.matmul(pool2_flat, weights['dense_W1']),
        tf.matmul(unpool_binding_activations, weights['course_bindingW1'])),
        tf.matmul(gradient_unpool_binding_activations, weights['finegrained_bindingW1'])),
        biases['dense_b1'])

    logits, scalar_dic, dense1_drop, dense2_drop = fc_sequence(dense1, dropout_rate_placeholder, weights, biases, scalar_dic)

    return logits, scalar_dic

def VGG_predictions(features, dropout_rate_placeholder, weights, biases, dynamic_dic):

    print("Building standard VGG CNN")

    scalar_dic = {} 
    
    #First VGG block
    pool1_drop, _, _, _, scalar_dic = VGG_conv_sequence(inputs=tf.dtypes.cast(features, dtype=tf.float32), dropout_rate_placeholder=dropout_rate_placeholder, 
        conv_weights=[weights['conv_W1'], weights['conv_W2']], conv_biases=[biases['conv_b1'], biases['conv_b2']], 
        scalar_dic=scalar_dic, VGG_block=1)

    #Second VGG block
    pool2_drop, _, _, _, scalar_dic = VGG_conv_sequence(inputs=pool1_drop, dropout_rate_placeholder=dropout_rate_placeholder, 
        conv_weights=[weights['conv_W3'], weights['conv_W4']], conv_biases=[biases['conv_b3'], biases['conv_b4']], 
        scalar_dic=scalar_dic, VGG_block=2)

    #Third VGG block
    pool3_drop, _, _, _, scalar_dic = VGG_conv_sequence(inputs=pool2_drop, dropout_rate_placeholder=dropout_rate_placeholder, 
        conv_weights=[weights['conv_W5'], weights['conv_W6']], conv_biases=[biases['conv_b5'], biases['conv_b6']], 
        scalar_dic=scalar_dic, VGG_block=3)

    #Operations distinct from other networks:
    pool3_flat = tf.reshape(pool3_drop, [-1, 4 * 4 * 128])

    dense1 = tf.nn.bias_add(tf.matmul(pool3_flat, weights['dense_W1']), biases['dense_b1'])

    logits, scalar_dic, _, _ = fc_sequence(dense1, dropout_rate_placeholder, weights, biases, scalar_dic)

    return logits, scalar_dic

def BindingVGG_predictions(features, dropout_rate_placeholder, weights, biases, dynamic_dic):

    print("Building a Binding VGG-like CNN")

    scalar_dic = {}
    
    #First VGG block
    pool1_drop, pool1_indices, relu1B, _, scalar_dic = VGG_conv_sequence(inputs=tf.dtypes.cast(features, dtype=tf.float32), dropout_rate_placeholder=dropout_rate_placeholder, 
        conv_weights=[weights['conv_W1'], weights['conv_W2']], conv_biases=[biases['conv_b1'], biases['conv_b2']], 
        scalar_dic=scalar_dic, VGG_block=1)

    #Second VGG block
    pool2_drop, pool2_indices, relu2B, _, scalar_dic = VGG_conv_sequence(inputs=pool1_drop, dropout_rate_placeholder=dropout_rate_placeholder, 
        conv_weights=[weights['conv_W3'], weights['conv_W4']], conv_biases=[biases['conv_b3'], biases['conv_b4']], 
        scalar_dic=scalar_dic, VGG_block=2)

    unpool_binding_activations1, scalar_dic = unpooling_sequence(pool_drop=pool2_drop, 
        pool_indices=pool2_indices, relu=relu2B, relu_flat_shape=[-1, 16*16*64], 
        dropout_rate_placeholder=dropout_rate_placeholder, scalar_dic=scalar_dic)

    #Third VGG block
    pool3_drop, pool3_indices, relu3B, _, scalar_dic = VGG_conv_sequence(inputs=pool2_drop, dropout_rate_placeholder=dropout_rate_placeholder, 
        conv_weights=[weights['conv_W5'], weights['conv_W6']], conv_biases=[biases['conv_b5'], biases['conv_b6']], 
        scalar_dic=scalar_dic, VGG_block=3)

    unpool_binding_activations2, scalar_dic = unpooling_sequence(pool_drop=pool3_drop, 
        pool_indices=pool3_indices, relu=relu3B, relu_flat_shape=[-1, 8*8*128], 
        dropout_rate_placeholder=dropout_rate_placeholder, scalar_dic=scalar_dic)

    #Gradient unpooling
    gradient_unpool_binding_activations1, scalar_dic = gradient_unpooling_sequence(high_level=pool3_drop, 
        low_level=pool1_drop, low_flat_shape=[-1,16*16*32], dropout_rate_placeholder=dropout_rate_placeholder, 
        scalar_dic=scalar_dic, dynamic_dic=dynamic_dic)

    gradient_unpool_binding_activations2, scalar_dic = gradient_unpooling_sequence(high_level=pool3_drop, 
        low_level=pool2_drop, low_flat_shape=[-1,8*8*64], dropout_rate_placeholder=dropout_rate_placeholder, 
        scalar_dic=scalar_dic, dynamic_dic=dynamic_dic)


    pool3_flat = tf.reshape(pool3_drop, [-1, 4 * 4 * 128])

    dense1 = tf.nn.bias_add(tf.add(tf.add(tf.add(tf.add(
        tf.matmul(pool3_flat, weights['dense_W1']), 
        tf.matmul(unpool_binding_activations1, weights['course_bindingW1'])),
        tf.matmul(gradient_unpool_binding_activations1, weights['finegrained_bindingW1'])),
        tf.matmul(unpool_binding_activations2, weights['course_bindingW2'])),
        tf.matmul(gradient_unpool_binding_activations2, weights['finegrained_bindingW2'])), 
        biases['dense_b1'])

    logits, scalar_dic, _, _ = fc_sequence(dense1, dropout_rate_placeholder, weights, biases, scalar_dic)

    return logits, scalar_dic

def unpooling_sequence(pool_drop, pool_indices, relu, relu_flat_shape, dropout_rate_placeholder, scalar_dic):
    
    #Extract binding information for mid-level neurons that are driving the max-pooled (spatially invariant) representations
    unpool_binding_activations = max_unpool(pool_drop, pool_indices, relu)
    unpool_binding_activations_flat = tf.reshape(unpool_binding_activations, relu_flat_shape)

    scalar_dic['unpool_sparsity'] = tf.math.zero_fraction(unpool_binding_activations_flat)

    return unpool_binding_activations_flat, scalar_dic

def gradient_unpooling_sequence(high_level, low_level, low_flat_shape, dropout_rate_placeholder, scalar_dic, dynamic_dic):

    #Extract binding information for low-level neurons that are driving important (i.e. max-pooled) mid-level neurons
    binding_grad = tf.squeeze(tf.gradients(high_level, low_level, unconnected_gradients=tf.UnconnectedGradients.ZERO), 0) #Squeeze removes the dimension of the gradient tensor that stores dtype
    binding_grad_flat = tf.reshape(binding_grad, low_flat_shape)

    #Rather than using the k-largest gradients to apply a mask, use the k-smallest gradients; serves as a control for the gradient operation alone being beneficial
    if dynamic_dic['dynamic_var'] == 'kloser_gradients':
        print("Using the k-*smallest* gradients for 'gradient unpooling'.")
        #Note we use the negative sign to find the 'bottom-k'
        values, _ = tf.math.top_k(tf.negative(binding_grad_flat), k=round(low_flat_shape[1]*dynamic_dic['sparsification_kwinner']))
        kth = tf.reduce_max(tf.negative(values), axis=1)
        mask = tf.less_equal(binding_grad_flat, tf.expand_dims(kth, -1))

    else:
        #Use k-th largest value as a threshold for getting a boolean mask
        values, _ = tf.math.top_k(binding_grad_flat, k=round(low_flat_shape[1]*dynamic_dic['sparsification_kwinner']))
        kth = tf.reduce_min(values, axis=1)
        mask = tf.greater_equal(binding_grad_flat, tf.expand_dims(kth, -1))

    low_level_flat = tf.reshape(low_level, low_flat_shape) 
    gradient_unpool_binding_activations = tf.multiply(low_level_flat, tf.dtypes.cast(mask, dtype=tf.float32)) #Apply the Boolean mask element-wise

    scalar_dic['gradient_unpool_sparsity'] = tf.math.zero_fraction(gradient_unpool_binding_activations)

    return gradient_unpool_binding_activations, scalar_dic

def basic_conv_sequence(features, dropout_rate_placeholder, weights, biases, padding, scalar_dic):

    conv = tf.nn.conv2d(input=tf.dtypes.cast(features, dtype=tf.float32), filter=weights, 
                         strides=[1, 1, 1, 1], padding=padding)
    conv = tf.nn.bias_add(conv, biases)
    conv_drop = tf.nn.dropout(conv, rate=dropout_rate_placeholder)
    relu = tf.nn.relu(conv_drop)
    relu_sparsity = tf.math.zero_fraction(relu)
    pool, pool_indices = tf.nn.max_pool_with_argmax(relu, ksize=(1,2,2,1), strides=(1,2,2,1), padding="VALID")
    pool_drop = tf.nn.dropout(pool, rate=dropout_rate_placeholder)
    pool_sparsity = tf.math.zero_fraction(pool_drop)

    return pool_drop, pool_indices, relu, relu_sparsity, pool_sparsity

def fc_sequence(dense1, dropout_rate_placeholder, weights, biases, scalar_dic):

    dense1_drop = tf.nn.dropout(dense1, rate=dropout_rate_placeholder)
    dense1_drop = tf.nn.relu(dense1_drop)
    scalar_dic['dense1_sparsity'] = tf.math.zero_fraction(dense1_drop)
    dense2 = tf.nn.bias_add(tf.matmul(dense1_drop, weights['dense_W2']), biases['dense_b2'])
    dense2_drop = tf.nn.dropout(dense2, rate=dropout_rate_placeholder)
    dense2_drop = tf.nn.relu(dense2_drop)
    scalar_dic['dense2_sparsity'] = tf.math.zero_fraction(dense2_drop)
    logits = tf.nn.bias_add(tf.matmul(dense2_drop, weights['output_W']), biases['output_b'])

    return logits, scalar_dic, dense1_drop, dense2_drop

def VGG_conv_sequence(inputs, dropout_rate_placeholder, conv_weights, conv_biases, scalar_dic, VGG_block):

    #First set of convolutions
    convA = tf.nn.conv2d(input=inputs, filter=conv_weights[0], strides=[1, 1, 1, 1], padding="SAME")
    convA = tf.nn.bias_add(convA, conv_biases[0])
    convA_drop = tf.nn.dropout(convA, rate=dropout_rate_placeholder)
    reluA = tf.nn.relu(convA_drop)
    scalar_dic['reluA' + str(VGG_block) + '_sparsity'] = tf.math.zero_fraction(reluA)

    #Second set of convolutions
    convB = tf.nn.conv2d(input=reluA, filter=conv_weights[1], strides=[1, 1, 1, 1], padding="SAME")
    convB = tf.nn.bias_add(convB, conv_biases[1])
    convB_drop = tf.nn.dropout(convB, rate=dropout_rate_placeholder)
    reluB = tf.nn.relu(convB_drop)
    scalar_dic['reluB' + str(VGG_block) + '_sparsity'] = tf.math.zero_fraction(reluB)

    #Max-pooling
    pool, pool_indices = tf.nn.max_pool_with_argmax(reluB, ksize=(1,2,2,1), strides=(1,2,2,1), padding="VALID")
    pool_drop = tf.nn.dropout(pool, rate=dropout_rate_placeholder)

    return pool_drop, pool_indices, reluB, reluA, scalar_dic

#Define max_unpool function, used in the binding CNN
def max_unpool(pool, ind, prev_tensor, scope='unpool_2d'):
    """
    Code the creation of 'Twice22' from https://github.com/tensorflow/tensorflow/issues/2169
    
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

    if params['architecture'] == 'LeNet':
        predictions, scalar_dic = LeNet_predictions(x_placeholder, dropout_rate_placeholder, weights, biases, params['dynamic_dic']) 
    elif params['architecture'] == 'BindingCNN':
        predictions, scalar_dic = BindingCNN_predictions(x_placeholder, dropout_rate_placeholder, weights, biases, params['dynamic_dic']) 
    elif params['architecture'] == 'MadryCNN':
        predictions, scalar_dic = MadryCNN_predictions(x_placeholder, dropout_rate_placeholder, weights, biases, params['dynamic_dic']) 
    elif params['architecture'] == 'VGG':
        predictions, scalar_dic = VGG_predictions(x_placeholder, dropout_rate_placeholder, weights, biases, params['dynamic_dic'])
    elif params['architecture'] == 'BindingVGG':
        predictions, scalar_dic = BindingVGG_predictions(x_placeholder, dropout_rate_placeholder, weights, biases, params['dynamic_dic']) 

    cost = (tf.reduce_mean(tf.compat.v1.losses.softmax_cross_entropy(logits=predictions, onehot_labels=y_placeholder, 
        label_smoothing=params['label_smoothing'])) + tf.losses.get_regularization_loss())

    correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(y_placeholder, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    total_accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.float32)) #Used to store batched accuracy for evaluating on the entire test dataset

    #Create the chosen optimizer with tf.train.Adam, then add it to the graph with .minimize
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=params['learning_rate']).minimize(cost)

    #Create a Saver object to enable later re-loading of the learned weights
    saver = tf.compat.v1.train.Saver(var_list)

    #Merge TF summaries
    merged = tf.compat.v1.summary.merge_all()

    def train_batch(batch_x, batch_y, dropout_rate):
        run_optim = sess.run(optimizer, feed_dict = {x_placeholder: batch_x, y_placeholder: batch_y, dropout_rate_placeholder : dropout_rate})
        loss, acc = sess.run([cost, accuracy], feed_dict = {x_placeholder: batch_x, y_placeholder: batch_y, dropout_rate_placeholder : 0.0})
        return loss, acc

    with tf.compat.v1.Session() as sess:

        sess.run(tf.compat.v1.global_variables_initializer())

        network_name_str = str(iter_num) + params['architecture']
        print("\n\nTraining " + network_name_str)
        training_writer = tf.compat.v1.summary.FileWriter('tensorboard_data/tb_' + network_name_str + '/training', sess.graph)
        testing_writer = tf.compat.v1.summary.FileWriter('tensorboard_data/tb_' + network_name_str + '/testing')

        for epoch in range(params['training_epochs']):
            if params['dataset'] == 'cifar10':
                #Use data augmentation for cifar10
                batches=0
                datagen = ImageDataGenerator(width_shift_range=params['shift_range'], height_shift_range=params['shift_range'], horizontal_flip=True)
                for batch_x, batch_y in datagen.flow(training_data, training_labels, batch_size=params['batch_size']):
                    batches += 1

                    if params['Gaussian_noise'] != None:
                        print("Adding Gaussian noise to the training data")
                        batch_x = np.clip(batch_x + np.random.normal(0, scale=params['Gaussian_noise'], size=np.shape(batch_x)), 0, 1)
                    
                    if batches >= len(training_labels)/params['batch_size']:
                        break

                    loss, training_acc = train_batch(batch_x, batch_y, params['dropout_rate'])
            
            else:
                #Shuffle the training data
                rand_idx = np.random.permutation(len(training_labels))
                shuffled_training_data = training_data[rand_idx]
                shuffled_training_labels = training_labels[rand_idx]

                half_point = round(np.shape(shuffled_training_data)[0]/2)

                if params['Gaussian_noise'] != None:
                    print("Adding Gaussian noise to the first half of training data")
                    shuffled_training_data[:half_point] = np.clip(shuffled_training_data[:half_point] + np.random.normal(0, scale=params['Gaussian_noise'], size=np.shape(shuffled_training_data[:half_point])), 0, 1)

                if params['salt&pepper_noise'] != None:
                    print("Adding salt & pepper noise to the second half of training data")
                    shuffled_training_data[half_point:] = random_noise(shuffled_training_data[half_point:], mode='s&p', salt_vs_pepper=0.5, amount=(params['salt&pepper_noise']/(28*28)))

                training_accuracy_total = 0
                for training_batch in range(math.ceil(len(training_labels)/params['batch_size'])):
                    batch_x = shuffled_training_data[training_batch*params['batch_size']:min((training_batch+1)*params['batch_size'], len(training_labels))]
                    batch_y = shuffled_training_labels[training_batch*params['batch_size']:min((training_batch+1)*params['batch_size'], len(training_labels))]

                    loss, acc = train_batch(batch_x, batch_y, params['dropout_rate'])
                    
                    #Without drop-out, assess the performance on the training dataset
                    network_summary, batch_training_acc = sess.run([merged, total_accuracy], feed_dict={x_placeholder: batch_x, y_placeholder: batch_y, dropout_rate_placeholder : 0.0})
                    training_accuracy_total += batch_training_acc

                #Add the TensorBoard summary data of the network (weight histograms etc.), derived from the last batch run
                training_writer.add_summary(network_summary, epoch)
                training_acc = training_accuracy_total/len(training_labels)
                #Convert python variable into a Summary object for TensorFlow
                training_summary = tf.Summary(value=[tf.Summary.Value(tag="training_acc", simple_value=training_acc),])              
                training_writer.add_summary(training_summary, epoch)

            #Find the accuracy on the test dataset using batches to avoid issues of memory capacity
            testing_accuracy_total = 0
            for test_batch in range(math.ceil(len(testing_labels)/params['batch_size'])):

                test_batch_x = testing_data[test_batch*params['batch_size']:min((test_batch+1)*params['batch_size'], len(testing_labels))]
                test_batch_y = testing_labels[test_batch*params['batch_size']:min((test_batch+1)*params['batch_size'], len(testing_labels))]

                batch_testing_acc = sess.run(total_accuracy, feed_dict={x_placeholder: test_batch_x, y_placeholder: test_batch_y, dropout_rate_placeholder : 0.0})
                testing_accuracy_total += batch_testing_acc

            testing_acc = testing_accuracy_total/len(testing_labels)
            testing_summary = tf.Summary(value=[tf.Summary.Value(tag="testing_acc", simple_value=testing_acc),])              
            testing_writer.add_summary(testing_summary, epoch)

            #Note the loss given is only for the most recent batch
            print("At iteration " + str(epoch) + ", Loss = " + \
                 "{:.4f}".format(loss) + ", Training Accuracy = " + \
                                "{:.4f}".format(training_acc) + ", Testing Accuracy = " + \
                                "{:.4f}".format(testing_acc))

        print("Training complete")

        save_path = saver.save(sess, "network_weights_data/" + network_name_str + ".ckpt")

        print("Final testing Accuracy:","{:.5f}".format(testing_acc))

        #On small-memory data-sets, check layer-wise sparsity
        if params['dataset']!='cifar10':
            testing_sparsity = sess.run(scalar_dic, feed_dict={x_placeholder: testing_data, y_placeholder: testing_labels, dropout_rate_placeholder : 0.0})
        else:
            testing_sparsity = {'NA':-1.0}

        training_writer.close()
        testing_writer.close()

        return training_acc, testing_acc, network_name_str, testing_sparsity


if __name__ == '__main__':

    params = {'architecture':'BindingCNN',
    'dynamic_dic':{'dynamic_var':'None', 'sparsification_kwinner':0.4},
    'dataset':'mnist',
    'training_epochs':3,
    'crossval_bool':False,
    'dropout_rate':0.25,
    'label_smoothing':0.1,
    'learning_rate':0.001,
    'Gaussian_noise':None,
    'salt&pepper_noise':None,
    'He_modifier':1.0,
    'MLP_layer_1_dim':120,
    'MLP_layer_2_dim':84,
    'L2_regularization_scale_maxpool':0.0,
    'L2_regularization_scale_binding':0.0,
    'batch_size':128} #NB that drop-out 'rate' = 1 - 'keep probability'

    (training_data, training_labels, testing_data, testing_labels, _, _) = data_setup(params)

    x, y, dropout_rate_placeholder, var_list, weights, biases = initializer_fun(params, training_data, training_labels)
    iter_num = 0

    network_train(params, iter_num, var_list, training_data, training_labels, testing_data, testing_labels, weights, biases,
     x_placeholder=x, y_placeholder=y, dropout_rate_placeholder=dropout_rate_placeholder)


