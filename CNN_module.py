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
from skimage.util import random_noise
import os
import tfCore_adversarial_attacks as atk

# Disable unecessary logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

def data_setup(params):

    # if params['dataset'] == 'mnist' or 'mnist_SchottCNN':
    #     print("\nLoading MNIST data-set")
    #     (training_data, training_labels), (testing_data, testing_labels) = mnist.load_data()
    #     training_data = np.reshape(training_data, [np.shape(training_data)[0], 28, 28, 1])
    #     testing_data = np.reshape(testing_data, [np.shape(testing_data)[0], 28, 28, 1])

    # elif params['dataset'] == 'fashion_mnist':
    #     print("\nLoading Fashion MNIST data-set")
    #     (training_data, training_labels), (testing_data, testing_labels) = fashion_mnist.load_data()
    #     training_data = np.reshape(training_data, [np.shape(training_data)[0], 28, 28, 1])
    #     testing_data = np.reshape(testing_data, [np.shape(testing_data)[0], 28, 28, 1])

    print("\n\n *** temporary fix for CIFAR-10 dataset ***")

    if params['dataset'] == 'cifar10':
        print("\nLoading CIFAR-10 data-set")
        (training_data, training_labels), (testing_data, testing_labels) = cifar10.load_data()
    
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
    #He-initialization; note the 'Delving Deep into Rectifiers' used a value of 2.0
    initializer = tf.contrib.layers.variance_scaling_initializer(factor=2.0*params['He_modifier'])
    regularizer_l2 = tf.contrib.layers.l2_regularizer(scale=params['L2_regularization_scale'])

    print("\n*** L2 regularizaiton implemented!***\n")

    y = tf.compat.v1.placeholder(training_labels.dtype, [None, 10], name='y-input')

    if (params['dataset'] == 'mnist_SchottCNN'): #Define core variables for a LeNet-5 architecture for MNIST/FashionMNIST

        x = tf.compat.v1.placeholder(training_data.dtype, [None, 28, 28, 1], name='x-input')
        
        with tf.compat.v1.variable_scope(params['architecture']):
            weights = {
            'conv_W1' : tf.compat.v1.get_variable('CW1', shape=(5, 5, 1, 20), initializer=initializer),
            'conv_W2' : tf.compat.v1.get_variable('CW2', shape=(4, 4, 20, 70), initializer=initializer),
            'conv_W3' : tf.compat.v1.get_variable('CW3', shape=(3, 3, 70, 256), initializer=initializer),
            'conv_W4' : tf.compat.v1.get_variable('CW4', shape=(5, 5, 256, 10), initializer=initializer)
            }

            biases = {
            'conv_b1' : tf.compat.v1.get_variable('Cb1', shape=(20), initializer=initializer),
            'conv_b2' : tf.compat.v1.get_variable('Cb2', shape=(70), initializer=initializer),
            'conv_b3' : tf.compat.v1.get_variable('Cb3', shape=(256), initializer=initializer),
            'conv_b4' : tf.compat.v1.get_variable('Cb4', shape=(10), initializer=initializer)
            }

            decoder_weights = None

        var_list = [weights['conv_W1'], weights['conv_W2'], weights['conv_W3'], 
        weights['conv_W4'], biases['conv_b1'], biases['conv_b2'], biases['conv_b3'], 
        biases['conv_b4']]

    if params['architecture'] == 'MadryCNN':

        x = tf.compat.v1.placeholder(training_data.dtype, [None, 28, 28, 1], name='x-input')
        
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

            decoder_weights = None

        var_list = [weights['conv_W1'], weights['conv_W2'], weights['dense_W1'], 
        weights['output_W'], biases['conv_b1'], biases['conv_b2'], biases['dense_b1'], 
        biases['output_b']]


    elif (params['dataset'] == 'mnist') or (params['dataset'] == 'fashion_mnist'): #Define core variables for a LeNet-5 architecture for MNIST/FashionMNIST

        x = tf.compat.v1.placeholder(training_data.dtype, [None, 28, 28, 1], name='x-input')
        
        with tf.compat.v1.variable_scope(params['architecture']):
            weights = {
            'conv_W1' : tf.compat.v1.get_variable('CW1', shape=(5, 5, 1, 6), initializer=initializer),
            'conv_W2' : tf.compat.v1.get_variable('CW2', shape=(5, 5, 6, 16), initializer=initializer),
            'dense_W1' : tf.compat.v1.get_variable('DW1', shape=(400, params['MLP_layer_1_dim']), initializer=initializer),
            'dense_W2' : tf.compat.v1.get_variable('DW2', shape=(params['MLP_layer_1_dim'], params['MLP_layer_2_dim']), initializer=initializer),
            'output_W' : tf.compat.v1.get_variable('OW', shape=(params['MLP_layer_2_dim'], 10), initializer=initializer)
            }
            if (params['architecture'] == 'BindingCNN') or (params['architecture'] == 'controlCNN'):
                weights['course_bindingW1'] = tf.compat.v1.get_variable('courseW1', shape=(1600, params['MLP_layer_1_dim']), initializer=initializer, regularizer=regularizer_l2)
                weights['finegrained_bindingW1'] = tf.compat.v1.get_variable('fineW1', shape=(1176, params['MLP_layer_1_dim']), initializer=initializer, regularizer=regularizer_l2)

            if params['architecture'] == 'PixelCNN':
                weights['pixels_W1'] = tf.compat.v1.get_variable('pixelsW1', shape=(28*28*4, params['MLP_layer_1_dim']), initializer=initializer)


            biases = {
            'conv_b1' : tf.compat.v1.get_variable('Cb1', shape=(6), initializer=initializer),
            'conv_b2' : tf.compat.v1.get_variable('Cb2', shape=(16), initializer=initializer),
            'dense_b1' : tf.compat.v1.get_variable('Db1', shape=(params['MLP_layer_1_dim']), initializer=initializer),
            'dense_b2' : tf.compat.v1.get_variable('Db2', shape=(params['MLP_layer_2_dim']), initializer=initializer),
            'output_b' : tf.compat.v1.get_variable('Ob', shape=(10), initializer=initializer)
            }

            decoder_weights = {
                'de_conv_W1' : tf.get_variable('de_CW1', shape=(5, 5, 6, 1), initializer=initializer),
                'de_conv_W2' : tf.get_variable('de_CW2', shape=(5, 5, 16, 6), initializer=initializer),
                'de_dense_W1' : tf.get_variable('de_DW1', shape=(params['MLP_layer_1_dim'], 400), initializer=initializer),
                'de_dense_W2' : tf.get_variable('de_DW2', shape=(params['MLP_layer_2_dim'], params['MLP_layer_1_dim']), initializer=initializer),
                }

        var_list = [weights['conv_W1'], weights['conv_W2'], weights['dense_W1'], weights['dense_W2'], 
        weights['output_W'], biases['conv_b1'], biases['conv_b2'], biases['dense_b1'], 
        biases['dense_b2'], biases['output_b']]

        if (params['architecture'] == 'BindingCNN') or (params['architecture'] == 'controlCNN'):
            var_list.append(weights['course_bindingW1'])
            var_list.append(weights['finegrained_bindingW1'])

        if params['architecture'] == 'PixelCNN':
            var_list.append(weights['pixels_W1'])

        if params['meta_architecture'] == 'SAE':
            var_list.extend([decoder_weights['de_conv_W1'], decoder_weights['de_conv_W2'], 
                decoder_weights['de_dense_W1'], decoder_weights['de_dense_W2']])

    if (params['dataset'] == 'cifar10'): #Define core variables for a VGG-like architecture for CIFAR-10

        x = tf.compat.v1.placeholder(training_data.dtype, [None, 32, 32, 3], name='x-input')

        with tf.compat.v1.variable_scope(params['architecture']):
            weights = {
            'conv_W1' : tf.compat.v1.get_variable('CW1', shape=(3, 3, 3, 32), initializer=initializer),
            'conv_W2' : tf.compat.v1.get_variable('CW2', shape=(3, 3, 32, 32), initializer=initializer),
            'conv_W3' : tf.compat.v1.get_variable('CW3', shape=(3, 3, 32, 64), initializer=initializer),
            'conv_W4' : tf.compat.v1.get_variable('CW4', shape=(3, 3, 64, 64), initializer=initializer),
            'conv_W5' : tf.compat.v1.get_variable('CW5', shape=(3, 3, 64, 128), initializer=initializer),
            'conv_W6' : tf.compat.v1.get_variable('CW6', shape=(3, 3, 128, 128), initializer=initializer),
            'dense_W1' : tf.compat.v1.get_variable('DW1', shape=(4*4*128, params['MLP_layer_1_dim']), initializer=initializer),
            'dense_W2' : tf.compat.v1.get_variable('DW2', shape=(params['MLP_layer_1_dim'], params['MLP_layer_2_dim']), initializer=initializer),
            'output_W' : tf.compat.v1.get_variable('OW', shape=(params['MLP_layer_2_dim'], 10), initializer=initializer)
            }
            if (params['architecture'] == 'BindingVGG') or (params['architecture'] == 'controlVGG'):
                weights['course_bindingW1'] = tf.compat.v1.get_variable('courseW1', shape=(16*16*64, params['MLP_layer_1_dim']), initializer=initializer)
                weights['finegrained_bindingW1'] = tf.compat.v1.get_variable('fineW1', shape=(16*16*32, params['MLP_layer_1_dim']), initializer=initializer)
                weights['course_bindingW2'] = tf.compat.v1.get_variable('courseW2', shape=(8*8*128, params['MLP_layer_1_dim']), initializer=initializer)
                weights['finegrained_bindingW2'] = tf.compat.v1.get_variable('fineW2', shape=(8*8*64, params['MLP_layer_1_dim']), initializer=initializer)

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

            decoder_weights = None

        var_list = [weights['conv_W1'], weights['conv_W2'], weights['conv_W3'], weights['conv_W4'], 
        weights['conv_W5'], weights['conv_W6'], weights['dense_W1'], weights['dense_W2'], 
        weights['output_W'], biases['conv_b1'], biases['conv_b2'], biases['conv_b3'], biases['conv_b4'], 
        biases['conv_b5'], biases['conv_b6'], biases['dense_b1'], biases['dense_b2'], biases['output_b']]


        if (params['architecture'] == 'BindingVGG') or (params['architecture'] == 'controlVGG'):
                var_list.append(weights['course_bindingW1'])
                var_list.append(weights['finegrained_bindingW1'])
                var_list.append(weights['course_bindingW2'])
                var_list.append(weights['finegrained_bindingW2'])
    
    #Add summaries for each weight variable in the dictionary, for later use in TensorBoard
    for weights_key, weights_var in weights.items():
        var_summaries(weights_var, weights_key)

    for biases_key, biases_var in biases.items():
        var_summaries(biases_var, biases_key)

    return x, y, dropout_rate_placeholder, var_list, weights, biases, decoder_weights


def SchottCNN_predictions(features, dropout_rate_placeholder, weights, biases, dynamic_dic):

    print("Building Schott et al-style basic CNN")


#     # *** need to add batch-normalization

    
    elu1 = standard_conv_sequence(tf.dtypes.cast(features, dtype=tf.float32), weights['conv_W1'], biases['conv_b1'], strides=[1,1,1,1])

    elu2 = standard_conv_sequence(elu1, weights['conv_W2'], biases['conv_b2'], strides=[1,2,2,1])

    elu3 = standard_conv_sequence(elu2, weights['conv_W3'], biases['conv_b3'], strides=[1,2,2,1])

    #Note that ReLU/drop-out shouldn't normally be applied to logits, hence sequence not used
    conv4 = tf.nn.conv2d(input=elu3, filter=weights['conv_W4'], strides=[1,1,1,1], padding="VALID")
    logits = tf.nn.bias_add(conv4, biases['conv_b4'])
    logits = tf.squeeze(logits, axis=1)
    logits = tf.squeeze(logits, axis=1)

    scalar_dic = {} #Store the sparsity of layer activations for later analysis
    #Save latent activations and max-pooling indices for use in auto-encoder model
    AutoEncoder_vars = {}

    l1_reg_activations1 = 0
    l1_reg_activations2 = 0

    return logits, AutoEncoder_vars, scalar_dic, l1_reg_activations1, l1_reg_activations2

def standard_conv_sequence(inputs, conv_weights, conv_biases, strides):

    # *** note that the ELU rather than ReLU activation function is used, as per in Schott et al

    conv = tf.nn.conv2d(input=inputs, filter=conv_weights, strides=strides, padding="VALID")
    conv = tf.nn.bias_add(conv, conv_biases)
    elu = tf.nn.elu(conv)

    return elu


def LeNet_predictions(features, dropout_rate_placeholder, weights, biases, dynamic_dic):

    print("Building standard LeNet CNN")

    scalar_dic = {} #Store the sparsity of layer activations for later analysis
    pool1_drop, pool1_indices, relu1, scalar_dic = conv1_sequence(features, dropout_rate_placeholder, weights, biases, scalar_dic)

    pool2_drop, pool2_indices, relu2, scalar_dic = conv2_sequence(pool1_drop, dropout_rate_placeholder, weights, biases, scalar_dic)

    #Operations distinct from other networks:
    pool2_flat = tf.reshape(pool2_drop, [-1, 5 * 5 * 16])
    dense1 = tf.nn.bias_add(tf.matmul(pool2_flat, weights['dense_W1']), biases['dense_b1'])

    logits, scalar_dic, dense1_drop, dense2_drop = fc_sequence(dense1, dropout_rate_placeholder, weights, biases, scalar_dic)

    #Save latent activations and max-pooling indices for use in auto-encoder model
    AutoEncoder_vars = {'latent_rep':dense2_drop, 'pool2_indices':pool2_indices, 
        'unpool2_shape':relu2, 'pool1_indices':pool1_indices, 'unpool1_shape':relu1}

    #Any L1 activation regularization used on the standard LeNet-5 applies to the fully-connected layer
    l1_reg_activations1 = tf.norm(dense1_drop, ord=1, axis=None)
    l1_reg_activations2 = 0

    if dynamic_dic['dynamic_var'] == 'Add_logit_noise':
        print("Adding noise to logits")
        #Add noise as a control for Boundary attack resistance being related to e.g. numerical imprecision
        logits = logits + tf.random.normal(tf.shape(logits), mean=0.0, stddev=0.1)

    #If desired, visualize how the activations across each layer differ for clean examples (first half of batch), vs adversarial examples (second half)
    if dynamic_dic['analysis_var'] == 'Activations_across_layers':
        #Note that the activations to each image are compared with the corresponding adversarial image
        
        #Split the first and second half of the batch, which correspond to clean and adversarial versions of the same images
        clean_activations_pool1, adversarial_activations_pool1 = tf.split(pool1_drop, num_or_size_splits=2, axis=0)
        clean_activations_pool2, adversarial_activations_pool2 = tf.split(pool2_drop, num_or_size_splits=2, axis=0)
        clean_activations_dense1, adversarial_activations_dense1 = tf.split(dense1_drop, num_or_size_splits=2, axis=0)

        #Measure L-2 or L-inf distance as specified between activations in each layer (distance metric should correspond to the type of adversarial attack)
        distance_pool1 = tf.norm(clean_activations_pool1 - adversarial_activations_pool1, ord='euclidean')
        distance_pool2 = tf.norm(clean_activations_pool2 - adversarial_activations_pool2, ord='euclidean')
        distance_dense1 = tf.norm(clean_activations_dense1 - adversarial_activations_dense1, ord='euclidean')

        #Normalise the distance by the size of the layer
        distance_pool1 = distance_pool1/(14 * 14 * 6)
        distance_pool2 = distance_pool2/(5 * 5 * 16)
        distance_dense1 = distance_dense1/params['MLP_layer_1_dim']

        #Return a dictionary with the scalar values for that batch
        scalar_dic['distance_pool1'] = distance_pool1
        scalar_dic['distance_pool2'] = distance_pool2
        scalar_dic['distance_dense1'] = distance_dense1

    return logits, AutoEncoder_vars, scalar_dic, l1_reg_activations1, l1_reg_activations2


def PixelCNN_predictions(features, dropout_rate_placeholder, weights, biases, dynamic_dic):

    print("Building a Pixel CNN")

    scalar_dic = {} #Store the sparsity of layer activations for later analysis
    pool1_drop, pool1_indices, relu1, scalar_dic = conv1_sequence(features, dropout_rate_placeholder, weights, biases, scalar_dic)

    pool2_drop, pool2_indices, relu2, scalar_dic = conv2_sequence(pool1_drop, dropout_rate_placeholder, weights, biases, scalar_dic)

    #Operations distinct from other networks:
    pool2_flat = tf.reshape(pool2_drop, [-1, 5 * 5 * 16])

    #Flatten the input pixels to concatenate alongside the abstract pooled-features
    pixel_dim = tf.dtypes.cast(28*28*4, dtype=tf.int32)
    print("\nUsing %s pixels from input", pixel_dim)
    print("Concatenating four pixel layers side-by-side")
    features_flat = tf.concat((tf.concat((tf.reshape(tf.dtypes.cast(features, dtype=tf.float32), [-1, 28*28]), 
    	tf.reshape(tf.dtypes.cast(features, dtype=tf.float32), [-1, 28*28])), axis=1),
        tf.concat((tf.reshape(tf.dtypes.cast(features, dtype=tf.float32), [-1, 28*28]), 
        tf.reshape(tf.dtypes.cast(features, dtype=tf.float32), [-1, 28*28])), axis=1)), axis=1)
    features_flat = features_flat[:,0:pixel_dim]

    dense1 = tf.nn.bias_add(
        tf.add(tf.matmul(pool2_flat, weights['dense_W1']), 
            tf.matmul(features_flat, weights['pixels_W1'])), 
        biases['dense_b1'])

    logits, scalar_dic, dense1_drop, dense2_drop = fc_sequence(dense1, dropout_rate_placeholder, weights, biases, scalar_dic)

    AutoEncoder_vars = {}

    #Any L1 activation regularization used on the standard LeNet-5 applies to the fully-connected layer
    l1_reg_activations1 = tf.norm(dense1_drop, ord=1, axis=None)
    l1_reg_activations2 = 0

    return logits, AutoEncoder_vars, scalar_dic, l1_reg_activations1, l1_reg_activations2



#High capacity model for MNIST with the same architecture as used in Madry et, 2017
def MadryCNN_predictions(features, dropout_rate_placeholder, weights, biases, dynamic_dic):

    print("Building a Madry like CNN")

    scalar_dic = {} #Store the sparsity of layer activations for later analysis
    pool1_drop, pool1_indices, relu1, scalar_dic = conv1_sequence(features, dropout_rate_placeholder, weights, biases, scalar_dic)

    pool2_drop, pool2_indices, relu2, scalar_dic = conv2_sequence(pool1_drop, dropout_rate_placeholder, weights, biases, scalar_dic)

    #Operations distinct from other networks:
    pool2_flat = tf.reshape(pool2_drop, [-1, 5 * 5 * 32])
    dense1 = tf.nn.bias_add(tf.matmul(pool2_flat, weights['dense_W1']), biases['dense_b1'])

    #Note Madry architecture only has one fully connected layer
    dense1_drop = tf.nn.dropout(dense1, rate=dropout_rate_placeholder)
    dense1_drop = tf.nn.relu(dense1_drop)
    logits = tf.nn.bias_add(tf.matmul(dense1_drop, weights['output_W']), biases['output_b'])

    #Save latent activations and max-pooling indices for use in auto-encoder model
    AutoEncoder_vars = {}

    #Any L1 activation regularization used on the standard LeNet-5 applies to the fully-connected layer
    l1_reg_activations1 = 0
    l1_reg_activations2 = 0

    return logits, AutoEncoder_vars, scalar_dic, l1_reg_activations1, l1_reg_activations2



def BindingCNN_predictions(features, dropout_rate_placeholder, weights, biases, dynamic_dic):

    print("Building Binding CNN")

    scalar_dic = {}
    pool1_drop, pool1_indices, relu1, scalar_dic = conv1_sequence(features, dropout_rate_placeholder, weights, biases, scalar_dic)

    pool2_drop, pool2_indices, relu2, scalar_dic = conv2_sequence(pool1_drop, dropout_rate_placeholder, weights, biases, scalar_dic)
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
        (dynamic_dic['dynamic_var'] == 'kwinner_activations') or
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
    elif dynamic_dic['dynamic_var'] == "Ablate_max&gradient":
        print("Ablating max-pooling and gradient unpool activations")
        pool2_flat = tf.zeros(shape=tf.shape(pool2_flat))
        gradient_unpool_binding_activations = tf.zeros(shape=tf.shape(gradient_unpool_binding_activations))
    elif dynamic_dic['dynamic_var'] == "Ablate_max&unpool":
        print("Ablating max-pooling and unpooling activations")
        pool2_flat = tf.zeros(shape=tf.shape(pool2_flat))
        unpool_binding_activations = tf.zeros(shape=tf.shape(unpool_binding_activations))

    dense1 = tf.nn.bias_add(tf.add(tf.add(tf.matmul(pool2_flat, weights['dense_W1']),
        tf.matmul(unpool_binding_activations, weights['course_bindingW1'])),
        tf.matmul(gradient_unpool_binding_activations, weights['finegrained_bindingW1'])),
        biases['dense_b1'])

    logits, scalar_dic, dense1_drop, dense2_drop = fc_sequence(dense1, dropout_rate_placeholder, weights, biases, scalar_dic)

    AutoEncoder_vars = {'latent_rep':dense2_drop, 'pool2_indices':pool2_indices, 
        'unpool2_shape':relu2, 'pool1_indices':pool1_indices, 'unpool1_shape':relu1}

    #No L1 activation regularization is used for the binding model
    l1_reg_activations1 = 0
    l1_reg_activations2 = 0

    return logits, AutoEncoder_vars, scalar_dic, l1_reg_activations1, l1_reg_activations2


def controlCNN_predictions(features, dropout_rate_placeholder, weights, biases, dynamic_dic):

    print("Building control-version of Binding CNN")

    scalar_dic = {}
    pool1_drop, scalar_dic = conv1_sequence(features, dropout_rate_placeholder, weights, biases, scalar_dic)

    pool2_drop, _, relu2, scalar_dic = conv2_sequence(pool1_drop, dropout_rate_placeholder, weights, biases, scalar_dic)
    pool2_flat = tf.reshape(pool2_drop, [-1, 5 * 5 * 16])

    #Operations distinct from other networks:
    unpool_binding_activations = tf.reshape(relu2, [-1, 10*10*16])
    gradient_unpool_binding_activations = tf.reshape(pool1_drop, [-1, 14*14*6])

    scalar_dic['gradient_unpool_sparsity'] = tf.math.zero_fraction(gradient_unpool_binding_activations)

    dense1 = tf.nn.bias_add(tf.add(tf.add(tf.matmul(pool2_flat, weights['dense_W1']),
        tf.matmul(unpool_binding_activations, weights['course_bindingW1'])),
        tf.matmul(gradient_unpool_binding_activations, weights['finegrained_bindingW1'])),
        biases['dense_b1'])

    logits, scalar_dic, _, _ = fc_sequence(dense1, dropout_rate_placeholder, weights, biases, scalar_dic)

    AutoEncoder_vars = {}

    l1_reg_activations1 = tf.norm(unpool_binding_activations, ord=1, axis=None)
    l1_reg_activations2 = tf.norm(gradient_unpool_binding_activations, ord=1, axis=None)

    if dynamic_dic['dynamic_var'] == 'Add_logit_noise':
        print("Adding noise to logits")
        #Add noise as a control for Boundary attack resistance being related to e.g. numerical imprecision
        logits = logits + tf.random.normal(tf.shape(logits), mean=0.0, stddev=0.1)


    return logits, AutoEncoder_vars, scalar_dic, l1_reg_activations1, l1_reg_activations2


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

    AutoEncoder_vars = {}

    #We do not regularize activations with L1 norm in the VGG networks, so pass 0
    l1_reg_activations1 = 0
    l1_reg_activations2 = 0

    return logits, AutoEncoder_vars, scalar_dic, l1_reg_activations1, l1_reg_activations2

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

    AutoEncoder_vars = {}

    #We do not regularize activations with L1 norm in the VGG networks, so pass 0
    l1_reg_activations1 = 0
    l1_reg_activations2 = 0

    return logits, AutoEncoder_vars, scalar_dic, l1_reg_activations1, l1_reg_activations2

def unpooling_sequence(pool_drop, pool_indices, relu, relu_flat_shape, dropout_rate_placeholder, scalar_dic):
    
    #Extract binding information for mid-level neurons that are driving the max-pooled (spatially invariant) representations
    unpool_binding_activations = max_unpool(pool_drop, pool_indices, relu)
    unpool_binding_activations_flat = tf.reshape(unpool_binding_activations, relu_flat_shape)

    scalar_dic['unpool_sparsity'] = tf.math.zero_fraction(unpool_binding_activations_flat)

    return unpool_binding_activations_flat, scalar_dic

def gradient_unpooling_sequence(high_level, low_level, low_flat_shape, dropout_rate_placeholder, scalar_dic, dynamic_dic):

    print("\nThe k-winner sparsity of gradient unpooling is set to " + str(dynamic_dic['sparsification_kwinner']))
    print("The dropout (including during testing) sparsity of gradient unpooling is set to " + str(dynamic_dic['sparsification_dropout']))

    #Extract binding information for low-level neurons that are driving critical (i.e. max-pooled) mid-level neurons
    binding_grad = tf.squeeze(tf.gradients(high_level, low_level, unconnected_gradients=tf.UnconnectedGradients.ZERO), 0) #Squeeze removes the dimension of the gradient tensor that stores dtype
    binding_grad_flat = tf.reshape(binding_grad, low_flat_shape)

    #Rather than using k-largest gradients to apply a mask, simply up-project the k-largest activations
    if dynamic_dic['dynamic_var'] == 'kwinner_activations':
        print("Using the k-largest activations rather than k-largest gradients for 'gradient unpooling'.")
        low_level_flat = tf.reshape(low_level, low_flat_shape)
        values, _ = tf.math.top_k(low_level_flat, k=round(low_flat_shape[1]*dynamic_dic['sparsification_kwinner']))
        kth = tf.reduce_min(values, axis=1)
        mask = tf.greater_equal(low_level_flat, tf.expand_dims(kth, -1))
        gradient_unpool_binding_activations = tf.multiply(low_level_flat, tf.dtypes.cast(mask, dtype=tf.float32)) #Apply the Boolean mask element-wise

    #Rather than using the k-largest gradietns to apply a mask, use the k-smallest gradients; serves as a control for the gradient operation somehow being the trick
    elif dynamic_dic['dynamic_var'] == 'kloser_gradients':
        print("Using the k-*smallest* gradients for 'gradient unpooling'.")
        #Note we use the negative sign to find the 'bottom-k'
        values, _ = tf.math.top_k(tf.negative(binding_grad_flat), k=round(low_flat_shape[1]*dynamic_dic['sparsification_kwinner']))
        kth = tf.reduce_max(tf.negative(values), axis=1)
        mask = tf.less_equal(binding_grad_flat, tf.expand_dims(kth, -1))
        low_level_flat = tf.reshape(low_level, low_flat_shape) 
        gradient_unpool_binding_activations = tf.multiply(low_level_flat, tf.dtypes.cast(mask, dtype=tf.float32)) #Apply the Boolean mask element-wise

    else:
        #Use k-th largest value as a threshold for getting a boolean mask
        #K is typicall selected for approx top 10-15% gradients
        values, _ = tf.math.top_k(binding_grad_flat, k=round(low_flat_shape[1]*dynamic_dic['sparsification_kwinner']))
        kth = tf.reduce_min(values, axis=1)
        mask = tf.greater_equal(binding_grad_flat, tf.expand_dims(kth, -1))
        low_level_flat = tf.reshape(low_level, low_flat_shape) 
        gradient_unpool_binding_activations = tf.multiply(low_level_flat, tf.dtypes.cast(mask, dtype=tf.float32)) #Apply the Boolean mask element-wise


    #Apply drop-out to measure the effect of stochastic sparsity; note this dropout, if set above 0, is always applied (including during testing)
    gradient_unpool_binding_activations = tf.nn.dropout(gradient_unpool_binding_activations, rate=dynamic_dic['sparsification_dropout'])

    # tf.compat.v1.summary.histogram('Gradient_unpooling_activations', gradient_unpool_binding_activations)

    scalar_dic['gradient_unpool_sparsity'] = tf.math.zero_fraction(gradient_unpool_binding_activations)

    return gradient_unpool_binding_activations, scalar_dic

def conv1_sequence(features, dropout_rate_placeholder, weights, biases, scalar_dic):

    conv1 = tf.nn.conv2d(input=tf.dtypes.cast(features, dtype=tf.float32), filter=weights['conv_W1'], 
                         strides=[1, 1, 1, 1], padding="SAME")
    conv1 = tf.nn.bias_add(conv1, biases['conv_b1'])
    conv1_drop = tf.nn.dropout(conv1, rate=dropout_rate_placeholder)
    relu1 = tf.nn.relu(conv1_drop)
    scalar_dic['relu1_sparsity'] = tf.math.zero_fraction(relu1)
    pool1, pool1_indices = tf.nn.max_pool_with_argmax(relu1, ksize=(1,2,2,1), strides=(1,2,2,1), padding="VALID")
    pool1_drop = tf.nn.dropout(pool1, rate=dropout_rate_placeholder)

    return pool1_drop, pool1_indices, relu1, scalar_dic

def conv2_sequence(pool1_drop, dropout_rate_placeholder, weights, biases, scalar_dic):

    conv2 = tf.nn.conv2d(pool1_drop, weights['conv_W2'], strides=[1,1,1,1], padding="VALID")
    conv2 = tf.nn.bias_add(conv2, biases['conv_b2'])
    conv2_drop = tf.nn.dropout(conv2, rate=dropout_rate_placeholder)
    relu2 = tf.nn.relu(conv2_drop)
    scalar_dic['relu2_sparsity'] = tf.math.zero_fraction(relu2)
    pool2, pool2_indices = tf.nn.max_pool_with_argmax(relu2, ksize=(1,2,2,1), strides=(1,2,2,1), padding="VALID")
    pool2_drop = tf.nn.dropout(pool2, rate=dropout_rate_placeholder)
    scalar_dic['pool2_sparsity'] = tf.math.zero_fraction(pool2_drop)

    return pool2_drop, pool2_indices, relu2, scalar_dic

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

def AutoEncoder(features, dropout_rate_placeholder, weights, biases, decoder_weights, dynamic_dic, prediction_fun):

    print("\n***Building super-vised auto-encoder meta-architecture***\n")

    predictions, AutoEncoder_vars, _, _, _ = globals()[prediction_fun](features, dropout_rate_placeholder, weights, biases, dynamic_dic)

    #Note variable names are named using descending rather than ascending order
    #Decode FC layers
    de_dense2 = tf.matmul(AutoEncoder_vars['latent_rep'], decoder_weights['de_dense_W2'])
    de_dense1 = tf.matmul(de_dense2, decoder_weights['de_dense_W1'])

    #Reshape the flattened representation
    de_dense1 = tf.reshape(de_dense1, [-1, 5, 5, 16])

    #Decode pooling and convolutional layers
    de_unpooling2 = max_unpool(de_dense1, AutoEncoder_vars['pool2_indices'], 
        AutoEncoder_vars['unpool2_shape'])

    #Successive upsampling and convolution are preferable to deconvolution, as they reduce artefacts such as 'checker-boarding'
    upsample_2 = tf.image.resize_images(de_unpooling2, size=(14,14), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    de_conv2 = tf.nn.conv2d(upsample_2, decoder_weights['de_conv_W2'], strides=[1,1,1,1], padding="SAME") #Note same rather than valid padding is used in decoding

    de_unpooling1 = max_unpool(de_conv2, AutoEncoder_vars['pool1_indices'], 
        AutoEncoder_vars['unpool1_shape'])

    upsample_1 = tf.image.resize_images(de_unpooling1, size=(28,28), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    reconstruction_logits = tf.nn.conv2d(upsample_1, decoder_weights['de_conv_W1'], strides=[1,1,1,1], padding="SAME")

    return predictions, reconstruction_logits

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
    weights, biases, decoder_weights, x_placeholder, y_placeholder, dropout_rate_placeholder):

    if params['meta_architecture'] == 'CNN':
        if params['architecture'] == 'LeNet':
            predictions, AutoEncoder_vars, scalar_dic, l1_reg_activations1, l1_reg_activations2 = LeNet_predictions(x_placeholder, dropout_rate_placeholder, weights, biases, params['dynamic_dic']) 
            print("Adding an L1 regularization of " + str(params['L1_regularization_activations1']) + " to the penultimate LeNet fully-connected layer.")
        elif params['architecture'] == 'SchottCNN':
            predictions, AutoEncoder_vars, scalar_dic, l1_reg_activations1, l1_reg_activations2 = SchottCNN_predictions(x_placeholder, dropout_rate_placeholder, weights, biases, params['dynamic_dic']) 
        elif params['architecture'] == 'PixelCNN':
            predictions, AutoEncoder_vars, scalar_dic, l1_reg_activations1, l1_reg_activations2 = PixelCNN_predictions(x_placeholder, dropout_rate_placeholder, weights, biases, params['dynamic_dic']) 
        elif params['architecture'] == 'VGG':
            predictions, AutoEncoder_vars, scalar_dic, l1_reg_activations1, l1_reg_activations2 = VGG_predictions(x_placeholder, dropout_rate_placeholder, weights, biases, params['dynamic_dic'])
        elif params['architecture'] == 'BindingCNN':
            predictions, AutoEncoder_vars, scalar_dic, l1_reg_activations1, l1_reg_activations2 = BindingCNN_predictions(x_placeholder, dropout_rate_placeholder, weights, biases, params['dynamic_dic']) 
        elif params['architecture'] == 'MadryCNN':
            predictions, AutoEncoder_vars, scalar_dic, l1_reg_activations1, l1_reg_activations2 = MadryCNN_predictions(x_placeholder, dropout_rate_placeholder, weights, biases, params['dynamic_dic']) 
        elif params['architecture'] == 'BindingVGG':
            predictions, AutoEncoder_vars, scalar_dic, l1_reg_activations1, l1_reg_activations2 = BindingVGG_predictions(x_placeholder, dropout_rate_placeholder, weights, biases, params['dynamic_dic']) 
        elif params['architecture'] == 'controlCNN':
            predictions, AutoEncoder_vars, scalar_dic, l1_reg_activations1, l1_reg_activations2 = controlCNN_predictions(x_placeholder, dropout_rate_placeholder, weights, biases, params['dynamic_dic']) 

        cost = (tf.reduce_mean(tf.compat.v1.losses.softmax_cross_entropy(logits=predictions, onehot_labels=y_placeholder, label_smoothing=params['label_smoothing'])) + 
            params['L1_regularization_activations1']*l1_reg_activations1 + params['L1_regularization_activations2']*l1_reg_activations2) +  tf.losses.get_regularization_loss()

        correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(y_placeholder, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        total_accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.float32)) #Used to store batched accuracy for evaluating on the entire test dataset
        
    elif params['meta_architecture'] == 'SAE':
        predictions, reconstruction_logits = AutoEncoder(x_placeholder, dropout_rate_placeholder, weights, biases, decoder_weights, 
            dynamic_dic=params['dynamic_dic'], prediction_fun=params['architecture']+'_predictions') 
       
        cost = (params['predictive_weighting'] * tf.reduce_mean(tf.compat.v1.losses.softmax_cross_entropy(logits=predictions, onehot_labels=y_placeholder, 
            label_smoothing=params['label_smoothing'])) 
            + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.dtypes.cast(x_placeholder, dtype=tf.float32), logits=reconstruction_logits))) 

        correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(y_placeholder, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        total_accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.float32)) #Used to store batched accuracy for evaluating on the entire test dataset

        #Pass reconstructed image through sigmoid to recover pixel values
        decoded = tf.nn.sigmoid(reconstruction_logits)

    #Create the chosen optimizer with tf.train.Adam..., then add it to the graph with .minimize
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

        #Initialize variables; note the requirement for explicit initialization prevents expensive
        #initializers from being re-run when e.g. reloading a model from a checkpoint
        sess.run(tf.compat.v1.global_variables_initializer())

        network_name_str = str(iter_num) + params['architecture'] + '_adver_trained_' + str(params['adver_trained'])
        print("\n\nTraining " + network_name_str)
        training_writer = tf.compat.v1.summary.FileWriter('tensorboard_data/tb_' + network_name_str + '/training', sess.graph)
        testing_writer = tf.compat.v1.summary.FileWriter('tensorboard_data/tb_' + network_name_str + '/testing')

        for epoch in range(params['training_epochs']):
            if params['dataset'] == 'cifar10':
                #Use data augmentation
                batches=0
                datagen = ImageDataGenerator(width_shift_range=params['shift_range'], height_shift_range=params['shift_range'], horizontal_flip=True)
                for batch_x, batch_y in datagen.flow(training_data, training_labels, batch_size=params['batch_size']):
                    batches += 1
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
                    print("Adding Gaussian noise *to the first half* of training data")
                    #Also clip the data after noise
                    shuffled_training_data[:half_point] = np.clip(shuffled_training_data[:half_point] + np.random.normal(0, scale=params['Gaussian_noise'], size=np.shape(shuffled_training_data[:half_point])), 0, 1)

                if params['salt&pepper_noise'] != None:
                    print("Adding salt & pepper noise to *the second half* of training data")
        
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
        if params['dataset']!='cifar10' and params['meta_architecture']!='SAE':
            testing_sparsity = sess.run(scalar_dic, feed_dict={x_placeholder: testing_data, y_placeholder: testing_labels, dropout_rate_placeholder : 0.0})

            print("\nLayer-wise sparsity:")
            print(testing_sparsity)

            print("Mean sparsity is " + str(np.mean(np.fromiter(testing_sparsity.values(), dtype=float))))
        else:
            testing_sparsity = {'NA':-1.0}

        training_writer.close()
        testing_writer.close()

        return training_acc, testing_acc, network_name_str, testing_sparsity


if __name__ == '__main__':

    params = {'architecture':'BindingCNN',
    'dynamic_dic':{'dynamic_var':'None', 'sparsification_kwinner':0.15, 'sparsification_dropout':0.0},
    'dataset':'mnist',
    'meta_architecture':'CNN',
    'training_epochs':3,
    'adver_trained':False,
    'crossval_bool':False,
    'dropout_rate':0.25,
    'predictive_weighting':0.01,
    'label_smoothing':0.1,
    'L1_regularization_activations1':0.0,
    'L1_regularization_activations2':0.0,
    'learning_rate':0.001,
    'batch_size':128} #NB that drop-out 'rate' = 1 - 'keep probability'

    (training_data, training_labels, testing_data, testing_labels, _, _) = data_setup(params)

    x, y, dropout_rate_placeholder, var_list, weights, biases, decoder_weights = initializer_fun(params, training_data, training_labels)
    iter_num = 0

    network_train(params, iter_num, var_list, training_data, training_labels, testing_data, testing_labels, weights, biases, decoder_weights,
     x_placeholder=x, y_placeholder=y, dropout_rate_placeholder=dropout_rate_placeholder)


