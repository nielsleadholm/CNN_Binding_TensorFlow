#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
from keras.datasets import mnist, fashion_mnist
import matplotlib.pyplot as plt
from PIL import Image
import os
import tfCore_adversarial_attacks as atk

#Impelemnts a standard LeNet-5 like CNN and 'Binding CNN' architecture in TensorFlow Core

#Temporarily disable deprecation warnings (using tf 1.14)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 


def data_setup(params):
    #Note the shape of the images required by the custom CNNs is 2D, rather than flattened as for the Madry model
    if params['dataset'] == 'mnist':
        (training_data, training_labels), (testing_data, testing_labels) = mnist.load_data()
    elif params['dataset'] == 'fashion_mnist':
        (training_data, training_labels), (testing_data, testing_labels) = fashion_mnist.load_data()

    #Rescale images to values between 0:1 and reshape so each image is 28x28
    training_data = training_data/255
    training_data = np.reshape(training_data, [np.shape(training_data)[0], 28, 28, 1])

    testing_data = testing_data/255
    testing_data = np.reshape(testing_data, [np.shape(testing_data)[0], 28, 28, 1])

    #Transform the labels into one-hot encoding
    training_labels = np.eye(10)[training_labels.astype(int)]
    testing_labels = np.eye(10)[testing_labels.astype(int)]

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
        tf.summary.scalar('Mean', mean) #The tf.summary operation determines which graph node you would like to annotate, and scalar or histogram the type of summary

        with tf.name_scope('STD'):
            std = tf.sqrt(tf.reduce_mean(tf.square(variable - mean)))

        tf.summary.scalar('STD', std)
        tf.summary.scalar('Max', tf.reduce_max(variable))
        tf.summary.scalar('Min', tf.reduce_min(variable))
        tf.summary.histogram('Histogram', variable)

# # def backu_up_initializer_fun(params, training_data, training_labels):

#     tf.reset_default_graph() #Re-set the default graph to clear previous e.g. variable assignments

#     # Declare placeholders for the input features and labels
#     # The first dimension of the palceholder shape is set to None as this will later be defined by the batch size
#     x = tf.placeholder(training_data.dtype, [None, 28, 28, 1], name='x-input')
#     y = tf.placeholder(training_labels.dtype, [None, 10], name='y-input')

#     dropout_rate_placeholder = tf.placeholder(tf.float32)

#     with tf.variable_scope(params['architecture']):
#         initializer = tf.glorot_uniform_initializer()
#         weights = {
#         'conv_W1' : tf.get_variable('CW1', shape=(5, 5, 1, 6), initializer=initializer),
#         'conv_W2' : tf.get_variable('CW2', shape=(5, 5, 6, 16), initializer=initializer),
#         'dense_W1' : tf.get_variable('DW1', shape=(400, 120), initializer=initializer),
#         'dense_W2' : tf.get_variable('DW2', shape=(120, 84), initializer=initializer),
#         'output_W' : tf.get_variable('OW', shape=(84, 10), initializer=initializer)
#         }
#         weights['course_bindingW1'] = tf.get_variable('courseW1', shape=(1600, 120), initializer=initializer)
#         weights['finegrained_bindingW1'] = tf.get_variable('fineW1', shape=(1176, 120), initializer=initializer)

#         biases = {
#         'conv_b1' : tf.get_variable('Cb1', shape=(6), initializer=initializer),
#         'conv_b2' : tf.get_variable('Cb2', shape=(16), initializer=initializer),
#         'dense_b1' : tf.get_variable('Db1', shape=(120), initializer=initializer),
#         'dense_b2' : tf.get_variable('Db2', shape=(84), initializer=initializer),
#         'output_b' : tf.get_variable('Ob', shape=(10), initializer=initializer)
#         }

#         var_list = [weights['conv_W1'], weights['conv_W2'], weights['dense_W1'], weights['dense_W2'], 
#         weights['output_W'], biases['conv_b1'], biases['conv_b2'], biases['dense_b1'], 
#         biases['dense_b2'], biases['output_b']]

#         decoder_weights = None

#         if params['architecture'] == 'BindingCNN':
#                 var_list.append(weights['course_bindingW1'])
#                 var_list.append(weights['finegrained_bindingW1'])
#         elif params['architecture'] == 'BindingCNN_unpool':
#                 var_list.append(weights['course_bindingW1'])

#     return x, y, dropout_rate_placeholder, var_list, weights, biases, decoder_weights



def initializer_fun(params, training_data, training_labels):

    tf.reset_default_graph() #Re-set the default graph to clear previous e.g. variable assignments

    # Declare placeholders for the input features and labels
    # The first dimension of the palceholder shape is set to None as this will later be defined by the batch size

    x = tf.placeholder(training_data.dtype, [None, 28, 28, 1], name='x-input')
    y = tf.placeholder(training_labels.dtype, [None, 10], name='y-input')

    dropout_rate_placeholder = tf.placeholder(tf.float32)

    #Define weight and bias variables, and initialize values
    initializer = tf.glorot_uniform_initializer()
    regularizer_l1 = tf.contrib.layers.l1_regularizer(scale=params['L1_regularization_scale'])
    
    with tf.variable_scope(params['architecture']):
    #Note for example that the first convolutional weights layer has a 5x5 filter with 1 input channel, and 6 output channels
    #tf.get_variable will either get an existing variable with these parameters, or otherwise create a new one
        weights = {
        'conv_W1' : tf.get_variable('CW1', shape=(5, 5, 1, 6), initializer=initializer, regularizer=regularizer_l1),
        'conv_W2' : tf.get_variable('CW2', shape=(5, 5, 6, 16), initializer=initializer, regularizer=regularizer_l1),
        'dense_W1' : tf.get_variable('DW1', shape=(400, 120), initializer=initializer, regularizer=regularizer_l1),
        'dense_W2' : tf.get_variable('DW2', shape=(120, 84), initializer=initializer, regularizer=regularizer_l1),
        'output_W' : tf.get_variable('OW', shape=(84, 10), initializer=initializer, regularizer=regularizer_l1)
        }
        if params['architecture'] == 'BindingCNN':
            weights['course_bindingW1'] = tf.get_variable('courseW1', shape=(1600, 120), initializer=initializer, regularizer=regularizer_l1)
            weights['finegrained_bindingW1'] = tf.get_variable('fineW1', shape=(1176, 120), initializer=initializer, regularizer=regularizer_l1)
        elif params['architecture'] == 'BindingCNN_unpool':
            weights['course_bindingW1'] = tf.get_variable('courseW1', shape=(1600, 120), initializer=initializer, regularizer=regularizer_l1)

        #Add summaries for each weightseight variable in the dictionary, for later use in TensorBoard
        for weights_var in weights.values():
            var_summaries(weights_var)

        biases = {
        'conv_b1' : tf.get_variable('Cb1', shape=(6), initializer=initializer, regularizer=regularizer_l1),
        'conv_b2' : tf.get_variable('Cb2', shape=(16), initializer=initializer, regularizer=regularizer_l1),
        'dense_b1' : tf.get_variable('Db1', shape=(120), initializer=initializer, regularizer=regularizer_l1),
        'dense_b2' : tf.get_variable('Db2', shape=(84), initializer=initializer, regularizer=regularizer_l1),
        'output_b' : tf.get_variable('Ob', shape=(10), initializer=initializer, regularizer=regularizer_l1)
        }

        for biases_var in biases.values():
            var_summaries(biases_var)

        decoder_weights = {
        'de_conv_W1' : tf.get_variable('de_CW1', shape=(5, 5, 6, 1), initializer=initializer),
        'de_conv_W2' : tf.get_variable('de_CW2', shape=(5, 5, 16, 6), initializer=initializer),
        'de_dense_W1' : tf.get_variable('de_DW1', shape=(120, 400), initializer=initializer),
        'de_dense_W2' : tf.get_variable('de_DW2', shape=(84, 120), initializer=initializer),
        }

        #Add summaries for each weightseight variable in the dictionary, for later use in TensorBoard
        for decoder_weights_var in decoder_weights.values():
            var_summaries(decoder_weights_var)

    var_list = [weights['conv_W1'], weights['conv_W2'], weights['dense_W1'], weights['dense_W2'], 
    weights['output_W'], biases['conv_b1'], biases['conv_b2'], biases['dense_b1'], 
    biases['dense_b2'], biases['output_b']]

    if params['meta_architecture'] == 'auto_encoder':
        var_list.extend([decoder_weights['de_conv_W1'], decoder_weights['de_conv_W2'], 
            decoder_weights['de_dense_W1'], decoder_weights['de_dense_W2']])

    if params['architecture'] == 'BindingCNN':
            var_list.append(weights['course_bindingW1'])
            var_list.append(weights['finegrained_bindingW1'])
    elif params['architecture'] == 'BindingCNN_unpool':
            var_list.append(weights['course_bindingW1'])

    return x, y, dropout_rate_placeholder, var_list, weights, biases, decoder_weights



#Define the model
def LeNet_predictions(features, dropout_rate_placeholder, weights, biases, meta_architecture):
    conv1 = tf.nn.conv2d(input=tf.dtypes.cast(features, dtype=tf.float32), filter=weights['conv_W1'],
                         strides=[1, 1, 1, 1], padding="SAME")
    conv1 = tf.nn.bias_add(conv1, biases['conv_b1'])
    conv1_drop = tf.nn.dropout(conv1, rate=dropout_rate_placeholder)
    relu1 = tf.nn.relu(conv1_drop)
    tf.summary.histogram('Relu1_activations', relu1)
    pool1, pool1_indices = tf.nn.max_pool_with_argmax(relu1, ksize=(1,2,2,1), strides=(1,2,2,1), padding="VALID")
    pool1_drop = tf.nn.dropout(pool1, rate=dropout_rate_placeholder)
    #Note in the tuple defining strides for max_pool, the first entry is always 1 as this refers to the batches/indexed images,
    #rather than the dimensions of a particular image

    conv2 = tf.nn.conv2d(pool1_drop, weights['conv_W2'], strides=[1,1,1,1], padding="VALID")
    conv2 = tf.nn.bias_add(conv2, biases['conv_b2'])
    conv2_drop = tf.nn.dropout(conv2, rate=dropout_rate_placeholder)
    relu2 = tf.nn.relu(conv2_drop)
    relu2_sparsity = tf.math.zero_fraction(relu2)
    tf.summary.scalar('Relu2_sparsness', relu2_sparsity)
    tf.summary.histogram('Relu2_activations', relu2)
    pool2, pool2_indices = tf.nn.max_pool_with_argmax(relu2, ksize=(1,2,2,1), strides=(1,2,2,1), padding="VALID")
    pool2_drop = tf.nn.dropout(pool2, rate=dropout_rate_placeholder)

    #Flatten Pool 2 before connecting it (fully) with the dense layers 1 and 2
    pool2_flat = tf.reshape(pool2, [-1, 5 * 5 * 16])
    dense1 = tf.add(tf.matmul(pool2_flat, weights['dense_W1']), biases['dense_b1'])
    dense1_drop = tf.nn.dropout(dense1, rate=dropout_rate_placeholder)
    dense1_drop = tf.nn.relu(dense1_drop)
    tf.summary.histogram('Dense1_activations', dense1_drop)
    dense2 = tf.add(tf.matmul(dense1_drop, weights['dense_W2']), biases['dense_b2'])
    dense2_drop = tf.nn.dropout(dense2, rate=dropout_rate_placeholder)
    dense2_drop = tf.nn.relu(dense2_drop)
    tf.summary.histogram('Dense2_activations', dense2_drop)

    #Save middle-layer activation of choice for use of any auto-encoder model
    AutoEncoder_vars = {'latent_rep':dense2_drop, 'pool2':pool2, 'pool2_indices':pool2_indices, 
        'unpool2_shape':relu2, 'pool1':pool1, 'pool1_indices':pool1_indices, 'unpool1_shape':relu1}

    logits = tf.add(tf.matmul(dense2_drop, weights['output_W']), biases['output_b'])


    return logits, AutoEncoder_vars

# 
#  **** Refactor this so it's a generic decoder layer that can be used for any of the three input CNNs ***
# 


def AutoEncoder(features, dropout_rate_placeholder, weights, biases, decoder_weights, meta_architecture):

    classification_logits, AutoEncoder_vars = BindingCNN_predictions(features, dropout_rate_placeholder, weights, biases, meta_architecture)

    #Note variable names are named using descending rather than ascending order
    #Decode FC layers
    de_dense2 = tf.matmul(AutoEncoder_vars['latent_rep'], decoder_weights['de_dense_W2'])
    de_dense1 = tf.matmul(de_dense2, decoder_weights['de_dense_W1'])

    #Reshape the flattened representation
    de_dense1 = tf.reshape(de_dense1, [-1, 5, 5, 16])

    #Decode pooling and convolutional layers
    de_unpooling2 = max_unpool(AutoEncoder_vars['pool2'], AutoEncoder_vars['pool2_indices'], 
        AutoEncoder_vars['unpool2_shape'], scope='unpool_2d')

    #Successive upsampling and convolution are preferable to deconvolution, as they reduce artefacts such as 'checker-boarding'
    upsample_2 = tf.image.resize_images(de_unpooling2, size=(14,14), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    de_conv2 = tf.nn.conv2d(upsample_2, decoder_weights['de_conv_W2'], strides=[1,1,1,1], padding="SAME") #Note same rather than valid padding is used in decoding

    de_unpooling1 = max_unpool(AutoEncoder_vars['pool1'], AutoEncoder_vars['pool1_indices'], 
        AutoEncoder_vars['unpool1_shape'], scope='unpool_2d')

    upsample_1 = tf.image.resize_images(de_unpooling1, size=(28,28), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    reconstruction_logits = tf.nn.conv2d(upsample_1, decoder_weights['de_conv_W1'], strides=[1,1,1,1], padding="SAME")

    return classification_logits, reconstruction_logits


#Define the convolutional model now with binding information
def BindingCNN_predictions(features, dropout_rate_placeholder, weights, biases, meta_architecture):

    # # *** add white Gaussian noise ***
    # # this would also end up adding noise to the adversarial examples...
    # features = features + tf.random.normal(tf.shape(features), mean=0.0, stddev=1.0)
    # # with a low probability, save an example image
    # # if np.random.random(0, 1) < 0.01:
    # #     plt.imsave('example' + '.png', features[0, :, :, 0], cmap='gray')

    # # *** add white Gaussian noise ***

    conv1 = tf.nn.conv2d(input=tf.dtypes.cast(features, dtype=tf.float32), filter=weights['conv_W1'], 
                         strides=[1, 1, 1, 1], padding="SAME")
    conv1 = tf.nn.bias_add(conv1, biases['conv_b1'])
    conv1_drop = tf.nn.dropout(conv1, rate=dropout_rate_placeholder)
    relu1 = tf.nn.relu(conv1_drop)
    relu1_sparsity = tf.math.zero_fraction(relu1)
    #tf.summary.scalar('Relu1_sparsness', relu1_sparsity)
    #tf.summary.histogram('Relu1_activations', relu1)
    pool1, pool1_indices = tf.nn.max_pool_with_argmax(relu1, ksize=(1,2,2,1), strides=(1,2,2,1), padding="VALID")
    pool1_drop = tf.nn.dropout(pool1, rate=dropout_rate_placeholder)

    conv2 = tf.nn.conv2d(pool1_drop, weights['conv_W2'], strides=[1,1,1,1], padding="VALID")
    conv2 = tf.nn.bias_add(conv2, biases['conv_b2'])
    conv2_drop = tf.nn.dropout(conv2, rate=dropout_rate_placeholder)
    relu2 = tf.nn.relu(conv2_drop)
    relu2_sparsity = tf.math.zero_fraction(relu2)
    #tf.summary.scalar('Relu2_sparsness', relu2_sparsity)
    #tf.summary.histogram('Relu2_activations', relu2)

    #Perform second max-pooling, and extract binding information for mid-level neurons that are driving the max-pooled (spatially invariant) representations
    invariant_pool2, invariant_binding_indeces = tf.nn.max_pool_with_argmax(relu2, ksize=(1,2,2,1), strides=(1,2,2,1), padding="VALID")
    invariant_pool2_drop = tf.nn.dropout(invariant_pool2, rate=dropout_rate_placeholder)
    invariant_pool2_drop_flat = tf.reshape(invariant_pool2_drop, [-1, 5 * 5 * 16])
    invariant_binding_activations = max_unpool(invariant_pool2, invariant_binding_indeces, relu2, scope='unpool_2d')
    invariant_binding_activations_flat = tf.reshape(invariant_binding_activations, [-1, 10 * 10 * 16])
    #Record information for TensorBoard visualization
    invariant_binding_activations_sparsity = tf.math.zero_fraction(invariant_binding_activations_flat)
    #tf.summary.scalar('Invariant_binding_sparsness', invariant_binding_activations_sparsity)
    #tf.summary.histogram('Invariant_binding_activations', invariant_binding_activations_flat)

    #Extract binding information for low-level neurons that are driving critical mid-level neurons
    binding_grad = tf.squeeze(tf.gradients(invariant_pool2_drop, pool1_drop), 0) #Squeeze removes the dimension of the gradient tensor that stores dtype
    binding_grad_flat = tf.reshape(binding_grad, [-1, 14 * 14 * 6])

    #Using k-th largest value as a threshold for getting a boolean mask
    values, _ = tf.math.top_k(binding_grad_flat, k=400)
    kth = tf.reduce_min(values, axis=1)
    mask = tf.greater_equal(binding_grad_flat, tf.expand_dims(kth, -1))
    pool1_drop_flat = tf.reshape(pool1_drop, [-1, 14 * 14 * 6]) 
    early_binding_activations = tf.multiply(pool1_drop_flat, tf.dtypes.cast(mask, dtype=tf.float32)) #Apply the Boolean mask element-wise to just the pool1 activations (i.e. not including conv2 transformation)

    #Record information for TensorBoard visualization
    #early_binding_activations_sparsity = tf.math.zero_fraction(early_binding_activations_flat)
    #tf.summary.scalar('Early_binding_sparsness', early_binding_activations_sparsity)
    #tf.summary.histogram('Early_binding_activations', early_binding_activations_flat)
    
    #Continue standard feed-forward calculations, but with binding information projected upwards
    dense1 = tf.add(tf.add(tf.add(tf.matmul(invariant_pool2_drop_flat, weights['dense_W1']), biases['dense_b1']),
        tf.add(tf.matmul(invariant_binding_activations_flat, weights['course_bindingW1']), biases['dense_b1'])),
        tf.add(tf.matmul(early_binding_activations, weights['finegrained_bindingW1']), biases['dense_b1']))


    dense1_drop = tf.nn.dropout(dense1, rate=dropout_rate_placeholder)
    dense1_relu = tf.nn.relu(dense1_drop)
    #tf.summary.histogram('Dense1_activations', dense1_relu)
    dense2 = tf.add(tf.matmul(dense1_relu, weights['dense_W2']), biases['dense_b2'])
    dense2_drop = tf.nn.dropout(dense2, rate=dropout_rate_placeholder)
    dense2_relu = tf.nn.relu(dense2_drop)
    #tf.summary.histogram('Dense2_activations', dense2_relu)

    AutoEncoder_vars = {'latent_rep':dense2_relu, 'pool2':invariant_pool2, 'pool2_indices':invariant_binding_indeces, 
        'unpool2_shape':relu2, 'pool1':pool1, 'pool1_indices':pool1_indices, 'unpool1_shape':relu1}

    logits = tf.add(tf.matmul(dense2_relu, weights['output_W']), biases['output_b'])

    
    return logits, AutoEncoder_vars

#Binding CNN without binding mask (just uprojects low level information)
def BindingCNN_control1_predictions(features, dropout_rate_placeholder, weights, biases, meta_architecture):

    print("Using up-projection binding-prediction function.")

    conv1 = tf.nn.conv2d(input=tf.dtypes.cast(features, dtype=tf.float32), filter=weights['conv_W1'], 
                         strides=[1, 1, 1, 1], padding="SAME")
    conv1 = tf.nn.bias_add(conv1, biases['conv_b1'])
    conv1_drop = tf.nn.dropout(conv1, rate=dropout_rate_placeholder)
    relu1 = tf.nn.relu(conv1_drop)
    relu1_sparsity = tf.math.zero_fraction(relu1)
    tf.summary.scalar('Relu1_sparsness', relu1_sparsity)
    tf.summary.histogram('Relu1_activations', relu1)
    pool1, pool1_indices = tf.nn.max_pool_with_argmax(relu1, ksize=(1,2,2,1), strides=(1,2,2,1), padding="VALID")
    pool1_drop = tf.nn.dropout(pool1, rate=dropout_rate_placeholder)

    conv2 = tf.nn.conv2d(pool1_drop, weights['conv_W2'], strides=[1,1,1,1], padding="VALID")
    conv2 = tf.nn.bias_add(conv2, biases['conv_b2'])
    conv2_drop = tf.nn.dropout(conv2, rate=dropout_rate_placeholder)
    relu2 = tf.nn.relu(conv2_drop)
    relu2_sparsity = tf.math.zero_fraction(relu2)
    tf.summary.scalar('Relu2_sparsness', relu2_sparsity)
    tf.summary.histogram('Relu2_activations', relu2)

    #Uprojected information is just lower-level activations without any further processing
    invariant_pool2, invariant_binding_indeces = tf.nn.max_pool_with_argmax(relu2, ksize=(1,2,2,1), strides=(1,2,2,1), padding="VALID")
    invariant_pool2_drop = tf.nn.dropout(invariant_pool2, rate=dropout_rate_placeholder)
    invariant_pool2_drop_flat = tf.reshape(invariant_pool2_drop, [-1, 5 * 5 * 16])
    invariant_binding_activations = relu2
    invariant_binding_activations_flat = tf.reshape(invariant_binding_activations, [-1, 10 * 10 * 16])
    #Record information for TensorBoard visualization
    invariant_binding_activations_sparsity = tf.math.zero_fraction(invariant_binding_activations_flat)
    tf.summary.scalar('Invariant_binding_sparsness', invariant_binding_activations_sparsity)
    tf.summary.histogram('Invariant_binding_activations', invariant_binding_activations_flat)

    #Uprojected information is just lower-level activations without any further processing
    early_binding_activations = pool1_drop
    early_binding_activations_flat = tf.reshape(early_binding_activations, [-1, 14 * 14 * 6])
    #Record information for TensorBoard visualization
    early_binding_activations_sparsity = tf.math.zero_fraction(early_binding_activations_flat)
    tf.summary.scalar('Early_binding_sparsness', early_binding_activations_sparsity)
    tf.summary.histogram('Early_binding_activations', early_binding_activations_flat)
    
    #Continue standard feed-forward calculations, but with binding information projected upwards
    dense1 = tf.add(tf.add(tf.add(tf.matmul(invariant_pool2_drop_flat, weights['dense_W1']), biases['dense_b1']),
        tf.add(tf.matmul(invariant_binding_activations_flat, weights['course_bindingW1']), biases['dense_b1'])),
        tf.add(tf.matmul(early_binding_activations_flat, weights['finegrained_bindingW1']), biases['dense_b1']))


    dense1_drop = tf.nn.dropout(dense1, rate=dropout_rate_placeholder)
    dense1_relu = tf.nn.relu(dense1_drop)
    tf.summary.histogram('Dense1_activations', dense1_relu)
    dense2 = tf.add(tf.matmul(dense1_relu, weights['dense_W2']), biases['dense_b2'])
    dense2_drop = tf.nn.dropout(dense2, rate=dropout_rate_placeholder)
    dense2_relu = tf.nn.relu(dense2_drop)
    tf.summary.histogram('Dense2_activations', dense2_relu)

    AutoEncoder_vars = {'latent_rep':dense2_relu, 'pool2':invariant_pool2, 'pool2_indices':invariant_binding_indeces, 
        'unpool2_shape':relu2, 'pool1':pool1, 'pool1_indices':pool1_indices, 'unpool1_shape':relu1}

    logits = tf.add(tf.matmul(dense2_relu, weights['output_W']), biases['output_b'])
    
    # *** temporary addition of noise to check boundary resistance ***

    #logits = tf.add(logits, tf.random.normal(shape=tf.shape(logits), mean=0, stddev=1.0))

    return logits, AutoEncoder_vars



#Binding CNN architecture with the standard unpooling operation only (i.e. no gradient unpooling)
def BindingCNN_unpool_predictions(features, dropout_rate_placeholder, weights, biases):

    conv1 = tf.nn.conv2d(input=tf.dtypes.cast(features, dtype=tf.float32), filter=weights['conv_W1'], 
                         strides=[1, 1, 1, 1], padding="SAME")
    conv1 = tf.nn.bias_add(conv1, biases['conv_b1'])
    conv1_drop = tf.nn.dropout(conv1, rate=dropout_rate_placeholder)
    relu1 = tf.nn.relu(conv1_drop)
    relu1_sparsity = tf.math.zero_fraction(relu1)
    tf.summary.scalar('Relu1_sparsness', relu1_sparsity)
    tf.summary.histogram('Relu1_activations', relu1)
    pool1 = tf.nn.max_pool(relu1, ksize=(1,2,2,1), strides=(1,2,2,1), padding="VALID")
    pool1_drop = tf.nn.dropout(pool1, rate=dropout_rate_placeholder)
    conv2 = tf.nn.conv2d(pool1_drop, weights['conv_W2'], strides=[1,1,1,1], padding="VALID")
    conv2 = tf.nn.bias_add(conv2, biases['conv_b2'])
    conv2_drop = tf.nn.dropout(conv2, rate=dropout_rate_placeholder)
    relu2 = tf.nn.relu(conv2_drop)
    relu2_sparsity = tf.math.zero_fraction(relu2)
    tf.summary.scalar('Relu2_sparsness', relu2_sparsity)
    tf.summary.histogram('Relu2_activations', relu2)

    #Perform second max-pooling, and extract binding information for mid-level neurons that are driving the max-pooled (spatially invariant) representations
    invariant_pool2, invariant_binding_indeces = tf.nn.max_pool_with_argmax(relu2, ksize=(1,2,2,1), strides=(1,2,2,1), padding="VALID")
    invariant_pool2_drop = tf.nn.dropout(invariant_pool2, rate=dropout_rate_placeholder)
    invariant_pool2_drop_flat = tf.reshape(invariant_pool2_drop, [-1, 5 * 5 * 16])
    invariant_binding_activations = max_unpool(invariant_pool2, invariant_binding_indeces, relu2, scope='unpool_2d')
    invariant_binding_activations_flat = tf.reshape(invariant_binding_activations, [-1, 10 * 10 * 16])
    #Record information for TensorBoard visualization
    invariant_binding_activations_sparsity = tf.math.zero_fraction(invariant_binding_activations_flat)
    tf.summary.scalar('Invariant_binding_sparsness', invariant_binding_activations_sparsity)
    tf.summary.histogram('Invariant_binding_activations', invariant_binding_activations_flat)

    #Continue standard feed-forward calculations, but with binding information projected upwards
    dense1 = tf.add(tf.add(tf.matmul(invariant_pool2_drop_flat, weights['dense_W1']), biases['dense_b1']),
        tf.add(tf.matmul(invariant_binding_activations_flat, weights['course_bindingW1']), biases['dense_b1']))

    dense1_drop = tf.nn.dropout(dense1, rate=dropout_rate_placeholder)
    dense1_relu = tf.nn.relu(dense1_drop)
    tf.summary.histogram('Dense1_activations', dense1_relu)
    dense2 = tf.add(tf.matmul(dense1_relu, weights['dense_W2']), biases['dense_b2'])
    dense2_drop = tf.nn.dropout(dense2, rate=dropout_rate_placeholder)
    dense2_relu = tf.nn.relu(dense2_drop)
    tf.summary.histogram('Dense2_activations', dense2_relu)

    logits = tf.add(tf.matmul(dense2_relu, weights['output_W']), biases['output_b'])

    return logits


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
    with tf.variable_scope(scope):
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

        set_output_shape = [set_input_shape[0], prev_tensor_shape[1], prev_tensor_shape[2], set_input_shape[3]]
        ret.set_shape(set_output_shape)

        return ret

#Primary training function
def network_train(params, iter_num, var_list, training_data, training_labels, testing_data, testing_labels, 
    weights, biases, decoder_weights, x_placeholder, y_placeholder, dropout_rate_placeholder):

    if params['meta_architecture'] == 'CNN':
        if params['architecture'] == 'LeNet':
            predictions, _ = LeNet_predictions(x_placeholder, dropout_rate_placeholder, weights, biases, meta_architecture=params['meta_architecture']) #NB that x was defined earlier with tf.placeholder
        elif params['architecture'] == 'BindingCNN':
            predictions, _ = BindingCNN_predictions(x_placeholder, dropout_rate_placeholder, weights, biases, meta_architecture=params['meta_architecture']) #NB that x was defined earlier with tf.placeholder
        elif params['architecture'] == 'BindingCNN_unpool':
            predictions, _ = BindingCNN_unpool_predictions(x_placeholder, dropout_rate_placeholder, weights, biases, decoder_weights) #NB that x was defined earlier with tf.placeholder  

        #Define the main Tensors (left hand) and Operations (right hand) that will be used during training
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=y_placeholder)) + tf.losses.get_regularization_loss()
        tf.summary.scalar('Softmax_cross_entropy', cost)


        correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(y_placeholder, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('Accuracy', accuracy)
        accuracy_summary = tf.summary.scalar(name="Accuracy_values", tensor=accuracy)
    elif params['meta_architecture'] == 'auto_encoder':
        classification_logits, reconstruction_logits = AutoEncoder(x_placeholder, dropout_rate_placeholder, weights, biases, decoder_weights, meta_architecture=params['meta_architecture'])
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.dtypes.cast(x_placeholder, dtype=tf.float32), 
            logits=reconstruction_logits) + tf.losses.get_regularization_loss()
        cost = tf.reduce_mean(loss)
        accuracy = tf.ones([1]) #Dummy tensor for accuracy
        #Pass reconstructed image through sigmoid to recover pixel values
        decoded = tf.nn.sigmoid(reconstruction_logits)
    elif params['meta_architecture'] == 'SAE':
        classification_logits, reconstruction_logits = AutoEncoder(x_placeholder, dropout_rate_placeholder, weights, biases, decoder_weights, meta_architecture=params['meta_architecture'])
        cost = (params['predictive_weighting'] * tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=classification_logits, labels=y_placeholder)) 
            + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.dtypes.cast(x_placeholder, dtype=tf.float32), logits=reconstruction_logits)) 
            + tf.losses.get_regularization_loss())

        correct_prediction = tf.equal(tf.argmax(classification_logits, 1), tf.argmax(y_placeholder, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('Accuracy', accuracy)
        accuracy_summary = tf.summary.scalar(name="Accuracy_values", tensor=accuracy)

        #Pass reconstructed image through sigmoid to recover pixel values
        decoded = tf.nn.sigmoid(reconstruction_logits)

    #Create the chosen optimizer with tf.train.Adam..., then add it to the graph with .minimize
    optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate']).minimize(cost)

    #Define values to be written with the summary method for later visualization
    loss_summary = tf.summary.scalar(name="Loss_values", tensor=cost)

    #Create a Saver object to enable later re-loading of the learned weights
    saver = tf.train.Saver(var_list)

    #Merge and provide directory for saving TF summaries
    merged = tf.summary.merge_all()

    with tf.Session() as sess:

        #Initialize variables; note the requirement for explicit initialization prevents expensive
        #initializers from being re-run when e.g. relaoding a model from a checkpoint
        sess.run(tf.global_variables_initializer())
        network_name_str = (str(iter_num) + params['architecture'] + '_L1-' + str(params['L1_regularization_scale']) + '_L2-' + str(params['L2_regularization_scale']) + '_drop-' + str(params['dropout_rate']))
        print("Training " + network_name_str)
        training_writer = tf.summary.FileWriter('tensorboard_data/tb_' + network_name_str + '/training', sess.graph)
        testing_writer = tf.summary.FileWriter('tensorboard_data/tb_' + network_name_str + '/testing')

        for epoch in range(params['training_epochs']):

            for batch in range(int(len(training_labels)/params['batch_size'])):

                batch_x = training_data[batch*params['batch_size']:min((batch+1)*params['batch_size'], len(training_labels))]
                #if params['meta_architecture'] == 'CNN':
                batch_y = training_labels[batch*params['batch_size']:min((batch+1)*params['batch_size'], len(training_labels))]
                
                #else:
                #    batch_y = training_data[batch*params['batch_size']:min((batch+1)*params['batch_size'], len(training_labels))]
                
                #Recall that tf.Session.run is the main method for running a tf.Operation or evaluation a tf.Tensor
                #By passing or more Tensors or Operations, TensorFlow will execute the operations needed
                run_optim = sess.run(optimizer, feed_dict = {x_placeholder: batch_x, y_placeholder: batch_y, dropout_rate_placeholder : params['dropout_rate']})

                loss, acc = sess.run([cost, accuracy], feed_dict = {x_placeholder: batch_x, y_placeholder: batch_y, dropout_rate_placeholder : 0.0})

            training_summ, training_acc = sess.run([merged, accuracy], feed_dict={x_placeholder: batch_x, y_placeholder: batch_y, dropout_rate_placeholder : 0.0})
            training_writer.add_summary(training_summ, epoch)

            testing_summ, testing_acc = sess.run([merged, accuracy], feed_dict={x_placeholder: testing_data, y_placeholder: testing_labels, dropout_rate_placeholder : 0.0})

            if params['meta_architecture'] == 'auto_encoder':
                training_acc, testing_acc = 1.0, 1.0

            # if params['meta_architecture'] != 'CNN':
            #     #Save an example reconstructed image; save as many as there are training epochs
            #     image_example = sess.run(decoded, feed_dict = {x_placeholder: testing_data, y_placeholder: testing_labels, dropout_rate_placeholder : 0.0})
            #     for ii in range(params['training_epochs']):
            #         plt.imsave('reconstruction' + str(ii) + '_epoch' + str(epoch) + '.png', image_example[ii, :, :, 0], cmap='gray')

            testing_writer.add_summary(testing_summ, epoch)

            print("At iteration " + str(epoch) + ", Loss = " + \
                 "{:.4f}".format(loss) + ", Training Accuracy = " + \
                                "{:.4f}".format(training_acc) + ", Testing Accuracy = " + \
                                "{:.4f}".format(testing_acc))


        print("Training complete")

        # if params['meta_architecture'] != 'CNN':
        #     for ii in range(params['training_epochs']):
        #         plt.imsave('reconstruction' + str(ii) + '_original.png', testing_data[ii, :, :, 0], cmap='gray')

        save_path = saver.save(sess, "network_weights_data/" + network_name_str + ".ckpt")

        print("Final testing Accuracy:","{:.5f}".format(testing_acc))

        training_writer.close()
        testing_writer.close()

        return training_acc , testing_acc, network_name_str

#Runs if using the .py file in isolation, to test e.g. a particular network setup
if __name__ == '__main__':

    params = {'architecture':'BindingCNN',
    'meta_architecture':'CNN',
    'dataset':'mnist',
    'training_epochs':2,
    'crossval_bool':False,
    'dropout_rate':0.25,
    'predictive_weighting':0.01,
    'L1_regularization_scale':0.0,
    'L2_regularization_scale':0.0,
    'learning_rate':0.001,
    'batch_size':128} #NB that drop-out 'rate' = 1 - 'keep probability'

    (training_data, training_labels, testing_data, testing_labels, _, _) = data_setup(params)

    x, y, dropout_rate_placeholder, var_list, weights, biases, decoder_weights = initializer_fun(params, training_data, training_labels)
    iter_num = 0

    network_train(params, iter_num, var_list, training_data, training_labels, testing_data, testing_labels, weights, biases, decoder_weights, x_placeholder=x, y_placeholder=y, dropout_rate_placeholder=dropout_rate_placeholder)


