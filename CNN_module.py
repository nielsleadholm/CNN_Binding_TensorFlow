#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt
import os
import tfCore_adversarial_attacks as atk

#Temporarily disable deprecation warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

def data_setup(crossval_boolean):
    #Load the full MNIST dataset
    #Note the shape of the images required by the custom CNNs is 2D, rather than flattened as for the Madry model
    (training_data, training_labels), (testing_data, testing_labels) = mnist.load_data()

    # print("*type* Training data dtype is " + str(training_data.dtype))
    # print("Testing data dtype is " + str(testing_data.dtype))
    #Rescale images to values between 0:1 and reshape so each image is 28x28
    training_data = training_data/255
    training_data = np.reshape(training_data, [np.shape(training_data)[0], 28, 28, 1])

    testing_data = testing_data/255
    testing_data = np.reshape(testing_data, [np.shape(testing_data)[0], 28, 28, 1])

    # print("*type* Training data dtype is " + str(training_data.dtype))
    # print("Testing data dtype is " + str(testing_data.dtype))

    #Transform the labels into one-hot encoding
    training_labels = np.eye(10)[training_labels.astype(int)]
    testing_labels = np.eye(10)[testing_labels.astype(int)]

    if crossval_boolean == 1:
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

def initializer_fun(params, training_data, training_labels):

    tf.reset_default_graph() #Re-set the default graph to clear previous e.g. variable assignments

    # Declare placeholders for the input features and labels
    # The first dimension of the palceholder shape is set to None as this will later be defined by the batch size
    with tf.name_scope('Input'):
        x = tf.placeholder(training_data.dtype, [None, 28, 28, 1], name='x-input')
        y = tf.placeholder(training_labels.dtype, [None, 10], name='y-input')

    with tf.name_scope('Drop-Out'):
        dropout_rate_placeholder = tf.placeholder(tf.float32)
        tf.summary.scalar('Dropout_Rate', dropout_rate_placeholder)

    #Define weight and bias variables, and initialize values
    initializer = tf.glorot_uniform_initializer()
    regularizer_l1 = tf.contrib.layers.l1_regularizer(scale=params['L1_regularization_scale'])
    regularizer_l2 = tf.contrib.layers.l2_regularizer(scale=params['L2_regularization_scale'])

    if params['architecture'] == 'LeNet':
        dense_dimension = 400
    elif params['architecture'] == 'BindingCNN':
        dense_dimension = 400+1600+1176
    elif params['architecture'] == 'BindingCNN_unpool':
        dense_dimension = 400+1600

    #Note for example that the first convolutional weights layer has a 5x5 filter with 1 input channel, and 6 output channels
    #tf.get_variable will either get an existing variable with these parameters, or otherwise create a new one
    with tf.name_scope('Weights'):
        weights = {
        'conv_W1' : tf.get_variable('CW1', shape=(5, 5, 1, 6), initializer=initializer, regularizer=regularizer_l1),
        'conv_W2' : tf.get_variable('CW2', shape=(5, 5, 6, 16), initializer=initializer, regularizer=regularizer_l1),
        'dense_W1' : tf.get_variable('DW1', shape=(dense_dimension, 120), initializer=initializer, regularizer=regularizer_l2),
        'dense_W2' : tf.get_variable('DW2', shape=(120, 84), initializer=initializer, regularizer=regularizer_l2),
        'output_W' : tf.get_variable('OW', shape=(84, 10), initializer=initializer, regularizer=regularizer_l2)
        }

        #Add summaries for each weight variable in the dictionary, for later use in TensorBoard
        for weights_var in weights.values():
            var_summaries(weights_var)

    with tf.name_scope('Biases'):
        biases = {
        'conv_b1' : tf.get_variable('Cb1', shape=(6), initializer=initializer, regularizer=regularizer_l1),
        'conv_b2' : tf.get_variable('Cb2', shape=(16), initializer=initializer, regularizer=regularizer_l1),
        'dense_b1' : tf.get_variable('Db1', shape=(120), initializer=initializer, regularizer=regularizer_l2),
        'dense_b2' : tf.get_variable('Db2', shape=(84), initializer=initializer, regularizer=regularizer_l2),
        'output_b' : tf.get_variable('Ob', shape=(10), initializer=initializer, regularizer=regularizer_l2)
        }

        for biases_var in biases.values():
            var_summaries(biases_var)

    var_list = [weights['conv_W1'], weights['conv_W2'], weights['dense_W1'], weights['dense_W2'], 
    weights['output_W'], biases['conv_b1'], biases['conv_b2'], biases['dense_b1'], 
    biases['dense_b2'], biases['output_b']]

    return x, y, dropout_rate_placeholder, var_list, weights, biases


#Define the model
def LeNet_predictions(features, dropout_rate_placeholder, weights, biases):
    conv1 = tf.nn.conv2d(input=tf.dtypes.cast(features, dtype=tf.float32), filter=weights['conv_W1'],
                         strides=[1, 1, 1, 1], padding="SAME")
    conv1 = tf.nn.bias_add(conv1, biases['conv_b1'])
    conv1_drop = tf.nn.dropout(conv1, rate=dropout_rate_placeholder)
    relu1 = tf.nn.relu(conv1_drop)
    tf.summary.histogram('Relu1_activations', relu1)
    pool1 = tf.nn.max_pool(relu1, ksize=(1,2,2,1), strides=(1,2,2,1), padding="VALID")
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
    pool2 = tf.nn.max_pool(relu2, ksize=(1,2,2,1), strides=(1,2,2,1), padding="VALID")
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

    logits = tf.add(tf.matmul(dense2_drop, weights['output_W']), biases['output_b'])

    return logits



#Define the convolutional model now with binding information
def BindingCNN_predictions(features, dropout_rate_placeholder, weights, biases):

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
    #Note in the tuple defining strides for max_pool, the first entry is always 1 as this refers to the batches/indexed images
    
    #Multiply pool1 element-wise by a tensor of 1's to simplify later gradient calculation
    pool1_identity_w = tf.ones(tf.shape(pool1_drop))
    pool1_drop = tf.multiply(pool1_drop, pool1_identity_w)
    
    #Continue standard feed-forward calculations
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

    #Extract binding information for low-level neurons that are driving critical mid-level neurons
    #Specifically, the gradients provide the degree to which neurons in pool1, given the weights of conv2, contributed to the neurons that were eventually max-pooled
    binding_grad = tf.squeeze(tf.gradients(invariant_pool2_drop, pool1_identity_w), 0) #Squeeze removes the dimension of the gradient tensor that stores dtype
    #Creates a mask using all of these extracted neurons that had a non-zero influence; assuming a sufficient degree of sparsity in the activations, all of these should be 'important'
    mask = tf.dtypes.cast(tf.not_equal(binding_grad, 0), dtype=tf.float32)
    early_binding_activations = tf.multiply(pool1_drop, mask) #Apply the Boolean mask element-wise to just the pool1 activations (i.e. not including conv2 transformation)
    early_binding_activations_flat = tf.reshape(early_binding_activations, [-1, 14 * 14 * 6])
    #Record information for TensorBoard visualization
    early_binding_activations_sparsity = tf.math.zero_fraction(early_binding_activations_flat)
    tf.summary.scalar('Early_binding_sparsness', early_binding_activations_sparsity)
    tf.summary.histogram('Early_binding_activations', early_binding_activations_flat)

    #Continue standard feed-forward calculations, but with binding information projected upwards (concatinated)
    dense1 = tf.add(tf.matmul(tf.concat([invariant_pool2_drop_flat, invariant_binding_activations_flat, 
                                         early_binding_activations_flat], axis=1), 
                              weights['dense_W1']), biases['dense_b1'])
    dense1_drop = tf.nn.dropout(dense1, rate=dropout_rate_placeholder)
    dense1_relu = tf.nn.relu(dense1_drop)
    tf.summary.histogram('Dense1_activations', dense1_relu)
    dense2 = tf.add(tf.matmul(dense1_relu, weights['dense_W2']), biases['dense_b2'])
    dense2_drop = tf.nn.dropout(dense2, rate=dropout_rate_placeholder)
    dense2_relu = tf.nn.relu(dense2_drop)
    tf.summary.histogram('Dense2_activations', dense2_relu)

    logits = tf.add(tf.matmul(dense2_relu, weights['output_W']), biases['output_b'])

    return logits

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

    #Continue standard feed-forward calculations, but with binding information projected upwards (concatinated)
    dense1 = tf.add(tf.matmul(tf.concat([invariant_pool2_drop_flat, invariant_binding_activations_flat], axis=1), 
                              weights['dense_W1']), biases['dense_b1'])
    dense1_drop = tf.nn.dropout(dense1, rate=dropout_rate_placeholder)
    dense1_relu = tf.nn.relu(dense1_drop)
    tf.summary.histogram('Dense1_activations', dense1_relu)
    dense2 = tf.add(tf.matmul(dense1_relu, weights['dense_W2']), biases['dense_b2'])
    dense2_drop = tf.nn.dropout(dense2, rate=dropout_rate_placeholder)
    dense2_relu = tf.nn.relu(dense2_drop)
    tf.summary.histogram('Dense2_activations', dense2_relu)

    logits = tf.add(tf.matmul(dense2_relu, weights['output_W']), biases['output_b'])

    return logits


#Define max_unpool function, used in the binding CNN
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
def network_train(params, var_list, training_data, training_labels, testing_data, testing_labels, weights, biases, x_placeholder, y_placeholder, dropout_rate_placeholder):

    if params['architecture'] == 'LeNet':
        predictions = LeNet_predictions(x_placeholder, dropout_rate_placeholder, weights, biases) #NB that x was defined earlier with tf.placeholder
    elif params['architecture'] == 'BindingCNN':
        predictions = BindingCNN_predictions(x_placeholder, dropout_rate_placeholder, weights, biases) #NB that x was defined earlier with tf.placeholder
    elif params['architecture'] == 'BindingCNN_unpool':
        predictions = BindingCNN_unpool_predictions(x_placeholder, dropout_rate_placeholder, weights, biases) #NB that x was defined earlier with tf.placeholder
    

    #Define the main Tensors (left hand) and Operations (right hand) that will be used during training
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=y_placeholder)) + tf.losses.get_regularization_loss()
    tf.summary.scalar('Softmax_cross_entropy', cost)

    correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(y_placeholder, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('Accuracy', accuracy)

    #Create the chosen optimizer with tf.train.Adam..., then add it to the graph with .minimize
    optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate']).minimize(cost)

    #Define values to be written with the summary method for later visualization
    loss_summary = tf.summary.scalar(name="Loss_values", tensor=cost)
    accuracy_summary = tf.summary.scalar(name="Accuracy_values", tensor=accuracy)

    #Create a Saver object to enable later re-loading of the learned weights
    saver = tf.train.Saver(var_list)

    #Merge and provide directory for saving TF summaries
    merged = tf.summary.merge_all()

    #Aside on understanding 'with' and 'tf.Session()'
        #Python's 'with' statement enables the evaluation of tf.Session, while ensuring
        #that the associated __exit__ method (similar to e.g. closing a file) will always
        #be executed even if an error is raised
        #tf.Session() provides a connection between the Python program and the C++ runtime
        #It also caches information about the tf.Graph to enable efficient re-use of data
        #As tf.Session owns physical resources (such as the GPU), 'with' is particularly important
    with tf.Session() as sess:

        #Initialize variables; note the requirement for explicit initialization prevents expensive
        #initializers from being re-run when e.g. relaoding a model from a checkpoint
        sess.run(tf.global_variables_initializer())
        network_name_str = (params['architecture'] + '_L1-' + str(params['L1_regularization_scale']) + '_L2-' + str(params['L2_regularization_scale']) + '_drop-' + str(params['dropout_rate']))
        print("Training " + network_name_str)
        training_writer = tf.summary.FileWriter('tensorboard_data/tb_' + network_name_str + '/training', sess.graph)
        testing_writer = tf.summary.FileWriter('tensorboard_data/tb_' + network_name_str + '/testing')

        for epoch in range(params['training_epochs']):

            for batch in range(int(len(training_labels)/params['batch_size'])):

                batch_x = training_data[batch*params['batch_size']:min((batch+1)*params['batch_size'], len(training_labels))]
                batch_y = training_labels[batch*params['batch_size']:min((batch+1)*params['batch_size'], len(training_labels))]

                #Recall that tf.Session.run is the main method for running a tf.Operation or evaluation a tf.Tensor
                #By passing or more Tensors or Operations, TensorFlow will execute the operations needed
                run_optim = sess.run(optimizer, feed_dict = {x_placeholder: batch_x, y_placeholder: batch_y, dropout_rate_placeholder : params['dropout_rate']})

                loss, acc = sess.run([cost, accuracy], feed_dict = {x_placeholder: batch_x, y_placeholder: batch_y, dropout_rate_placeholder : 0.0})

            training_summ, training_acc = sess.run([merged, accuracy], feed_dict={x_placeholder: batch_x, y_placeholder: batch_y, dropout_rate_placeholder : 0.0})
            training_writer.add_summary(training_summ, epoch)

            testing_summ, testing_acc = sess.run([merged, accuracy], feed_dict={x_placeholder: testing_data, y_placeholder: testing_labels, dropout_rate_placeholder : 0.0})
            testing_writer.add_summary(testing_summ, epoch)

            print("At iteration " + str(epoch) + ", Loss = " + \
                 "{:.4f}".format(loss) + ", Training Accuracy = " + \
                                "{:.4f}".format(training_acc) + ", Testing Accuracy = " + \
                                "{:.4f}".format(testing_acc))


        print("Training complete")

        save_path = saver.save(sess, "network_weights_data/" + network_name_str + ".ckpt")

        print("Final testing Accuracy:","{:.5f}".format(testing_acc))

        training_writer.close()
        testing_writer.close()

        return training_acc , testing_acc, network_name_str

#Runs if using the .py file in isolation, to test e.g. a particular network setup
if __name__ == '__main__':

    params = {'architecture':'LeNet',
    'training_epochs':1,
    'dropout_rate':0.5,
    'L1_regularization_scale':0.0,
    'L2_regularization_scale':0.0,
    'learning_rate':0.001,
    'batch_size':128} #NB that drop-out 'rate' = 1 - 'keep probability'

    (training_data, training_labels, testing_data, testing_labels, _, _) = data_setup(crossval_boolean=0)

    x, y, dropout_rate_placeholder, var_list, weights, biases = initializer_fun(params, training_data, training_labels)

    network_train(params, var_list, training_data, training_labels, testing_data, testing_labels, weights, biases, x_placeholder=x, y_placeholder=y, dropout_rate_placeholder=dropout_rate_placeholder)

