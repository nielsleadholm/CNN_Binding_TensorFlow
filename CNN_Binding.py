#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt
import os

#The following implements a simple CNN based on the architecture of LeNet-5, for the MNIST dataset
#Further modifications are intended to implement 'hierarchical feature binding' in a deep learning context
#For information on hierarchical feature binding, see the paper "The Emergence of Polychronization and Feature 
#Binding in a Spiking Neural Network Model of the Primate Ventral Visual System", Eguchi, 2018 at
#(http://psycnet.apa.org/fulltext/2018-25960-001.html)

#Temporarily disable deprecation warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

params = {'training_epochs':30,
    'dropout_rate':0.0,
    'L1_regularization_scale':0.0,
    'L2_regularization_scale':0.0,
    'learning_rate':0.001,
    'batch_size':128,
    'num_classes':10} #NB that drop-out 'rate' = 1 - keep probability


def data_setup(params):
    #Load the full MNIST dataset
    (training_data, training_labels), (testing_data, testing_labels) = mnist.load_data()

    #Rescale images to values between 0:1 and reshape so each image is 28x28
    training_data = training_data/255
    training_data = np.reshape(training_data, [np.shape(training_data)[0], 28, 28, 1])

    testing_data = testing_data/255
    testing_data = np.reshape(testing_data, [np.shape(testing_data)[0], 28, 28, 1])

    #Transform the labels into one-hot encoding
    training_labels = np.eye(params['num_classes'])[training_labels.astype(int)]
    testing_labels = np.eye(params['num_classes'])[testing_labels.astype(int)]

    return (training_data, training_labels, testing_data, testing_labels)

#Define a summary variables function for later visualisation of the network in TensorBoard
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

(training_data, training_labels, testing_data, testing_labels) = data_setup(params)

# Declare placeholders for the input features and labels
# The first dimension of the palceholder shape is set to None as this will later be defined by the batch size
with tf.name_scope('Input'):
    x = tf.placeholder(training_data.dtype, [None, 28, 28, 1], name='x-input')
    y = tf.placeholder(training_labels.dtype, [None, params['num_classes']], name='y-input')

with tf.name_scope('Drop-Out'):
    dropout_rate_placeholder = tf.placeholder(tf.float32)
    tf.summary.scalar('Dropout_Rate', dropout_rate_placeholder)

#Define weight and bias variables, and initialize values
initializer = tf.glorot_uniform_initializer()
regularizer_l1 = tf.contrib.layers.l1_regularizer(scale=params['L1_regularization_scale'])
regularizer_l2 = tf.contrib.layers.l2_regularizer(scale=params['L2_regularization_scale'])
#Note for example that the first convolutional weights layer has a 5x5 filter with 1 input channel, and 6 output channels
#tf.get_variable will either get an existing variable with these parameters, or otherwise create a new one
with tf.name_scope('Weights'):
    weights_Binding = {
        'conv_W1_bind' : tf.get_variable('CW1_bind', shape=(5, 5, 1, 6), initializer=initializer, regularizer=regularizer_l1),
        'conv_W2_bind' : tf.get_variable('CW2_bind', shape=(5, 5, 6, 16), initializer=initializer, regularizer=regularizer_l1),
        'dense_W1_bind' : tf.get_variable('DW1_bind', shape=(400+1600+1176, 120), initializer=initializer, regularizer=regularizer_l2),
        'dense_W2_bind' : tf.get_variable('DW2_bind', shape=(120, 84), initializer=initializer, regularizer=regularizer_l2),
        'output_W_bind' : tf.get_variable('OW_bind', shape=(84, params['num_classes']), initializer=initializer, regularizer=regularizer_l2)
    }
    #Add summaries for each weight variable in the dictionary, for later use in TensorBoard
    for weights_var in weights_Binding.values():
        var_summaries(weights_var)

with tf.name_scope('Biases'):

    biases_Binding = {
        'conv_b1_bind' : tf.get_variable('Cb1_bind', shape=(6), initializer=initializer, regularizer=regularizer_l1),
        'conv_b2_bind' : tf.get_variable('Cb2_bind', shape=(16), initializer=initializer, regularizer=regularizer_l1),
        'dense_b1_bind' : tf.get_variable('Db1_bind', shape=(120), initializer=initializer, regularizer=regularizer_l2),
        'dense_b2_bind' : tf.get_variable('Db2_bind', shape=(84), initializer=initializer, regularizer=regularizer_l2),
        'output_b_bind' : tf.get_variable('Ob_bind', shape=(params['num_classes']), initializer=initializer, regularizer=regularizer_l2)
    }

    for biases_var in biases_Binding.values():
        var_summaries(biases_var)

var_list_Binding = [weights_Binding['conv_W1_bind'], weights_Binding['conv_W2_bind'], weights_Binding['dense_W1_bind'], 
                    weights_Binding['dense_W2_bind'], weights_Binding['output_W_bind'], biases_Binding['conv_b1_bind'], 
                    biases_Binding['conv_b2_bind'], biases_Binding['dense_b1_bind'], biases_Binding['dense_b2_bind'], 
                    biases_Binding['output_b_bind']]

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

#Define the convolutional model now with binding information
def cnn_binding_predictions(features, dropout_rate_placeholder):

    conv1 = tf.nn.conv2d(input=tf.dtypes.cast(features, dtype=tf.float32), filter=weights_Binding['conv_W1_bind'], 
                         strides=[1, 1, 1, 1], padding="SAME")
    conv1 = tf.nn.bias_add(conv1, biases_Binding['conv_b1_bind'])
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
    conv2 = tf.nn.conv2d(pool1_drop, weights_Binding['conv_W2_bind'], strides=[1,1,1,1], padding="VALID")
    conv2 = tf.nn.bias_add(conv2, biases_Binding['conv_b2_bind'])
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
                              weights_Binding['dense_W1_bind']), biases_Binding['dense_b1_bind'])
    dense1_drop = tf.nn.dropout(dense1, rate=dropout_rate_placeholder)
    dense1_relu = tf.nn.relu(dense1_drop)
    tf.summary.histogram('Dense1_activations', dense1_relu)
    dense2 = tf.add(tf.matmul(dense1_relu, weights_Binding['dense_W2_bind']), biases_Binding['dense_b2_bind'])
    dense2_drop = tf.nn.dropout(dense2, rate=dropout_rate_placeholder)
    dense2_relu = tf.nn.relu(dense2_drop)
    tf.summary.histogram('Dense2_activations', dense2_relu)

    logits = tf.add(tf.matmul(dense2_relu, weights_Binding['output_W_bind']), biases_Binding['output_b_bind'])

    return logits


#Define the training function of the new Binding-CNN
def BindingNet_train(params, var_list, training_data, training_labels, testing_data, testing_labels):
    
    predictions = cnn_binding_predictions(x, dropout_rate_placeholder) #NB that x was defined earlier with tf.placeholder
        
    #Define the main Tensors (left hand) and Operations (right hand) that will be used during training
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=y)) + tf.losses.get_regularization_loss()
    correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    #Create the chosen optimizer with tf.train.Adam..., then add it to the graph with .minimize
    optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate']).minimize(cost)
    
    #Define values to be written with the summary method for later visualization
    loss_summary = tf.summary.scalar(name="Loss_values", tensor=cost)
    accuracy_summary = tf.summary.scalar(name="Accuracy_values", tensor=accuracy)
    
    saver = tf.train.Saver(var_list)
    
    #Merge summaries for TensorBoard
    merged = tf.summary.merge_all()
    
    #Carry out training
    with tf.Session() as sess:
        #Initialize the new variables
        sess.run(tf.global_variables_initializer())
        training_writer = tf.summary.FileWriter('tb_Binding/training', sess.graph)
        testing_writer = tf.summary.FileWriter('tb_Binding/testing')

        for epoch in range(params['training_epochs']):

            for batch in range(int(len(training_labels)/params['batch_size'])):

                batch_x = training_data[batch*params['batch_size']:min((batch+1)*params['batch_size'], len(training_labels))]
                batch_y = training_labels[batch*params['batch_size']:min((batch+1)*params['batch_size'], len(training_labels))]
                
                #Recall that tf.Session.run is the main method for running a tf.Operation or evaluation a tf.Tensor
                #By passing one or more Tensors or Operations, TensorFlow will execute the operations needed
                run_optim = sess.run(optimizer, feed_dict = {x: batch_x, y: batch_y, dropout_rate_placeholder : params['dropout_rate']})

                loss, acc = sess.run([cost, accuracy], feed_dict = {x: batch_x, y: batch_y, dropout_rate_placeholder : 0.0})
                
            training_summ, training_acc = sess.run([merged, accuracy], feed_dict={x: batch_x, y: batch_y, dropout_rate_placeholder : 0.0})
            training_writer.add_summary(training_summ, epoch)

            testing_summ, testing_acc = sess.run([merged, accuracy], feed_dict={x: testing_data, y: testing_labels, dropout_rate_placeholder : 0.0})
            testing_writer.add_summary(testing_summ, epoch)

            print("At iteration " + str(epoch) + ", Loss = " + \
                 "{:.4f}".format(loss) + ", Training Accuracy = " + \
                                "{:.4f}".format(training_acc) + ", Testing Accuracy = " + \
                                "{:.4f}".format(testing_acc))

        print("Training complete")
        
        save_path = saver.save(sess, "/Binding_CNN.ckpt")
        print("Model saved in Binding_CNN.ckpt")
        
        test_acc, _ = sess.run([accuracy,cost], feed_dict={x: testing_data, y: testing_labels, dropout_rate_placeholder : 0.0})

        print("Final testing Accuracy:","{:.5f}".format(test_acc))

        training_writer.close()
        testing_writer.close()


BindingNet_train(params, var_list_Binding, training_data, training_labels, testing_data, testing_labels)




