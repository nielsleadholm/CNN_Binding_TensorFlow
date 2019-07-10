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

#The following is an implementation of a simple CNN based on the architecture of LeNet-5 for the MNIST dataset

def data_setup(crossval_boolean):
    #Load the full MNIST dataset
    #Note the shape of the images required by the custom CNNs is 2D, rather than flattened as for the Madry model
    (training_data, training_labels), (testing_data, testing_labels) = mnist.load_data()


    #Rescale images to values between 0:1 and reshape so each image is 28x28
    training_data = training_data/255
    training_data = np.reshape(training_data, [np.shape(training_data)[0], 28, 28, 1])

    testing_data = testing_data/255
    testing_data = np.reshape(testing_data, [np.shape(testing_data)[0], 28, 28, 1])

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

def initilaizer_fun(params, training_data, training_labels):

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

    #Note for example that the first convolutional weights layer has a 5x5 filter with 1 input channel, and 6 output channels
    #tf.get_variable will either get an existing variable with these parameters, or otherwise create a new one
    with tf.name_scope('Weights'):
        weights_LeNet = {
        'conv_W1' : tf.get_variable('CW1', shape=(5, 5, 1, 6), initializer=initializer, regularizer=regularizer_l1),
        'conv_W2' : tf.get_variable('CW2', shape=(5, 5, 6, 16), initializer=initializer, regularizer=regularizer_l1),
        'dense_W1' : tf.get_variable('DW1', shape=(400, 120), initializer=initializer, regularizer=regularizer_l2),
        'dense_W2' : tf.get_variable('DW2', shape=(120, 84), initializer=initializer, regularizer=regularizer_l2),
        'output_W' : tf.get_variable('OW', shape=(84, 10), initializer=initializer, regularizer=regularizer_l2)
        }

        #Add summaries for each weight variable in the dictionary, for later use in TensorBoard
        for weights_var in weights_LeNet.values():
            var_summaries(weights_var)

    with tf.name_scope('Biases'):
        biases_LeNet = {
        'conv_b1' : tf.get_variable('Cb1', shape=(6), initializer=initializer, regularizer=regularizer_l1),
        'conv_b2' : tf.get_variable('Cb2', shape=(16), initializer=initializer, regularizer=regularizer_l1),
        'dense_b1' : tf.get_variable('Db1', shape=(120), initializer=initializer, regularizer=regularizer_l2),
        'dense_b2' : tf.get_variable('Db2', shape=(84), initializer=initializer, regularizer=regularizer_l2),
        'output_b' : tf.get_variable('Ob', shape=(10), initializer=initializer, regularizer=regularizer_l2)
        }

        for biases_var in biases_LeNet.values():
            var_summaries(biases_var)

    var_list_LeNet = [weights_LeNet['conv_W1'], weights_LeNet['conv_W2'], weights_LeNet['dense_W1'], weights_LeNet['dense_W2'], 
    weights_LeNet['output_W'], biases_LeNet['conv_b1'], biases_LeNet['conv_b2'], biases_LeNet['dense_b1'], 
    biases_LeNet['dense_b2'], biases_LeNet['output_b']]

    return x, y, dropout_rate_placeholder, var_list_LeNet, weights_LeNet, biases_LeNet


#Define the model
def cnn_predictions(features, dropout_rate_placeholder, weights_LeNet, biases_LeNet):
    conv1 = tf.nn.conv2d(input=tf.dtypes.cast(features, dtype=tf.float32), filter=weights_LeNet['conv_W1'],
                         strides=[1, 1, 1, 1], padding="SAME")
    conv1 = tf.nn.bias_add(conv1, biases_LeNet['conv_b1'])
    conv1_drop = tf.nn.dropout(conv1, rate=dropout_rate_placeholder)
    relu1 = tf.nn.relu(conv1_drop)
    tf.summary.histogram('Relu1_activations', relu1)
    pool1 = tf.nn.max_pool(relu1, ksize=(1,2,2,1), strides=(1,2,2,1), padding="VALID")
    pool1_drop = tf.nn.dropout(pool1, rate=dropout_rate_placeholder)
    #Note in the tuple defining strides for max_pool, the first entry is always 1 as this refers to the batches/indexed images,
    #rather than the dimensions of a particular image

    conv2 = tf.nn.conv2d(pool1_drop, weights_LeNet['conv_W2'], strides=[1,1,1,1], padding="VALID")
    conv2 = tf.nn.bias_add(conv2, biases_LeNet['conv_b2'])
    conv2_drop = tf.nn.dropout(conv2, rate=dropout_rate_placeholder)
    relu2 = tf.nn.relu(conv2_drop)
    relu2_sparsity = tf.math.zero_fraction(relu2)
    tf.summary.scalar('Relu2_sparsness', relu2_sparsity)
    tf.summary.histogram('Relu2_activations', relu2)
    pool2 = tf.nn.max_pool(relu2, ksize=(1,2,2,1), strides=(1,2,2,1), padding="VALID")
    pool2_drop = tf.nn.dropout(pool2, rate=dropout_rate_placeholder)

    #Flatten Pool 2 before connecting it (fully) with the dense layers 1 and 2
    pool2_flat = tf.reshape(pool2, [-1, 5 * 5 * 16])
    dense1 = tf.add(tf.matmul(pool2_flat, weights_LeNet['dense_W1']), biases_LeNet['dense_b1'])
    dense1_drop = tf.nn.dropout(dense1, rate=dropout_rate_placeholder)
    dense1_drop = tf.nn.relu(dense1_drop)
    tf.summary.histogram('Dense1_activations', dense1_drop)
    dense2 = tf.add(tf.matmul(dense1_drop, weights_LeNet['dense_W2']), biases_LeNet['dense_b2'])
    dense2_drop = tf.nn.dropout(dense2, rate=dropout_rate_placeholder)
    dense2_drop = tf.nn.relu(dense2_drop)
    tf.summary.histogram('Dense2_activations', dense2_drop)

    logits = tf.add(tf.matmul(dense2_drop, weights_LeNet['output_W']), biases_LeNet['output_b'])

    return logits

#Primary training function
def LeNet5_train(params, var_list, training_data, training_labels, testing_data, testing_labels, weights_LeNet, biases_LeNet, x_placeholder, y_placeholder, dropout_rate_placeholder):

    predictions = cnn_predictions(x_placeholder, dropout_rate_placeholder, weights_LeNet, biases_LeNet) #NB that x was defined earlier with tf.placeholder

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
        network_name_str = ('LeNet_L1-' + str(params['L1_regularization_scale']) + '_L2-' + str(params['L2_regularization_scale']) + '_drop-' + str(params['dropout_rate']))
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

    params = {'training_epochs':1,
    'dropout_rate':0.5,
    'L1_regularization_scale':0.0,
    'L2_regularization_scale':0.0,
    'learning_rate':0.001,
    'batch_size':128} #NB that drop-out 'rate' = 1 - 'keep probability'

    (training_data, training_labels, testing_data, testing_labels, _, _) = data_setup(crossval_boolean=0)

    x, y, dropout_rate_placeholder, var_list_LeNet, weights_LeNet, biases_LeNet = initilaizer_fun(params, training_data, training_labels)

    LeNet5_train(params, var_list_LeNet, training_data, training_labels, testing_data, testing_labels, weights_LeNet, biases_LeNet, x_placeholder=x, y_placeholder=y, dropout_rate_placeholder=dropout_rate_placeholder)

#test_image_iter = 52
# test_attack = atk.BIM_L2_attack(model_prediction_function=cnn_predictions,
#                         model_weights="/MNIST_LeNet5_CNN.ckpt",
#                         var_list=var_list_LeNet,
#                         input_data=testing_data[test_image_iter, :, :],
#                         input_label=np.argmax(testing_labels[test_image_iter,:]),
#                         input_placeholder=x)

# test_attack.execute_attack()


# test_evaluation = atk.parent_attack(model_prediction_function=cnn_predictions,
#                         model_weights="/MNIST_LeNet5_CNN.ckpt",
#                         var_list=var_list_LeNet,
#                         input_data=testing_data,
#                         input_label=testing_labels,
#                         input_placeholder=x)

# test_evaluation.evaluate_resistance()
