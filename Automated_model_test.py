#!/usr/bin/env python3
import numpy as np
import os
import tensorflow as tf
import mltest
import CNN_module as CNN

#Test to ensure all initialized variables in the graph are modified during learning

#Temporarily disable deprecation warnings (using tf 1.14)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

model_params = {
    'architecture':'BindingCNN',
    'dynamic_var':'None',
    'dataset':'mnist',
    'learning_rate':0.001,
    'meta_architecture':'CNN',
    'train_new_network':False,
    'adver_trained':False,
    'crossval_bool':False,
    'check_stochasticity':False,
    'num_network_duplicates':1,
    'training_epochs':30,
    'dropout_rate':0.25,
    'label_smoothing':0.1,
    'L1_regularization_activations1':0.0001,
    'L1_regularization_activations2':0.0001,
    'batch_size':128
    }

# Build your test function.
def test_mltest_suite(model_params):

    (training_data, training_labels, testing_data, testing_labels, crossval_data, crossval_labels) = CNN.data_setup(model_params)

    x_placeholder, y_placeholder, dropout_rate_placeholder, var_list, weights, biases = CNN.initializer_fun(model_params, training_data, training_labels)

    predictions, _, l1_reg_activations1, l1_reg_activations2 = getattr(CNN, model_params['architecture'] + '_predictions')(x_placeholder, 
        dropout_rate_placeholder, weights, biases, model_params['dynamic_var'])

    cost = (tf.reduce_mean(tf.compat.v1.losses.softmax_cross_entropy(logits=predictions, 
        onehot_labels=y_placeholder, label_smoothing=params['label_smoothing'])) + 
        params['L1_regularization_activations1']*l1_reg_activations1 + 
        params['L1_regularization_activations2']*l1_reg_activations2)

    train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=model_params['learning_rate']).minimize(cost)

    feed_dict = {
      x_placeholder: testing_data[0:model_params['batch_size']],
      y_placeholder: testing_labels[0:model_params['batch_size']],
      dropout_rate_placeholder: 0.0
    }
    # Run the test suite!
    mltest.test_suite(
        predictions,
        train_op,
        feed_dict=feed_dict, 
        var_list=var_list,
        test_output_range=False,
        test_nan_vals=True,
        test_inf_vals=True)

test_mltest_suite(model_params)
