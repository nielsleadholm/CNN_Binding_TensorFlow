#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import os
#import pandas as pd
import tfCore_adversarial_attacks as atk
import LeNet5_CNN as LeNet

#Parameters determining the networks that will be trained and evaluated
model_params = {'model_choice':'CNN_Binding.py',
	'training_epochs':5,
    'dropout_rate_min':0.0,
    'dropout_rate_max':0.5,
    'dropout_parameter_step_size':0.5,
    'L1_regularization_scale_min':0.01,
    'L1_regularization_scale_max':0.01,
    'L2_regularization_scale_min':0.01,
    'L2_regularization_scale_max':0.01,
    'L_regularization_parameter_step_size':0.01,
    'learning_rate':0.001,
    'batch_size':128}

#Parameters determining the adversariala attacks
adversarial_params = {'num_attack_examples':5,
	'boundary_attack_iterations':50,
	'boundary_attack_log_steps':50,
	'BIM_attack_epsilon':0.3}

#Specify training and cross-validation data
(training_data, training_labels, _, _, crossval_data, crossval_labels) = LeNet.data_setup(crossval_boolean=1)

#Iterate through drop_out_rate parameter values
for dropout_iter in np.arange(model_params['dropout_rate_min'], model_params['dropout_rate_max']+model_params['dropout_parameter_step_size'], 
		model_params['dropout_parameter_step_size']):

	#Iterate through L1 regularization parameter values
	for L1_iter in np.arange(model_params['L1_regularization_scale_min'], model_params['L1_regularization_scale_max']+model_params['L_regularization_parameter_step_size'], 
			model_params['L_regularization_parameter_step_size']):

		#Iterate through L2 regularization parameter values
		for L2_iter in np.arange(model_params['L2_regularization_scale_min'], model_params['L2_regularization_scale_max']+model_params['L_regularization_parameter_step_size'], 
				model_params['L_regularization_parameter_step_size']):

			iteration_params = {'training_epochs':model_params['training_epochs'],
			    'dropout_rate':dropout_iter,
			    'L1_regularization_scale':L1_iter,
			    'L2_regularization_scale':L2_iter,
			    'learning_rate':model_params['learning_rate'],
			    'batch_size':model_params['batch_size']}


			x, y, dropout_rate_placeholder, var_list_LeNet, weights_LeNet, biases_LeNet = LeNet.initilaizer_fun(iteration_params, training_data, training_labels)

			training_accuracy, crosval_accuracy, network_name_str = LeNet.LeNet5_train(iteration_params, var_list_LeNet, training_data, 
				training_labels, crossval_data, crossval_labels, weights_LeNet, biases_LeNet, x_placeholder=x, y_placeholder=y, dropout_rate_placeholder=dropout_rate_placeholder)

			#Create directory for storing images
			if os.path.exists('adversarial_images/') == 0:
				os.mkdir('adversarial_images/')

			blended_attack = atk.blended_noise_attack(model_prediction_function=LeNet.cnn_predictions,
                        model_weights=("network_weights_data/" + network_name_str + ".ckpt"),
                        var_list=var_list_LeNet,
                        weights_dic=weights_LeNet,
                        biases_dic=biases_LeNet,
                        input_data=crossval_data,
                        input_labels=crossval_labels,
                        input_placeholder=x,
                        dropout_rate_placeholder=0.0,
                        output_directory = network_name_str,
                        num_attack_examples=adversarial_params['num_attack_examples'])

			adversary_found, adversary_distance, adversary_arrays = blended_attack.evaluate_resistance()

			success_ratio = np.sum(adversary_found)/adversarial_params['num_attack_examples']
			mean_distance_successful_attacks = np.sum(adversary_distance)/np.sum(adversary_found)

			print("The success ratio is " + str(success_ratio))
			print("The mean distance is " + str(mean_distance_successful_attacks))

			# #Iterate through the main attack methods
			# BIM_L2 = atk.BIM_L2_attack(model_prediction_function=LeNet.cnn_predictions,
			#                         model_weights="/MNIST_LeNet5_CNN.ckpt",
			#                         var_list=var_list_LeNet,
			#                         input_data=crossval_data,
			#                         input_label=crossval_labels,
			#                         input_placeholder=x,
			#                         num_attack_examples=adversarial_params['num_attack_examples'],
			#                         epsilon=adversarial_params['BIM_attack_epsilon'])

			# adversary_found, adversary_distance, _ = BIM_L2.evaluate_resistance()


			# BIM_Linfinity = atk.BIM_Linfinity_attack(model_prediction_function=cnn_predictions,
			#                         model_weights="/MNIST_LeNet5_CNN.ckpt",
			#                         var_list=var_list_LeNet,
			#                         input_data=testing_data[test_image_iter, :, :],
			#                         input_label=np.argmax(testing_labels[test_image_iter,:]),
			#                         input_placeholder=x,
			#                         num_attack_examples=adversarial_params['num_attack_examples'],
			#                         epsilon=adversarial_params['BIM_attack_epsilon'])

			# adversary_found, adversary_distance, _ = BIM_Linfinity.evaluate_resistance()




#Create numpy array to store key data values, this is then added to the Pandas data structure
#for each model; the rows, indexed from 0, correspond to 0:training accuracy, 1:crossvalidation 
#accuracy, 2/3/4: percentage of successful adversaries, mean distance, and standard deviaition of distance 
#for the Blended Uniform Noise attack; 5/6/7, 8/9/10, 11/12/13, and 14/15/16 then correspond to these 
#three values for the pointwise, boundary, BIM2, and BIM-infinity attacks respectively
#data_row

# Data to save: 
# Percentage of successful adversaries in each of the attack categories (5)
# Mean distance, and standard deviation of each test category (10 : 2 for each of 5 types of attack)
# Meta data: key network parameters, distance metric for each distance, the adversarial attack type
# ?Pandas data structure the best way of capturing most of this (i.e. can have 'columns' with labelling based on meta-data)
# plus an additional text file with network parameters (e.g. number of epochs of training, number of attack examples), all organized in a hierarchical folder system

#As it is the columns of Pandas dataframes that have text labels, it might make most sense to organise the columns
#based on the network names (as this is otherwise the hardest to keep track of); the 12 rows are then established, 
#e.g. that row 6 refers to say the percentage of successful attacks with the BIM2 method

# After many different models have been iterated through, provide summary data and visualization on how the models performed in comparison
# e.g. a bar chart for each adversary type, and where each bar is a different model; should be possible to 
# iterate through each Pandas data file, extract required data, and create comparison plots




