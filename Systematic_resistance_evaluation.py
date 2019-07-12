#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import os
import json
import tfCore_adversarial_attacks as atk
import matplotlib.pyplot as plt
import LeNet5_CNN as LeNet

#Parameters determining the networks that will be trained and evaluated
model_params = {'model_choice':'CNN_Binding.py',
	'training_epochs':30,
    'dropout_rate_min':0.0,
    'dropout_rate_max':0.5,
    'dropout_parameter_step_size':0.1,
    'L1_regularization_scale_min':0.0,
    'L1_regularization_scale_max':0.01,
    'L2_regularization_scale_min':0.0,
    'L2_regularization_scale_max':0.01,
    'L_regularization_parameter_step_size':0.0025,
    'learning_rate':0.001,
    'batch_size':128}

#Parameters determining the adversariala attacks
adversarial_params = {'num_attack_examples':50,
	'boundary_attack_iterations':1000,
	'boundary_attack_log_steps':1000,
	'BIM_attack_epsilon':0.3}


#Specify training and cross-validation data
(training_data, training_labels, _, _, crossval_data, crossval_labels) = LeNet.data_setup(crossval_boolean=1)

def iterative_evaluation(model_params, adversarial_params, training_data, training_labels, crossval_data, crossval_labels):
	#Iterate through drop_out_rate parameter values
	iter_num = 0 #tracks the index number of the current network
	results_dic = {} #stores results associated with the network names as keys (which in turn reference important parameters)
	results_matrix = [] #stores results for easy plotting later

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

				results_list = []

				x_placeholder, y_placeholder, dropout_rate_placeholder, var_list_LeNet, weights_LeNet, biases_LeNet = LeNet.initilaizer_fun(iteration_params, training_data, training_labels)

				training_accuracy, crossval_accuracy, network_name_str = LeNet.LeNet5_train(iteration_params, var_list_LeNet, training_data, 
					training_labels, crossval_data, crossval_labels, weights_LeNet, biases_LeNet, x_placeholder=x_placeholder, y_placeholder=y_placeholder, dropout_rate_placeholder=dropout_rate_placeholder)

				#Store results in dictionary and array for later visualizaiton
				results_list.append(training_accuracy), results_list.append(crossval_accuracy)
				results_list = carry_out_attacks(adversarial_params, crossval_data, crossval_labels, x_placeholder, var_list_LeNet, weights_LeNet, biases_LeNet, network_name_str, results_list)
				results_dic[str(iter_num) + network_name_str] = [float(s) for s in results_list] #convert numpy float into Python float for later json dumping
				results_matrix.append(results_list)
				iter_num += 1 #indexed from 0, iter_num is used to keep track of which network names are associated with which result columns


	results_matrix = np.asarray(results_matrix)

	#Save results to file
	np.savetxt("Results_matrix.csv", results_matrix, delimiter=",")
	with open('Results_dic.json', 'w') as f:
		json.dump(results_dic, f, indent=4)



def carry_out_attacks(adversarial_params, crossval_data, crossval_labels, x_placeholder, var_list_LeNet, weights_LeNet, biases_LeNet, network_name_str, results_list):

	#Create directory for storing images
	if os.path.exists('adversarial_images/') == 0:
		os.mkdir('adversarial_images/')

	pointwise = atk.pointwise_attack(model_prediction_function=LeNet.cnn_predictions,
                model_weights=("network_weights_data/" + network_name_str + ".ckpt"),
                var_list=var_list_LeNet,
                weights_dic=weights_LeNet,
                biases_dic=biases_LeNet,
                input_data=crossval_data,
                input_labels=crossval_labels,
                input_placeholder=x_placeholder,
                dropout_rate_placeholder=0.0,
                output_directory = network_name_str,
                num_attack_examples=adversarial_params['num_attack_examples'])

	adversary_found, adversary_distance, adversary_arrays = pointwise.evaluate_resistance()
	results_list = analysis(results_list, adversary_found, adversary_distance, adversary_arrays)

	blended = atk.blended_noise_attack(model_prediction_function=LeNet.cnn_predictions,
                model_weights=("network_weights_data/" + network_name_str + ".ckpt"),
                var_list=var_list_LeNet,
                weights_dic=weights_LeNet,
                biases_dic=biases_LeNet,
                input_data=crossval_data,
                input_labels=crossval_labels,
                input_placeholder=x_placeholder,
                dropout_rate_placeholder=0.0,
                output_directory = network_name_str,
                num_attack_examples=adversarial_params['num_attack_examples'])

	adversary_found, adversary_distance, adversary_arrays = blended.evaluate_resistance()
	results_list = analysis(results_list, adversary_found, adversary_distance, adversary_arrays)

	boundary = atk.boundary_attack(model_prediction_function=LeNet.cnn_predictions,
                model_weights=("network_weights_data/" + network_name_str + ".ckpt"),
                var_list=var_list_LeNet,
                weights_dic=weights_LeNet,
                biases_dic=biases_LeNet,
                input_data=crossval_data,
                input_labels=crossval_labels,
                input_placeholder=x_placeholder,
                dropout_rate_placeholder=0.0,
                output_directory = network_name_str,
                num_attack_examples=adversarial_params['num_attack_examples'],
                num_iterations=adversarial_params['boundary_attack_iterations'],
                log_every_n_steps=adversarial_params['boundary_attack_log_steps'])

	adversary_found, adversary_distance, adversary_arrays = boundary.evaluate_resistance()
	results_list = analysis(results_list, adversary_found, adversary_distance, adversary_arrays)


	BIM2 = atk.BIM_L2_attack(model_prediction_function=LeNet.cnn_predictions,
                model_weights=("network_weights_data/" + network_name_str + ".ckpt"),
                var_list=var_list_LeNet,
                weights_dic=weights_LeNet,
                biases_dic=biases_LeNet,
                input_data=crossval_data,
                input_labels=crossval_labels,
                input_placeholder=x_placeholder,
                dropout_rate_placeholder=0.0,
                output_directory = network_name_str,
                num_attack_examples=adversarial_params['num_attack_examples'],
                epsilon=adversarial_params['BIM_attack_epsilon'])

	adversary_found, adversary_distance, adversary_arrays = BIM2.evaluate_resistance()
	results_list = analysis(results_list, adversary_found, adversary_distance, adversary_arrays)


	BIMInf = atk.BIM_Linfinity_attack(model_prediction_function=LeNet.cnn_predictions,
                model_weights=("network_weights_data/" + network_name_str + ".ckpt"),
                var_list=var_list_LeNet,
                weights_dic=weights_LeNet,
                biases_dic=biases_LeNet,
                input_data=crossval_data,
                input_labels=crossval_labels,
                input_placeholder=x_placeholder,
                dropout_rate_placeholder=0.0,
                output_directory = network_name_str,
                num_attack_examples=adversarial_params['num_attack_examples'],
                epsilon=adversarial_params['BIM_attack_epsilon'])

	adversary_found, adversary_distance, adversary_arrays = BIMInf.evaluate_resistance()
	results_list = analysis(results_list, adversary_found, adversary_distance, adversary_arrays)

	return results_list

def analysis(results_list, adversary_found, adversary_distance, adversary_arrays):

	success_ratio = np.sum(adversary_found)/adversarial_params['num_attack_examples']
	mean_distance = np.sum(adversary_distance)/np.sum(adversary_found)
	std_distance = np.std(adversary_distance)

	print("The success ratio is " + str(success_ratio))
	print("The mean distance is " + str(mean_distance))

	results_list.append(success_ratio), results_list.append(mean_distance), results_list.append(std_distance)

	return results_list


iterative_evaluation(model_params, adversarial_params, training_data, training_labels, crossval_data, crossval_labels)

#Create numpy array to store key data values, this is then added to the Pandas data structure
#for each model; the rows, indexed from 0, correspond to 0:training accuracy, 1:crossvalidation 
#accuracy, 2/3/4: percentage of successful adversaries, mean distance, and standard deviaition of distance 
#for the pointw-wise Noise attack; 5/6/7, 8/9/10, 11/12/13, and 14/15/16 then correspond to these 
#three values for the blended-uniform, boundary, BIM2, and BIM-infinity attacks respectively
#Columns should be named based on the model identifier

# plus an additional text file with network parameters (e.g. number of epochs of training, number of attack examples), all organized in a hierarchical folder system

# After many different models have been iterated through, provide summary data and visualization on how the models performed in comparison
# e.g. a bar chart for each adversary type, and where each bar is a different model; should be possible to 
# iterate through each Pandas data file, extract required data, and create comparison plots

# Rather than trying to show every model on the bar chart, the chart can be used to get an idea for the spread (how much a particular parameter matters for performance), and then print e.g. 'the top 3 performing models'
# Include in the rows information on the model parameter iters, so that I can use these to develop a mask, and then for any given fixed parameter value, see what influence the others have




