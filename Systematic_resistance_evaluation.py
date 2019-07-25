#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import os
import json
import tfCore_adversarial_attacks as atk
import matplotlib.pyplot as plt
import CNN_module as CNN

#Parameters determining the networks that will be trained and evaluated
model_params = {'architecture':'BindingCNN',
	'train_new_network':1,
	'cross_val_bool':0,
	'num_network_duplicates': 3,
    'training_epochs':5,
    'dropout_rate_min':0.25,
    'dropout_rate_max':0.25,
    'dropout_parameter_step_size':0.1,
    'L1_regularization_scale_min':0.0,
    'L1_regularization_scale_max':0.0,
    'L2_regularization_scale_min':0.0,
    'L2_regularization_scale_max':0.0,
    'L_regularization_parameter_step_size':0.00125,
    'learning_rate':0.001,
    'batch_size':128}

#Parameters determining the adversariala attacks
adversarial_params = {'num_attack_examples':5,
    'boundary_attack_iterations':50,
    'boundary_attack_log_steps':1000,
    'BIM_attack_epsilon':0.3}


#Specify training and cross-validation data
(training_data, training_labels, testing_data, testing_labels, crossval_data, crossval_labels) = CNN.data_setup(model_params['cross_val_bool'])

def iterative_evaluation(model_params, adversarial_params, training_data, training_labels, crossval_data, crossval_labels):
	#iter_num helps keep track of which network names are associated with which result columns; it indicates multiple networks with the same parameters, but new initializations

	results_dic = {} #stores results associated with the network names as keys (which in turn reference important parameters)
	results_matrix = [] #stores results for easy plotting later

	for iter_num in range(model_params['num_network_duplicates']):
		#Iterate through drop_out_rate parameter values

		# for dropout_iter in np.arange(model_params['dropout_rate_min'], model_params['dropout_rate_max']+model_params['dropout_parameter_step_size'], 
		# 		model_params['dropout_parameter_step_size']):

		# 	#Iterate through L1 regularization parameter values
		# 	for L1_iter in np.arange(model_params['L1_regularization_scale_min'], model_params['L1_regularization_scale_max']+2*model_params['L_regularization_parameter_step_size'], 
		# 			model_params['L_regularization_parameter_step_size']):

		#        	#Iterate through L2 regularization parameter values
		# 		for L2_iter in np.arange(model_params['L2_regularization_scale_min'], model_params['L2_regularization_scale_max']+2*model_params['L_regularization_parameter_step_size'], 
		# 				model_params['L_regularization_parameter_step_size']):
		dropout_iter = model_params['dropout_rate_min']
		L1_iter = model_params['L1_regularization_scale_min']
		L2_iter = model_params['L2_regularization_scale_min']
		iteration_params = {'architecture':model_params['architecture'],
		    'training_epochs':model_params['training_epochs'],
		    'dropout_rate':dropout_iter,
		    'L1_regularization_scale':L1_iter,
		    'L2_regularization_scale':L2_iter,
		    'learning_rate':model_params['learning_rate'],
		    'batch_size':model_params['batch_size']}

		results_list = []

		x_placeholder, y_placeholder, dropout_rate_placeholder, var_list, weights, biases = CNN.initializer_fun(iteration_params, training_data, training_labels)

		#Option to run the training and adversarial testing with a cross-validation data-set, rather than the true test dataset
		if model_params['cross_val_bool'] == 1:
			evaluation_data = crossval_data
			evaluation_labels = crossval_labels
		else:
			evaluation_data = testing_data
			evaluation_labels = testing_labels

		if model_params['train_new_network'] ==1:

			training_accuracy, crossval_accuracy, network_name_str = CNN.network_train(iteration_params, iter_num, var_list, training_data, 
				training_labels, evaluation_data, evaluation_labels, weights, biases, x_placeholder=x_placeholder, 
			    y_placeholder=y_placeholder, dropout_rate_placeholder=dropout_rate_placeholder)

			#Store results in dictionary and array for later visualizaiton
			results_list.append(training_accuracy), results_list.append(crossval_accuracy)
		
		else:
			network_name_str = str(iter_num) + (model_params['architecture'] + '_L1-' + str(L1_iter) + '_L2-' + str(L2_iter) + '_drop-' + str(dropout_iter))
		
		#If testing, then every adversarial attack will use a different sample of the testing data
		if model_params['cross_val_bool'] == 1:
			input_data = crossval_data
			input_labels = crossval_labels
		else:
			lower_bound = adversarial_params['num_attack_examples']*iter_num
			upper_bound = adversarial_params['num_attack_examples']*iter_num+adversarial_params['num_attack_examples']
			print(lower_bound)
			print(upper_bound)
			input_data = testing_data[lower_bound:upper_bound]
			input_labels = testing_labels[lower_bound:upper_bound]
			print(input_data.shape)
			print(input_labels.shape)


		results_list = carry_out_attacks(adversarial_params, input_data, input_labels, x_placeholder, var_list, weights, biases, network_name_str, results_list)
		print(results_list)
		results_dic[network_name_str] = [float(s) for s in results_list] #convert numpy float into Python float for later json dumping
		print(results_dic)
		results_matrix.append(results_list)
		print(results_matrix)

		#Save the results matrix and dictionary on every iteration in case something goes wrong
		save_results_matrix = np.asarray(results_matrix)

		#Save results to file
		np.savetxt("Results_matrix.csv", save_results_matrix, delimiter=",")
		with open('Results_dic.json', 'w') as f:
			json.dump(results_dic, f, indent=4)



def carry_out_attacks(adversarial_params, input_data, input_labels, x_placeholder, var_list, weights, biases, network_name_str, results_list):

	#Create directory for storing images
	if os.path.exists('adversarial_images/') == 0:
		os.mkdir('adversarial_images/')

	# pointwise = atk.pointwise_attack(model_prediction_function=getattr(CNN, model_params['architecture'] + '_predictions'),
 #                model_weights=("network_weights_data/" + network_name_str + ".ckpt"),
 #                var_list=var_list,
 #                weights_dic=weights,
 #                biases_dic=biases,
 #                input_data=input_data,
 #                input_labels=input_labels,
 #                input_placeholder=x_placeholder,
 #                dropout_rate_placeholder=0.0,
 #                output_directory = network_name_str,
 #                num_attack_examples=adversarial_params['num_attack_examples'])

	# adversary_found, adversary_distance, adversary_arrays, _ = pointwise.evaluate_resistance()
	# results_list = analysis(results_list, adversary_found, adversary_distance, adversary_arrays)

	# blended = atk.blended_noise_attack(model_prediction_function=getattr(CNN, model_params['architecture'] + '_predictions'),
 #                model_weights=("network_weights_data/" + network_name_str + ".ckpt"),
 #                var_list=var_list,
 #                weights_dic=weights,
 #                biases_dic=biases,
 #                input_data=input_data,
 #                input_labels=input_labels,
 #                input_placeholder=x_placeholder,
 #                dropout_rate_placeholder=0.0,
 #                output_directory = network_name_str,
 #                num_attack_examples=adversarial_params['num_attack_examples'])

	# adversary_found, adversary_distance, adversary_arrays, _ = blended.evaluate_resistance()
	# results_list = analysis(results_list, adversary_found, adversary_distance, adversary_arrays)

	boundary = atk.boundary_attack(model_prediction_function=getattr(CNN, model_params['architecture'] + '_predictions'),
                model_weights=("network_weights_data/" + network_name_str + ".ckpt"),
                var_list=var_list,
                weights_dic=weights,
                biases_dic=biases,
                input_data=input_data,
                input_labels=input_labels,
                input_placeholder=x_placeholder,
                dropout_rate_placeholder=0.0,
                output_directory = network_name_str,
                num_attack_examples=adversarial_params['num_attack_examples'],
                num_iterations=adversarial_params['boundary_attack_iterations'],
                log_every_n_steps=adversarial_params['boundary_attack_log_steps'])

	adversary_found, adversary_distance, adversary_arrays, perturb_list = boundary.evaluate_resistance()
	results_list = analysis(results_list, adversary_found, adversary_distance, adversary_arrays, perturb_list)


	# BIM2 = atk.BIM_L2_attack(model_prediction_function=getattr(CNN, model_params['architecture'] + '_predictions'),
 #                model_weights=("network_weights_data/" + network_name_str + ".ckpt"),
 #                var_list=var_list,
 #                weights_dic=weights,
 #                biases_dic=biases,
 #                input_data=input_data,
 #                input_labels=input_labels,
 #                input_placeholder=x_placeholder,
 #                dropout_rate_placeholder=0.0,
 #                output_directory = network_name_str,
 #                num_attack_examples=adversarial_params['num_attack_examples'],
 #                epsilon=adversarial_params['BIM_attack_epsilon'])

	# adversary_found, adversary_distance, adversary_arrays, _ = BIM2.evaluate_resistance()
	# results_list = analysis(results_list, adversary_found, adversary_distance, adversary_arrays)


	# BIMInf = atk.BIM_Linfinity_attack(model_prediction_function=getattr(CNN, model_params['architecture'] + '_predictions'),
 #                model_weights=("network_weights_data/" + network_name_str + ".ckpt"),
 #                var_list=var_list,
 #                weights_dic=weights,
 #                biases_dic=biases,
 #                input_data=input_data,
 #                input_labels=input_labels,
 #                input_placeholder=x_placeholder,
 #                dropout_rate_placeholder=0.0,
 #                output_directory = network_name_str,
 #                num_attack_examples=adversarial_params['num_attack_examples'],
 #                epsilon=adversarial_params['BIM_attack_epsilon'])

	# adversary_found, adversary_distance, adversary_arrays, _ = BIMInf.evaluate_resistance()
	# results_list = analysis(results_list, adversary_found, adversary_distance, adversary_arrays)

	return results_list

def analysis(results_list, adversary_found, adversary_distance, adversary_arrays, perturb_list):

	success_ratio = np.sum(adversary_found)/adversarial_params['num_attack_examples']
	mean_distance = np.sum(adversary_distance)/np.sum(adversary_found)
	std_distance = np.std(adversary_distance)
	print("The success ratio is " + str(success_ratio))
	print("The mean distance is " + str(mean_distance))

	if len(perturb_list) > 0:
		mean_perturb = np.sum(perturb_list)/len(perturb_list)
		print("The mean perturbation is " + str(mean_perturb))


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




