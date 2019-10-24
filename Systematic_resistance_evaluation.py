#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import os
import json
import tfCore_adversarial_attacks as atk
import matplotlib.pyplot as plt
from PIL import Image
import CNN_module as CNN

#Parameters determining the networks that will be trained and evaluated
model_params = {
	'architecture':'LeNet',
	'control_modification':None,
	'meta_architecture':'CNN',
	'dataset':'mnist',
	'train_new_network':True,
	'crossval_bool':False,
	'num_network_duplicates':1,
    'training_epochs':30,
    'predictive_weighting':0.01,
    'dropout_rate_min':0.5,
    'dropout_rate_max':0.5,
    'dropout_parameter_step_size':0.1,
    'L1_regularization_scale_min':0.0,
    'L1_regularization_scale_max':0.0,
    'L2_regularization_scale_min':0.0,
    'L2_regularization_scale_max':0.0,
    'L_regularization_parameter_step_size':0.0025,
    'learning_rate':0.001,
    'batch_size':128
	}
#predictive weighting parameter determines the balance between how much the supervised auto-encoder's
# cost is dependent on prediction of class vs reconstruction (auto-encoder)

#Parameters determining the adversarial attacks
adversarial_params = {
	'num_attack_examples':5,
    'boundary_attack_iterations':1000,
    'boundary_attack_log_steps':1000,
    'distance_range':None,
    'transfer_attack_model_for_gen':'BindingCNN',
    'transfer_attack_BaseAttack_for_gen':'Boundary'
    }


#Specify training and cross-validation data
(training_data, training_labels, testing_data, testing_labels, crossval_data, crossval_labels) = CNN.data_setup(model_params)

def iterative_evaluation(model_params, adversarial_params, training_data, training_labels, crossval_data, crossval_labels):
	#iter_num helps keep track of which network names are associated with which result columns; it indicates multiple networks with the same parameters, but new initializations

	results_dic = {} #stores results associated with the network names as keys (which in turn reference important parameters)
	results_matrix = [] #stores results for easy plotting later

	for iter_num in range(model_params['num_network_duplicates']):
		#Iterate through drop_out_rate parameter values

		# #Iterate through L1 regularization parameter values
		# for L1_iter in np.arange(model_params['L1_regularization_scale_min'], model_params['L1_regularization_scale_max']+2*model_params['L_regularization_parameter_step_size'], 
		# 		model_params['L_regularization_parameter_step_size']):

	 #       	#Iterate through L2 regularization parameter values
		# 	for L2_iter in np.arange(model_params['L2_regularization_scale_min'], model_params['L2_regularization_scale_max']+2*model_params['L_regularization_parameter_step_size'], 
		# 			model_params['L_regularization_parameter_step_size']):

		# 		for dropout_iter in np.arange(model_params['dropout_rate_min'], model_params['dropout_rate_max']+model_params['dropout_parameter_step_size'], 
		# 				model_params['dropout_parameter_step_size']):

					dropout_iter = model_params['dropout_rate_min']
					L1_iter = model_params['L1_regularization_scale_min']
					L2_iter = model_params['L2_regularization_scale_min']
					iteration_params = {'architecture':model_params['architecture'],
						'control_modification':model_params['control_modification'],
						'meta_architecture':model_params['meta_architecture'],
						'dataset':model_params['dataset'],
					    'training_epochs':model_params['training_epochs'],
					    'predictive_weighting':model_params['predictive_weighting'],
					    'dropout_rate':dropout_iter,
					    'L1_regularization_scale':L1_iter,
					    'L2_regularization_scale':L2_iter,
					    'learning_rate':model_params['learning_rate'],
					    'batch_size':model_params['batch_size']}

					results_list = []

					x_placeholder, y_placeholder, dropout_rate_placeholder, var_list, weights, biases, decoder_weights = CNN.initializer_fun(iteration_params, training_data, training_labels)

					# #Visualise image reconstruction of the auto-encoder before and after training
					# if iteration_params['meta_architecture'] == 'auto_encoder' or iteration_params['meta_architecture'] == 'SAE':
					# 	input_example = np.expand_dims(testing_data[0], axis=0)
					# 	print(np.shape(input_example))
					# 	predictions = CNN.AutoEncoder(input_example, dropout_rate_placeholder, weights, biases, decoder_weights)
					# 	decoded = tf.nn.sigmoid(predictions)
					# 	print(np.shape(testing_data[0, :, :, 0]))
					# 	print(np.shape(decoded[0, :, :, 0]))
					# 	# im = Image.fromarray(testing_data[0, :, :, 0])
					# 	# if im.mode != 'RGB':
					# 	# 	im = im.convert('RGB')
					# 	# im.save('test.png')
					# 	print(testing_data[0, :, :, 0])
					# 	print(decoded[0, :, :, 0].eval())
					# 	plt.imsave('original.png', testing_data[0, :, :, 0], cmap='gray')
					# 	plt.imsave('before.png', decoded[0, :, :, 0], cmap='gray')

					#Option to run the training and adversarial testing with a cross-validation data-set, rather than the true test dataset
					if model_params['crossval_bool'] == True:
						evaluation_data = crossval_data
						evaluation_labels = crossval_labels
					else:
						evaluation_data = testing_data
						evaluation_labels = testing_labels

					if model_params['train_new_network'] == True:

						training_accuracy, crossval_accuracy, network_name_str = CNN.network_train(iteration_params, iter_num, var_list, training_data, 
							training_labels, evaluation_data, evaluation_labels, weights, biases, decoder_weights, x_placeholder=x_placeholder, 
						    y_placeholder=y_placeholder, dropout_rate_placeholder=dropout_rate_placeholder)

						#Store results in dictionary and array for later visualizaiton
						results_list.append(training_accuracy), results_list.append(crossval_accuracy)
					
					else:
						network_name_str = str(iter_num) + (model_params['architecture'] + '_L1-' + str(L1_iter) + '_L2-' + str(L2_iter) + '_drop-' + str(dropout_iter))
					
					#If testing, then every adversarial attack will use a different sample of the testing data
					if model_params['crossval_bool'] == 1:
						input_data = crossval_data
						input_labels = crossval_labels
					else:
						lower_bound = adversarial_params['num_attack_examples']*iter_num
						upper_bound = adversarial_params['num_attack_examples']*iter_num+adversarial_params['num_attack_examples']
						input_data = testing_data[lower_bound:upper_bound]
						input_labels = testing_labels[lower_bound:upper_bound]


					# # Optional check for stochasticity
					# stoch_check = atk.check_stochasticity(model_prediction_function=CNN.BindingCNN_control1_predictions,
			  #               model_weights=("network_weights_data/" + network_name_str + ".ckpt"),
			  #               var_list=var_list,
			  #               weights_dic=weights,
			  #               biases_dic=biases,
			  #               input_data=input_data,
			  #               input_labels=input_labels,
			  #               input_placeholder=x_placeholder,
			  #               dropout_rate_placeholder=0,
			  #               output_directory = network_name_str,
			  #               meta_architecture=model_params['meta_architecture'],
			  #               num_attack_examples=adversarial_params['num_attack_examples'])
					# stoch_check.perform_check()


					#network_name_str = 'adver_trained'

					results_list = carry_out_attacks(adversarial_params, input_data, input_labels, x_placeholder, var_list, weights, biases, network_name_str, results_list, iter_num)
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



def carry_out_attacks(adversarial_params, input_data, input_labels, x_placeholder, var_list, weights, biases, network_name_str, results_list, iter_num):

	#Create directory for storing images
	if os.path.exists('adversarial_images/') == 0:
		os.mkdir('adversarial_images/')

	attack_dic = {'model_prediction_function':getattr(CNN, model_params['architecture'] + '_predictions'),
                'model_weights':("network_weights_data/" + network_name_str + ".ckpt"),
                'var_list':var_list,
                'weights_dic':weights,
                'biases_dic':biases,
                'input_data':input_data,
                'input_labels':input_labels,
                'input_placeholder':x_placeholder,
                'dropout_rate_placeholder':0.0,
                'output_directory':network_name_str,
                'meta_architecture':model_params['meta_architecture'],
                'num_attack_examples':adversarial_params['num_attack_examples']}



	# print("\n\n***Performing L-0 Distance Attacks***\n\n")

	# pointwise_L0 = atk.pointwise_attack_L0(attack_dic)
	# adversary_found, adversary_distance, adversaries_array, perturb_list = pointwise_L0.evaluate_resistance()
	# results_list = analysis(results_list, adversary_found, adversary_distance, adversaries_array, perturb_list)

	# salt = atk.salt_pepper_attack(attack_dic)
	# adversary_found, adversary_distance, adversaries_array, perturb_list = salt.evaluate_resistance()
	# results_list = analysis(results_list, adversary_found, adversary_distance, adversaries_array, perturb_list)



	# print("\n\n***Performing L-Inf Distance Attacks***\n\n")

	# FGSM = atk.FGSM_attack(attack_dic)
	# adversary_found, adversary_distance, adversaries_array, perturb_list = FGSM.evaluate_resistance()
	# results_list = analysis(results_list, adversary_found, adversary_distance, adversaries_array, perturb_list)

	# BIM_LInf = atk.BIM_Linfinity_attack(attack_dic)
	# adversary_found, adversary_distance, adversaries_array, perturb_list = BIM_LInf.evaluate_resistance()
	# results_list = analysis(results_list, adversary_found, adversary_distance, adversaries_array, perturb_list)

	# DeepFool_LInf = atk.DeepFool_LInf_attack(attack_dic)
	# adversary_found, adversary_distance, adversaries_array, perturb_list = DeepFool_LInf.evaluate_resistance()
	# results_list = analysis(results_list, adversary_found, adversary_distance, adversaries_array, perturb_list)

	# MIM = atk.MIM_attack(attack_dic)
	# adversary_found, adversary_distance, adversaries_array, perturb_list = MIM.evaluate_resistance()
	# results_list = analysis(results_list, adversary_found, adversary_distance, adversaries_array, perturb_list)


	# print("\n\n***Performing L-2 Distance Attacks***\n\n")

	# # blended = atk.blended_noise_attack(attack_dic)
	# # adversary_found, adversary_distance, adversaries_array, perturb_list = blended.evaluate_resistance()
	# # results_list = analysis(results_list, adversary_found, adversary_distance, adversaries_array, perturb_list)

	# gaussian = atk.gaussian_noise_attack(attack_dic)
	# adversary_found, adversary_distance, adversaries_array, perturb_list = gaussian.evaluate_resistance()
	# results_list = analysis(results_list, adversary_found, adversary_distance, adversaries_array, perturb_list)

	# pointwise_L2 = atk.pointwise_attack_L2(attack_dic)
	# adversary_found, adversary_distance, adversaries_array, perturb_list = pointwise_L2.evaluate_resistance()
	# results_list = analysis(results_list, adversary_found, adversary_distance, adversaries_array, perturb_list)

	# FGM = atk.FGM_attack(attack_dic)
	# adversary_found, adversary_distance, adversaries_array, perturb_list = FGM.evaluate_resistance()
	# results_list = analysis(results_list, adversary_found, adversary_distance, adversaries_array, perturb_list)

	# BIM_L2 = atk.BIM_L2_attack(attack_dic)
	# adversary_found, adversary_distance, adversaries_array, perturb_list = BIM_L2.evaluate_resistance()
	# results_list = analysis(results_list, adversary_found, adversary_distance, adversaries_array, perturb_list)

	# DeepFool_L2 = atk.DeepFool_L2_attack(attack_dic)
	# adversary_found, adversary_distance, adversaries_array, perturb_list = DeepFool_L2.evaluate_resistance()
	# results_list = analysis(results_list, adversary_found, adversary_distance, adversaries_array, perturb_list)

	# boundary = atk.boundary_attack(attack_dic,
 #                num_iterations=adversarial_params['boundary_attack_iterations'],
 #                log_every_n_steps=adversarial_params['boundary_attack_log_steps'])
	# adversary_found, adversary_distance, adversaries_array, perturb_list = boundary.evaluate_resistance()
	# results_list = analysis(results_list, adversary_found, adversary_distance, adversaries_array, perturb_list)


	print("\n\n***Performing Additional Attacks***\n\n")

	blended = atk.blended_noise_attack(attack_dic)
	adversary_found, adversary_distance, adversaries_array, perturb_list = blended.evaluate_resistance()
	results_list = analysis(results_list, adversary_found, adversary_distance, adversaries_array, perturb_list)

	local_search = atk.local_search_attack(attack_dic)
	adversary_found, adversary_distance, adversaries_array, perturb_list = local_search.evaluate_resistance()
	results_list = analysis(results_list, adversary_found, adversary_distance, adversaries_array, perturb_list)

	gaussian_blur = atk.gaussian_blur_attack(attack_dic)
	adversary_found, adversary_distance, adversaries_array, perturb_list = gaussian_blur.evaluate_resistance()
	results_list = analysis(results_list, adversary_found, adversary_distance, adversaries_array, perturb_list)

	spatial = atk.spatial_attack(attack_dic)
	adversary_found, adversary_distance, adversaries_array, perturb_list = spatial.evaluate_resistance()
	results_list = analysis(results_list, adversary_found, adversary_distance, adversaries_array, perturb_list)



# # *** Temporary implementation: iterate through a series of distance-specified transfer attacks ***

# 	success_ratio = 0
# 	for jj in range(15):
# 		adversarial_params['distance_range'] = (jj, jj+1)

# 		transfer = atk.transfer_attack(attack_dic,
# 	                model_under_attack=model_params['architecture'],
# 	                model_adversarial_gen=str(iter_num) + adversarial_params['transfer_attack_model_for_gen'],
# 	                attack_type_dir=adversarial_params['transfer_attack_BaseAttack_for_gen'],
# 	                distance_range = adversarial_params['distance_range'])

# 		adversary_found = transfer.evaluate_resistance()
		
# 		#Keep a cumulative score of the succcess ratio
# 		success_ratio += np.sum(adversary_found)/adversarial_params['num_attack_examples']

		
# 		print("The success ratio is " + str(success_ratio))
# 		results_list.append(success_ratio)



	return results_list

def analysis(results_list, adversary_found, adversary_distance, adversaries_array, perturb_list):

	success_ratio = np.sum(adversary_found)/adversarial_params['num_attack_examples']
	mean_distance = np.sum(adversary_distance)/adversarial_params['num_attack_examples']
	median_distance = np.median(adversary_distance)
	std_distance = np.std(adversary_distance)
	print("The success ratio is " + str(success_ratio))
	print("The mean distance is " + str(mean_distance))
	print("The median distance is " + str(median_distance))

	if len(perturb_list) > 0:
		mean_perturb = np.sum(perturb_list)/len(perturb_list)
		print("The mean perturbation is " + str(mean_perturb))


	results_list.append(success_ratio), results_list.append(median_distance), results_list.append(mean_distance), results_list.append(std_distance)

	return results_list


iterative_evaluation(model_params, adversarial_params, training_data, training_labels, crossval_data, crossval_labels)


