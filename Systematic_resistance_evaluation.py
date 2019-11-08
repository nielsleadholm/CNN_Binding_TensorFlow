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
	'architecture':'VGG',
	'dynamic_var':'None',
	'dataset':'cifar10',
    'learning_rate':0.001,
	'meta_architecture':'CNN',
	'train_new_network':True,
	'crossval_bool':False,
	'num_network_duplicates':5,
    'training_epochs':100,
    'dropout_rate_min':0.25,
    'dropout_rate_max':0.25,
    'dropout_parameter_step_size':0.1,
    'L1_regularization_scale_unpool':0.0,
    'L1_regularization_scale_gradient_unpool':0.0,
    'L1_regularization_scale_min':0.0001,
    'L1_regularization_scale_max':0.001,
    'L_regularization_parameter_step_size':0.0001,
    'batch_size':128
	}

#Parameters determining the adversarial attacks
#perturbation_threshold defines the threshold of the L-0, L-inf, and L-2 distances at which accuracy is evaluated
adversarial_params = {
	'num_attack_examples':50,
    'boundary_attack_iterations':1000,
    'boundary_attack_log_steps':1000,
    'distance_range':None,
    'transfer_attack_model_for_gen':'BindingCNN',
    'transfer_attack_BaseAttack_for_gen':'Boundary',
    'perturbation_threshold':[12, 0.3, 1.5]
    }


#Specify training and cross-validation data
(training_data, training_labels, testing_data, testing_labels, crossval_data, crossval_labels) = CNN.data_setup(model_params)

def iterative_evaluation(model_params, adversarial_params, training_data, training_labels, crossval_data, crossval_labels):
	#iter_num helps keep track of which network names are associated with which result columns; it indicates multiple networks with the same parameters, but new initializations

	all_results_dic = {} #stores results associated with the network names as keys (which in turn reference important parameters)

	for iter_num in range(model_params['num_network_duplicates']):
		#Iterate through drop_out_rate parameter values

		#Iterate through L1 regularization parameter values
		# print("Iterating over a series of L1 hyper-parameter values")
		# for L1_iter in np.arange(model_params['L1_regularization_scale_min'], model_params['L1_regularization_scale_max']+2*model_params['L_regularization_parameter_step_size'], 
		# 		model_params['L_regularization_parameter_step_size']):
					L1_unpool_iter = model_params['L1_regularization_scale_unpool']
					L1_gradient_unpool_iter = model_params['L1_regularization_scale_gradient_unpool']


		# 		for dropout_iter in np.arange(model_params['dropout_rate_min'], model_params['dropout_rate_max']+model_params['dropout_parameter_step_size'], 
		# 				model_params['dropout_parameter_step_size']):
					dropout_iter = model_params['dropout_rate_min']

					iteration_params = {'architecture':model_params['architecture'],
						'dynamic_var':model_params['dynamic_var'],
						'meta_architecture':model_params['meta_architecture'],
						'dataset':model_params['dataset'],
					    'training_epochs':model_params['training_epochs'],
					    'dropout_rate':dropout_iter,
					    'L1_regularization_scale_unpool':L1_unpool_iter,
					    'L1_regularization_scale_gradient_unpool':L1_gradient_unpool_iter,
					    'learning_rate':model_params['learning_rate'],
					    'batch_size':model_params['batch_size']}

					x_placeholder, y_placeholder, dropout_rate_placeholder, var_list, weights, biases = CNN.initializer_fun(iteration_params, training_data, training_labels)


					#Option to run the training and adversarial testing with a cross-validation data-set, rather than the true test dataset
					if model_params['crossval_bool'] == True:
						evaluation_data = crossval_data
						evaluation_labels = crossval_labels
					else:
						evaluation_data = testing_data
						evaluation_labels = testing_labels

					if model_params['train_new_network'] == True:

						training_accuracy, crossval_accuracy, network_name_str = CNN.network_train(iteration_params, iter_num, var_list, training_data, 
							training_labels, evaluation_data, evaluation_labels, weights, biases, x_placeholder=x_placeholder, 
						    y_placeholder=y_placeholder, dropout_rate_placeholder=dropout_rate_placeholder)

						all_results_dic.update({network_name_str:{}})
						#Store results in dictionary and array for later visualizaiton
						all_results_dic[network_name_str].update({'training_accuracy':float(training_accuracy), 'testing_accuracy':float(crossval_accuracy)})
					
					else:

						network_name_str = str(iter_num) + (model_params['architecture'] + '_L1-' + str(L1_iter) + '_drop-' + str(dropout_iter))
						

						# print("\n\n *** Using adversarially trained network *** \n\n")
						# network_name_str = 'adver_trained'


						all_results_dic.update({network_name_str:{}})

						#Evaluate the accuracy of the loaded model
						predictions, _, _ = getattr(CNN, iteration_params['architecture'] + '_predictions')(x_placeholder, dropout_rate_placeholder, weights, biases, iteration_params['dynamic_var'])
						correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(y_placeholder, 1))
						accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

						saver = tf.train.Saver(var_list)
						with tf.Session() as sess:
							saver.restore(sess, ("network_weights_data/" + network_name_str + ".ckpt"))
							acc = sess.run(accuracy, feed_dict={x_placeholder: testing_data, y_placeholder: testing_labels, dropout_rate_placeholder : 0.0})
							print("Testing Accuracy of loaded model = " + "{:.4f}".format(acc))
							all_results_dic[network_name_str].update({'testing_accuracy':float(acc)})
					
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
					# stoch_check = atk.check_stochasticity(model_prediction_function=,
			  #               model_weights=("network_weights_data/" + network_name_str + ".ckpt"),
			  #               var_list=var_list,
			  #               weights_dic=weights,
			  #               biases_dic=biases,
			  #               input_data=input_data,
			  #               input_labels=input_labels,
			  #               input_placeholder=x_placeholder,
			  #               dropout_rate_placeholder=0,
			  #               output_directory = network_name_str,
			  #               num_attack_examples=adversarial_params['num_attack_examples'])
					# stoch_check.perform_check()

					# print("\n\n *** Using adversarially trained network *** \n\n")
					# network_name_str = 'adver_trained'

					all_results_dic[network_name_str].update(carry_out_attacks(adversarial_params, input_data, input_labels, x_placeholder, var_list, weights, biases, network_name_str, iter_num, iteration_params['dynamic_var']))
					print("\n\nThe cumulative results are...\n")
					print(all_results_dic)

					with open('Results_dic.json', 'w') as f:
						json.dump(all_results_dic, f, indent=4)



def carry_out_attacks(adversarial_params, input_data, input_labels, x_placeholder, var_list, weights, biases, network_name_str, iter_num, dynamic_var):

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
                'num_attack_examples':adversarial_params['num_attack_examples'],
                'dynamic_var':dynamic_var}

	network_dic = {} #Hold the attack specific results for a given network

	L0_running_min = [] #Keep track of the minimum adversarial perturbation required for each image
	LInf_running_min = [] #Keep track of the minimum adversarial perturbation required for each image
	L2_running_min = [] #Keep track of the minimum adversarial perturbation required for each image

	# print("\n\n***Performing L-0 Distance Attacks***\n")

	# pointwise_L0 = atk.pointwise_attack_L0(attack_dic)
	# adversary_found, adversary_distance, adversaries_array, perturb_list = pointwise_L0.evaluate_resistance()
	# network_dic['pointwise_L0'] = [float(s) for s in analysis(adversary_distance, perturb_list, 
	# 	perturbation_threshold=adversarial_params['perturbation_threshold'][0])]
	# L0_running_min = adversary_distance
	
	# salt = atk.salt_pepper_attack(attack_dic)
	# adversary_found, adversary_distance, adversaries_array, perturb_list = salt.evaluate_resistance()
	# network_dic['salt'] = [float(s) for s in analysis(adversary_distance, perturb_list, 
	# 	perturbation_threshold=adversarial_params['perturbation_threshold'][0])]
	# L0_running_min = np.minimum(L0_running_min, adversary_distance) #Take element-wise minimum

	# print("\nResults across all L0 attacks...")
	# network_dic['all_L0'] = [float(s) for s in analysis(L0_running_min, perturb_list=[], perturbation_threshold=adversarial_params['perturbation_threshold'][0])]

	print("\n\n***Performing L-Inf Distance Attacks***\n")

	FGSM = atk.FGSM_attack(attack_dic)
	adversary_found, adversary_distance, adversaries_array, perturb_list = FGSM.evaluate_resistance()
	network_dic['FGSM'] = [float(s) for s in analysis(adversary_distance, perturb_list, 
		perturbation_threshold=adversarial_params['perturbation_threshold'][1])]
	LInf_running_min = adversary_distance
	
	BIM_LInf = atk.BIM_Linfinity_attack(attack_dic)
	adversary_found, adversary_distance, adversaries_array, perturb_list = BIM_LInf.evaluate_resistance()
	network_dic['BIM_LInf'] = [float(s) for s in analysis(adversary_distance, perturb_list, 
		perturbation_threshold=adversarial_params['perturbation_threshold'][1])]
	LInf_running_min = np.minimum(LInf_running_min, adversary_distance)
	
	DeepFool_LInf = atk.DeepFool_LInf_attack(attack_dic)
	adversary_found, adversary_distance, adversaries_array, perturb_list = DeepFool_LInf.evaluate_resistance()
	network_dic['DeepFool_LInf'] = [float(s) for s in analysis(adversary_distance, perturb_list, 
		perturbation_threshold=adversarial_params['perturbation_threshold'][1])]
	LInf_running_min = np.minimum(LInf_running_min, adversary_distance)

	MIM = atk.MIM_attack(attack_dic)
	adversary_found, adversary_distance, adversaries_array, perturb_list = MIM.evaluate_resistance()
	network_dic['MIM'] = [float(s) for s in analysis(adversary_distance, perturb_list, 
		perturbation_threshold=adversarial_params['perturbation_threshold'][1])]
	LInf_running_min = np.minimum(LInf_running_min, adversary_distance)

	print("\nResults across all LInf attacks...")
	network_dic['all_LInf'] = [float(s) for s in analysis(LInf_running_min, perturb_list=[], perturbation_threshold=adversarial_params['perturbation_threshold'][1])]


	print("\n\n***Performing L-2 Distance Attacks***\n")

	gaussian = atk.gaussian_noise_attack(attack_dic)
	adversary_found, adversary_distance, adversaries_array, perturb_list = gaussian.evaluate_resistance()
	network_dic['gaussian'] = [float(s) for s in analysis(adversary_distance, perturb_list, 
		perturbation_threshold=adversarial_params['perturbation_threshold'][2])]
	L2_running_min = adversary_distance

	pointwise_L2 = atk.pointwise_attack_L2(attack_dic)
	adversary_found, adversary_distance, adversaries_array, perturb_list = pointwise_L2.evaluate_resistance()
	network_dic['pointwise_L2'] = [float(s) for s in analysis(adversary_distance, perturb_list, 
		perturbation_threshold=adversarial_params['perturbation_threshold'][2])]
	L2_running_min = np.minimum(L2_running_min, adversary_distance)
	
	FGM = atk.FGM_attack(attack_dic)
	adversary_found, adversary_distance, adversaries_array, perturb_list = FGM.evaluate_resistance()
	network_dic['FGM'] = [float(s) for s in analysis(adversary_distance, perturb_list, 
		perturbation_threshold=adversarial_params['perturbation_threshold'][2])]
	L2_running_min = np.minimum(L2_running_min, adversary_distance)

	BIM_L2 = atk.BIM_L2_attack(attack_dic)
	adversary_found, adversary_distance, adversaries_array, perturb_list = BIM_L2.evaluate_resistance()
	network_dic['BIM_L2'] = [float(s) for s in analysis(adversary_distance, perturb_list, 
		perturbation_threshold=adversarial_params['perturbation_threshold'][2])]
	L2_running_min = np.minimum(L2_running_min, adversary_distance)

	DeepFool_L2 = atk.DeepFool_L2_attack(attack_dic)
	adversary_found, adversary_distance, adversaries_array, perturb_list = DeepFool_L2.evaluate_resistance()
	network_dic['DeepFool_L2'] = [float(s) for s in analysis(adversary_distance, perturb_list, 
		perturbation_threshold=adversarial_params['perturbation_threshold'][2])]
	L2_running_min = np.minimum(L2_running_min, adversary_distance)
	
	boundary = atk.boundary_attack(attack_dic,
                num_iterations=adversarial_params['boundary_attack_iterations'],
                log_every_n_steps=adversarial_params['boundary_attack_log_steps'])
	adversary_found, adversary_distance, adversaries_array, perturb_list = boundary.evaluate_resistance()
	network_dic['boundary'] = [float(s) for s in analysis(adversary_distance, perturb_list, 
		perturbation_threshold=adversarial_params['perturbation_threshold'][2])]
	L2_running_min = np.minimum(L2_running_min, adversary_distance)
	
	print("\nResults across all L2 attacks...")
	network_dic['all_L2'] = [float(s) for s in analysis(L2_running_min, perturb_list=[], perturbation_threshold=adversarial_params['perturbation_threshold'][2])]

	print("\n\n***Performing Additional Attacks***\n")

	blended = atk.blended_noise_attack(attack_dic)
	adversary_found, adversary_distance, adversaries_array, perturb_list = blended.evaluate_resistance()
	network_dic['blended'] = [float(s) for s in analysis(adversary_distance, perturb_list, 
		perturbation_threshold=adversarial_params['perturbation_threshold'][2])]

	# local_search = atk.local_search_attack(attack_dic)
	# adversary_found, adversary_distance, adversaries_array, perturb_list = local_search.evaluate_resistance()
	# network_dic['local_search'] = [float(s) for s in analysis(adversary_distance, perturb_list, 
	# 	perturbation_threshold=adversarial_params['perturbation_threshold'][2])]

	# gaussian_blur = atk.gaussian_blur_attack(attack_dic)
	# adversary_found, adversary_distance, adversaries_array, perturb_list = gaussian_blur.evaluate_resistance()
	# network_dic['gaussian_blur'] = [float(s) for s in analysis(adversary_distance, perturb_list, 
	# 	perturbation_threshold=adversarial_params['perturbation_threshold'][2])]

	# spatial = atk.spatial_attack(attack_dic)
	# adversary_found, adversary_distance, adversaries_array, perturb_list = spatial.evaluate_resistance()
	# network_dic['spatial'] = [float(s) for s in analysis(adversary_distance, perturb_list, 
	# 	perturbation_threshold=adversarial_params['perturbation_threshold'][2])]

	# #Note model_adversarial_gen determines the model that was used 
	# transfer = atk.transfer_attack(attack_dic,
 #                model_under_attack=model_params['architecture'],
 #                model_adversarial_gen=str(iter_num) + adversarial_params['transfer_attack_model_for_gen'],
 #                attack_type_dir=adversarial_params['transfer_attack_BaseAttack_for_gen'],
 #                distance_metric = adversarial_params['distance_metric'])

	# adversary_found, adversary_distance = transfer.evaluate_resistance()
	
	# #Keep a cumulative score of the succcess ratio
	# success_ratio += np.sum(adversary_found)/adversarial_params['num_attack_examples']




	return network_dic



#Find the proportion of examples above a given threshold
def threshold_accuracy(perturbation_threshold, adversary_distance):
	#Boolean mask where adversary distances is above the threshold
	threshold_exceeded = (adversary_distance > perturbation_threshold)
	threshold_acc = np.sum(threshold_exceeded)/len(adversary_distance)
	return threshold_acc

def analysis(adversary_distance, perturb_list, perturbation_threshold):

	median_distance = np.median(adversary_distance)
	threshold_acc = threshold_accuracy(perturbation_threshold, adversary_distance)
	print("The median distance is " + str(median_distance))
	print("The classiciation accuracy at a threshold of " + str(perturbation_threshold) + " is " + str(threshold_acc))

	if len(perturb_list) > 0:
		mean_perturb = np.sum(perturb_list)/len(perturb_list)
		print("The mean perturbation is " + str(mean_perturb))

	return median_distance, threshold_acc


iterative_evaluation(model_params, adversarial_params, training_data, training_labels, crossval_data, crossval_labels)


