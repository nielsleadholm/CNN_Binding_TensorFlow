#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import os
import json
import tfCore_adversarial_attacks as atk
import matplotlib.pyplot as plt
import mltest
import math
import time
from PIL import Image
import CNN_module as CNN
import Adversarial_training

#Parameters determining the networks that will be trained and evaluated
#dynamic_dic allows for temporary changes to a model's forward pass, such as adding stochasticity or ablating layers; see CNN_module for details
#Options for dynamic_dic are:
	# "Add_logit_noise"
	# "Ablate_unpooling"
	# "Ablate_gradient_unpooling"
	# "Ablate_binding"
	# "Ablate_maxpooling"
	# "Ablate_max&gradient"
	# "Ablate_max&unpool"
#sparsification_kwinner determines the sparsity of gradient unpooling activations (select a value between 0 and 1); note the actual
# binding activation sparsity may differ (specifically be more sparse) depending on how sparse the activations are before the gradient unpooling operation
#sparsification_dropout applies a dropout rate to binding layers at all times (i.e. including during testing) to measure the effect of activation sparsity
#meta_architecture determines whether a standard CNN or e.g. an auto-encoder architecture is to be used
#adver_trained, if True, will train or load an adversarially trained model
#L1 regularization has two separate constants:
	# For a LeNet model, activations1 refers to the penultimate fully-connected layer (as in Guo et al), and activations2 has no effect
	# For a control model, activations1 refers to unpooling, activations2 to gradient unpooling
	# For any other models, these activations have no regularizing effect
model_params = {
	'architecture':'LeNet',
	'dynamic_dic':{'dynamic_var':'None', 'sparsification_kwinner':0.15, 'sparsification_dropout':0.0},
	'dataset':'mnist',
    'learning_rate':0.001,
	'meta_architecture':'CNN',
	'train_new_network':True,
	'adver_trained':False,
	'crossval_bool':True,
	'check_stochasticity':False,
	'num_network_duplicates':5,
    'training_epochs':45,
    'dropout_rate':0.25,
    'label_smoothing':0.1,
    'L1_regularization_activations1':0.0001,
    'L1_regularization_activations2':0.0,
    'batch_size':128
	}

#Parameters determining the adversarial attacks
#perturbation_threshold defines the threshold of the L-0, L-inf, and L-2 distances at which accuracy is evaluated
adversarial_params = {
	'num_attack_examples':128,
    'boundary_attack_iterations':1000,
    'boundary_attack_log_steps':1000,
    'perturbation_threshold':[12, 0.3, 1.5],
    'save_images':False
    }

#Specify training and cross-validation data
(training_data, training_labels, testing_data, testing_labels, crossval_data, crossval_labels) = CNN.data_setup(model_params)

#Option to run the training and adversarial testing with a cross-validation data-set, rather than the true test dataset
if model_params['crossval_bool'] == True:
	evaluation_data = crossval_data
	evaluation_labels = crossval_labels
else:
	evaluation_data = testing_data
	evaluation_labels = testing_labels

def iterative_evaluation(model_params, adversarial_params, training_data, training_labels, evaluation_data, evaluation_labels):

	all_results_dic = {}

	for iter_num in range(model_params['num_network_duplicates']):

		if (model_params['train_new_network'] == True) and (model_params['adver_trained'] == True):

			print("\nUsing adversarial training...")
			eval_accuracy, network_name_str = Adversarial_training.adver_training(model_params, iter_num, training_data, 
				training_labels, evaluation_data, evaluation_labels)
			
			all_results_dic.update({network_name_str:{}})			
			all_results_dic[network_name_str].update({'testing_accuracy':float(eval_accuracy)})
		
		#Note Adversarial_training uses it's own initializer
		x_placeholder, y_placeholder, dropout_rate_placeholder, var_list, weights, biases = CNN.initializer_fun(model_params, training_data, training_labels)

		if (model_params['train_new_network'] == True) and (model_params['adver_trained'] == False):
			print("\nUsing standard (non-adversarial) training...")
			training_accuracy, eval_accuracy, network_name_str, eval_sparsity = CNN.network_train(model_params, iter_num, var_list, training_data, 
				training_labels, evaluation_data, evaluation_labels, weights, biases, x_placeholder=x_placeholder, 
			    y_placeholder=y_placeholder, dropout_rate_placeholder=dropout_rate_placeholder)

			all_results_dic.update({network_name_str:{}})
			testing_sparsity_float = [dict([key, float(value)] for key, value in eval_sparsity.items())] 
			all_results_dic[network_name_str].update({'training_accuracy':float(training_accuracy), 'testing_accuracy':float(eval_accuracy),
				'testing_sparsity':testing_sparsity_float})
		
		elif (model_params['train_new_network'] == False) or (model_params['adver_trained'] == True):

			network_name_str = str(iter_num) + model_params['architecture'] + '_adver_trained_' + str(model_params['adver_trained'])
			all_results_dic.update({network_name_str:{}})

			#Evaluate the accuracy of the loaded model
			predictions, sparsity_dic, activation_dic, _ = getattr(CNN, model_params['architecture'] + '_predictions')(x_placeholder, dropout_rate_placeholder, weights, biases, model_params['dynamic_dic'])
			correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(y_placeholder, 1))
			total_accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.float32)) #Used to store batched accuracy for evaluating the test dataset

			#Used for the mltest suite:
			dummy_cost = tf.reduce_mean(tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(logits=predictions, labels=y_placeholder))
			dummy_op = tf.compat.v1.train.AdamOptimizer(learning_rate=model_params['learning_rate']).minimize(dummy_cost)

			saver = tf.train.Saver(var_list)
			with tf.Session() as sess:
				saver.restore(sess, ("network_weights_data/" + network_name_str + ".ckpt"))

				# print("\nRunning mltest for NaN/Inf values on a single batch...")
				# mltest.test_suite(
				# 	predictions,
				# 	train_op=dummy_op,
				# 	feed_dict={x_placeholder: testing_data[0:model_params['batch_size']], 
				# 		y_placeholder: testing_labels[0:model_params['batch_size']], dropout_rate_placeholder : 0.0}, 
				# 	var_list=var_list,
				# 	test_all_inputs_dependent=False,
				# 	test_other_vars_dont_change=False,
				# 	test_output_range=False,
				# 	test_nan_vals=True,
				# 	test_inf_vals=True)
				# print("--> passed mltest check.")

				#Assess accuracy batch-wise to avoid memory issues in large models
				accuracy_total = 0 
				for test_batch in range(math.ceil(len(testing_labels)/model_params['batch_size'])):

					test_batch_x = testing_data[test_batch*model_params['batch_size']:min((test_batch+1)*model_params['batch_size'], len(testing_labels))]
					test_batch_y = testing_labels[test_batch*model_params['batch_size']:min((test_batch+1)*model_params['batch_size'], len(testing_labels))]

					batch_testing_acc = sess.run(total_accuracy, feed_dict={x_placeholder: test_batch_x, y_placeholder: test_batch_y, dropout_rate_placeholder : 0.0})
					accuracy_total += batch_testing_acc

				testing_acc = accuracy_total/len(testing_labels)

				print("Testing Accuracy of loaded model = " + "{:.4f}".format(testing_acc))
				all_results_dic[network_name_str].update({'testing_accuracy':float(testing_acc)})

				#On small-memory data-sets, check layer-wise sparsity on all testing data
				if model_params['dataset']!='cifar10':
					testing_sparsity = sess.run(sparsity_dic, feed_dict={x_placeholder: testing_data, y_placeholder: testing_labels, dropout_rate_placeholder : 0.0})

					print("\nLayer-wise sparsity:")
					print(testing_sparsity)

					#Convert values in the dictonary to standard float for .json saving
					testing_sparsity_float = [dict([key, float(value)] for key, value in testing_sparsity.items())] 

					all_results_dic[network_name_str].update({'testing_sparsity':testing_sparsity_float})


		# Optional check for stochasticity in a model's predictions
		if model_params['check_stochasticity'] == True:
			stoch_check = atk.check_stochasticity(model_prediction_function=getattr(CNN, model_params['architecture'] + '_predictions'),
	                model_weights=("network_weights_data/" + network_name_str + ".ckpt"),
	                var_list=var_list,
	                weights_dic=weights,
	                biases_dic=biases,
	                input_data=evaluation_data,
	                input_labels=evaluation_labels,
	                input_placeholder=x_placeholder,
	                dropout_rate_placeholder=0,
	                output_directory = network_name_str,
	                num_attack_examples=adversarial_params['num_attack_examples'])
			stoch_check.perform_check()

		all_results_dic[network_name_str].update(carry_out_attacks(adversarial_params, evaluation_data, evaluation_labels, 
			x_placeholder, var_list, weights, biases, network_name_str, iter_num, model_params['dynamic_dic']))
		print("\n\nThe cumulative results are...\n")
		print(all_results_dic)

		with open('Results_dic.json', 'w') as f:
			json.dump(all_results_dic, f, indent=4)



def carry_out_attacks(adversarial_params, input_data, input_labels, x_placeholder, var_list, weights, biases, network_name_str, iter_num, dynamic_dic):

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
                'dynamic_dic':dynamic_dic,
                'batch_size':model_params['batch_size'],
                'save_images':adversarial_params['save_images']}

	network_dic = {} #Hold the attack specific results for a given network

	L0_running_min = [] #Keep track of the minimum adversarial perturbation required for each image
	LInf_running_min = [] #Keep track of the minimum adversarial perturbation required for each image
	L2_running_min = [] #Keep track of the minimum adversarial perturbation required for each image

	# print("\n\n***Performing L-0 Distance Attacks***\n")

	# attack_name = 'pointwise_L0'
	# pointwise_L0 = atk.pointwise_attack_L0(attack_dic)
	# adversary_found, adversary_distance, adversaries_array, perturb_list = pointwise_L0.evaluate_resistance()
	# network_dic[attack_name] = [float(s) for s in analysis(adversary_distance, perturb_list, 
	# 	perturbation_threshold=adversarial_params['perturbation_threshold'][0])]
	# L0_running_min = adversary_distance
	# np.savetxt('adversarial_images/' + network_name_str + '/' + attack_name + '_distances.txt', adversary_distance)

	# attack_name = 'salt'
	# salt = atk.salt_pepper_attack(attack_dic)
	# adversary_found, adversary_distance, adversaries_array, perturb_list = salt.evaluate_resistance()
	# network_dic[attack_name] = [float(s) for s in analysis(adversary_distance, perturb_list, 
	# 	perturbation_threshold=adversarial_params['perturbation_threshold'][0])]
	# L0_running_min = np.minimum(L0_running_min, adversary_distance) #Take element-wise minimum
	# np.savetxt('adversarial_images/' + network_name_str + '/' + attack_name + '_distances.txt', adversary_distance)

	# print("\nResults across all L0 attacks...")
	# network_dic['all_L0'] = [float(s) for s in analysis(L0_running_min, perturb_list=[], perturbation_threshold=adversarial_params['perturbation_threshold'][0])]

	print("\n\n***Performing L-Inf Distance Attacks***\n")

	attack_name = 'FGSM'
	FGSM = atk.FGSM_attack(attack_dic)
	adversary_found, adversary_distance, adversaries_array, perturb_list = FGSM.evaluate_resistance()
	network_dic[attack_name] = [float(s) for s in analysis(adversary_distance, perturb_list, 
		perturbation_threshold=adversarial_params['perturbation_threshold'][1])]
	LInf_running_min = adversary_distance
	np.savetxt('adversarial_images/' + network_name_str + '/' + attack_name + '_distances.txt', adversary_distance)
	
	attack_name = 'BIM_LInf'
	BIM_LInf = atk.BIM_Linfinity_attack(attack_dic)
	adversary_found, adversary_distance, adversaries_array, perturb_list = BIM_LInf.evaluate_resistance()
	network_dic[attack_name] = [float(s) for s in analysis(adversary_distance, perturb_list, 
		perturbation_threshold=adversarial_params['perturbation_threshold'][1])]
	LInf_running_min = np.minimum(LInf_running_min, adversary_distance)
	np.savetxt('adversarial_images/' + network_name_str + '/' + attack_name + '_distances.txt', adversary_distance)

	attack_name = 'DeepFool_LInf'
	DeepFool_LInf = atk.DeepFool_LInf_attack(attack_dic)
	adversary_found, adversary_distance, adversaries_array, perturb_list = DeepFool_LInf.evaluate_resistance()
	network_dic[attack_name] = [float(s) for s in analysis(adversary_distance, perturb_list, 
		perturbation_threshold=adversarial_params['perturbation_threshold'][1])]
	LInf_running_min = np.minimum(LInf_running_min, adversary_distance)
	np.savetxt('adversarial_images/' + network_name_str + '/' + attack_name + '_distances.txt', adversary_distance)
	
	attack_name = 'MIM'
	MIM = atk.MIM_attack(attack_dic)
	adversary_found, adversary_distance, adversaries_array, perturb_list = MIM.evaluate_resistance()
	network_dic[attack_name] = [float(s) for s in analysis(adversary_distance, perturb_list, 
		perturbation_threshold=adversarial_params['perturbation_threshold'][1])]
	LInf_running_min = np.minimum(LInf_running_min, adversary_distance)
	np.savetxt('adversarial_images/' + network_name_str + '/' + attack_name + '_distances.txt', adversary_distance)
	
	print("\nResults across all LInf attacks...")
	network_dic['all_LInf'] = [float(s) for s in analysis(LInf_running_min, perturb_list=[], perturbation_threshold=adversarial_params['perturbation_threshold'][1])]


	print("\n\n***Performing L-2 Distance Attacks***\n")

	attack_name = 'gaussian'
	gaussian = atk.gaussian_noise_attack(attack_dic)
	adversary_found, adversary_distance, adversaries_array, perturb_list = gaussian.evaluate_resistance()
	network_dic[attack_name] = [float(s) for s in analysis(adversary_distance, perturb_list, 
		perturbation_threshold=adversarial_params['perturbation_threshold'][2])]
	L2_running_min = adversary_distance
	np.savetxt('adversarial_images/' + network_name_str + '/' + attack_name + '_distances.txt', adversary_distance)
	
	attack_name = 'pointwise_L2'
	pointwise_L2 = atk.pointwise_attack_L2(attack_dic)
	adversary_found, adversary_distance, adversaries_array, perturb_list = pointwise_L2.evaluate_resistance()
	network_dic[attack_name] = [float(s) for s in analysis(adversary_distance, perturb_list, 
		perturbation_threshold=adversarial_params['perturbation_threshold'][2])]
	L2_running_min = np.minimum(L2_running_min, adversary_distance)
	np.savetxt('adversarial_images/' + network_name_str + '/' + attack_name + '_distances.txt', adversary_distance)

	attack_name = 'FGM'
	FGM = atk.FGM_attack(attack_dic)
	adversary_found, adversary_distance, adversaries_array, perturb_list = FGM.evaluate_resistance()
	network_dic[attack_name] = [float(s) for s in analysis(adversary_distance, perturb_list, 
		perturbation_threshold=adversarial_params['perturbation_threshold'][2])]
	L2_running_min = np.minimum(L2_running_min, adversary_distance)
	np.savetxt('adversarial_images/' + network_name_str + '/' + attack_name + '_distances.txt', adversary_distance)

	attack_name = 'BIM_L2'
	BIM_L2 = atk.BIM_L2_attack(attack_dic)
	adversary_found, adversary_distance, adversaries_array, perturb_list = BIM_L2.evaluate_resistance()
	network_dic[attack_name] = [float(s) for s in analysis(adversary_distance, perturb_list, 
		perturbation_threshold=adversarial_params['perturbation_threshold'][2])]
	L2_running_min = np.minimum(L2_running_min, adversary_distance)
	np.savetxt('adversarial_images/' + network_name_str + '/' + attack_name + '_distances.txt', adversary_distance)

	attack_name = 'DeepFool_L2'
	DeepFool_L2 = atk.DeepFool_L2_attack(attack_dic)
	adversary_found, adversary_distance, adversaries_array, perturb_list = DeepFool_L2.evaluate_resistance()
	network_dic[attack_name] = [float(s) for s in analysis(adversary_distance, perturb_list, 
		perturbation_threshold=adversarial_params['perturbation_threshold'][2])]
	L2_running_min = np.minimum(L2_running_min, adversary_distance)
	np.savetxt('adversarial_images/' + network_name_str + '/' + attack_name + '_distances.txt', adversary_distance)

	attack_name = 'boundary'
	boundary = atk.boundary_attack(attack_dic,
                num_iterations=adversarial_params['boundary_attack_iterations'],
                log_every_n_steps=adversarial_params['boundary_attack_log_steps'])
	adversary_found, adversary_distance, adversaries_array, perturb_list = boundary.evaluate_resistance()
	network_dic[attack_name] = [float(s) for s in analysis(adversary_distance, perturb_list, 
		perturbation_threshold=adversarial_params['perturbation_threshold'][2])]
	L2_running_min = np.minimum(L2_running_min, adversary_distance)
	np.savetxt('adversarial_images/' + network_name_str + '/' + attack_name + '_distances.txt', adversary_distance)

	print("\nResults across all L2 attacks...")
	network_dic['all_L2'] = [float(s) for s in analysis(L2_running_min, perturb_list=[], perturbation_threshold=adversarial_params['perturbation_threshold'][2])]

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

pre = time.time()
iterative_evaluation(model_params, adversarial_params, training_data, training_labels, evaluation_data, evaluation_labels)
print("Elapsed time is " + str(time.time() - pre))

