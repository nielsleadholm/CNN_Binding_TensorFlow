#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import os
import json
import tfCore_adversarial_attacks as atk
import matplotlib.pyplot as plt
import mltest
import math
import pandas as pd
import time
import foolbox
from PIL import Image
import CNN_module as CNN
import Adversarial_training

def iterative_evaluation(model_params, adversarial_params, training_data, training_labels, evaluation_data, evaluation_labels):

	all_results_df = pd.DataFrame({})

	for iter_num in range(model_params['num_network_duplicates']):

		iter_dic = {}

		print('\nUsing a label smoothing value of ' + str(model_params['label_smoothing']))
		
		training_pretime = time.time()

		if (model_params['train_new_network'] == True) and (model_params['adver_trained'] == True):

			print("\nUsing adversarial training...")
			eval_accuracy, network_name_str = Adversarial_training.adver_training(model_params, iter_num, training_data, 
				training_labels, evaluation_data, evaluation_labels)
			
			iter_dic.update({'testing_accuracy':float(eval_accuracy)})
		
		#Note Adversarial_training uses it's own initializer
		x_placeholder, y_placeholder, dropout_rate_placeholder, var_list, weights, biases, decoder_weights = CNN.initializer_fun(model_params, training_data, training_labels)

		if (model_params['train_new_network'] == True) and (model_params['adver_trained'] == False):
			print("\nUsing standard (non-adversarial) training...")
			training_accuracy, eval_accuracy, network_name_str, eval_sparsity = CNN.network_train(model_params, iter_num, var_list, training_data, 
				training_labels, evaluation_data, evaluation_labels, weights, biases, decoder_weights, x_placeholder=x_placeholder, 
			    y_placeholder=y_placeholder, dropout_rate_placeholder=dropout_rate_placeholder)

			testing_sparsity_float = [dict([key, float(value)] for key, value in eval_sparsity.items())] 
			iter_dic.update({'training_accuracy':float(training_accuracy), 'testing_accuracy':float(eval_accuracy),
				'testing_sparsity':testing_sparsity_float})
		
		if model_params['train_new_network'] == True:
			training_total_time = time.time() - training_pretime
			iter_dic.update({'training_time':training_total_time})

		elif (model_params['train_new_network'] == False) or (model_params['adver_trained'] == True):

			network_name_str = str(iter_num) + model_params['architecture'] + '_adver_trained_' + str(model_params['adver_trained'])

			testing_writer = tf.compat.v1.summary.FileWriter('tensorboard_data/tb_' + network_name_str + '/testing')
			merged = tf.compat.v1.summary.merge_all()

			#Evaluate the accuracy of the loaded model
			predictions, _, scalar_dic, _, _ = getattr(CNN, model_params['architecture'] + '_predictions')(x_placeholder, dropout_rate_placeholder, weights, biases, model_params['dynamic_dic'])
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
				testing_pretime = time.time()

				for test_batch in range(math.ceil(len(testing_labels)/model_params['batch_size'])):

					test_batch_x = testing_data[test_batch*model_params['batch_size']:min((test_batch+1)*model_params['batch_size'], len(testing_labels))]
					test_batch_y = testing_labels[test_batch*model_params['batch_size']:min((test_batch+1)*model_params['batch_size'], len(testing_labels))]

					network_summary, batch_testing_acc = sess.run([merged, total_accuracy], feed_dict={x_placeholder: test_batch_x, y_placeholder: test_batch_y, dropout_rate_placeholder : 0.0})
					accuracy_total += batch_testing_acc

				testing_writer.add_summary(network_summary)

				testing_acc = accuracy_total/len(testing_labels)
				#Convert python variable into a Summary object for TensorFlow
				testing_summary = tf.Summary(value=[tf.Summary.Value(tag="testing_acc", simple_value=testing_acc),])
				testing_writer.add_summary(testing_summary)

				testing_writer.close()
				
				print("Testing Accuracy of loaded model = " + "{:.4f}".format(testing_acc))
				iter_dic.update({'testing_accuracy':float(testing_acc)})

				testing_total_time = time.time() - testing_pretime
				print("Elapsed time for evaluating all test examples is " + str(testing_total_time))

				#On small-memory data-sets, check layer-wise sparsity on all testing data
				if model_params['dataset']!='cifar10':
					testing_sparsity = sess.run(scalar_dic, feed_dict={x_placeholder: testing_data, y_placeholder: testing_labels, dropout_rate_placeholder : 0.0})

					print("\nLayer-wise sparsity:")
					print(testing_sparsity)

					iter_dic.update(testing_sparsity)


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

		#Analyse the activations across layersto clean vs. adversarial images
		if model_params['dynamic_dic']['analysis_var'] == 'Activations_across_layers':
			analyse_activations(adversarial_params, evaluation_data, evaluation_labels, x_placeholder, var_list, weights, biases, network_name_str, iter_num, model_params['dynamic_dic'], dropout_rate_placeholder)


		update_dic, adver_pred_dic, adver_data_dic = carry_out_attacks(model_params, adversarial_params, getattr(CNN, model_params['architecture'] + '_predictions'), evaluation_data, evaluation_labels, 
			x_placeholder, var_list, weights, biases, network_name_str, iter_num, model_params['dynamic_dic'])
		
		iter_dic.update(update_dic)

		print("\n\nThe cumulative results are...\n")
		print(iter_dic)
		print(iter_num)
		iter_df = pd.DataFrame(data=iter_dic, index=[iter_num], dtype=np.float32)
		all_results_df = all_results_df.append(iter_df)
		print(all_results_df)
		all_results_df.to_pickle('Results.pkl')
		all_results_df.to_csv('Results.csv')


def analyse_activations(adversarial_params, input_data, input_labels, x_placeholder, var_list, weights, biases, network_name_str, iter_num, dynamic_dic, dropout_rate_placeholder):
	
	#Select batch
	batch_size = 512
	input_data, input_labels = input_data[0:batch_size], input_labels[0:batch_size]

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
                'num_attack_examples':batch_size,
                'dynamic_dic':dynamic_dic,
                'batch_size':batch_size,
                'save_images':adversarial_params['save_images'],
                'estimate_gradients':adversarial_params['estimate_gradients']}


	#Generate adversarial attacks for the batch
	attack_name = 'FGM'
	FGM = atk.FGM_attack(attack_dic)
	_, _, adversaries_array, _ = FGM.evaluate_resistance()

	print(np.shape(input_data))
	print(np.shape(adversaries_array))

	plt.imsave('Clean1.png', np.squeeze(input_data[0], axis=2))
	plt.imsave('Adver1.png', np.squeeze(adversaries_array[0], axis=2))

	plt.imsave('Clean2.png', np.squeeze(input_data[batch_size-1], axis=2))
	plt.imsave('Adver2.png', np.squeeze(adversaries_array[batch_size-1], axis=2))

	#Concatenate clean examples and adversarial images as a single batch
	joint_images = np.concatenate((input_data, adversaries_array), axis=0)

	print(np.shape(joint_images))

	# *** analysising activations across layers ***
	#In a standard LeNet netork, compare how activations across the different layers change when presenting a clean vs. corresponding adversarial image

	#Specify the adversarial images/attack type (created in carry_out_attacks), and distance metric
	#Distance metric used to compare layer activations should match that used to generate adversarial images

	#Feed batch to network; returns dictionary with scalar values corresponding to mean distances for each layer
	_, _, scalar_dic, _, _ = getattr(CNN, model_params['architecture'] + '_predictions')(x_placeholder, dropout_rate_placeholder, weights, biases, model_params['dynamic_dic'])

	saver = tf.train.Saver(var_list)
	with tf.Session() as sess:
		saver.restore(sess, ("network_weights_data/" + network_name_str + ".ckpt"))

		activations_across_layers_dic = sess.run(scalar_dic, feed_dict={x_placeholder: joint_images, dropout_rate_placeholder : 0.0})


	print(activations_across_layers_dic['distance_pool1'])
	print(activations_across_layers_dic['distance_pool2'])
	print(activations_across_layers_dic['distance_dense1'])

	# *** analysing activation changes in binding vs max-pooling layers
	# TBC what exactly this analysis will involve
	# Might include comparing activations as I interpolate from one clean image to another clean image
	# Might include comparing activations as I present a clean and then adversarial image vs clean and then Gaussian noisy image

#Create the necessary adversarial examples on a surrogate model to enable performing a transfer attack
def transfer_attack_setup(adversarial_params, evaluation_data, evaluation_labels, training_data, training_labels):

	surrogate_params = {
		'architecture':'LeNet',
		'dynamic_dic':{'dynamic_var':'None', 'analysis_var':'None', 'sparsification_kwinner':0.1, 'sparsification_dropout':0.0},
		'dataset':'cifar10',
	    'learning_rate':0.001,
		'meta_architecture':'CNN',
		'train_new_network':True,
		'adver_trained':False,
		'crossval_bool':True,
		'check_stochasticity':False,
		'num_network_duplicates':1,
	    'training_epochs':45,
	    'Gaussian_noise':None,
	    'dropout_rate':0.25,
	    'He_modifier':1.0,
	    'shift_range':0.1,
	    'label_smoothing':0.0,
	    'L1_regularization_activations1':0.0,
	    'L1_regularization_activations2':0.0,
	    'batch_size':128
		}

	(surrogate_x_placeholder, surrogate_y_placeholder, surrogate_dropout_rate_placeholder, 
		surrogate_var_list, surrogate_weights, surrogate_biases, surrogate_decoder_weights) = CNN.initializer_fun(surrogate_params, training_data, training_labels)

	#Note the surrogate network is indexed with 'iter_num' 999
	if surrogate_params['train_new_network'] == True:
		print("\n Training a surrogate network for transfer attacks...")
		iter_num = 999
		training_accuracy, eval_accuracy, network_name_str, eval_sparsity = CNN.network_train(surrogate_params, iter_num, surrogate_var_list, training_data, 
			training_labels, evaluation_data, evaluation_labels, surrogate_weights, surrogate_biases, surrogate_decoder_weights, x_placeholder=surrogate_x_placeholder, 
		    y_placeholder=surrogate_y_placeholder, dropout_rate_placeholder=surrogate_dropout_rate_placeholder)

	surrogate_dic = {'model_prediction_function':getattr(CNN, surrogate_params['architecture'] + '_predictions'),
        'model_weights':"network_weights_data/" + network_name_str + ".ckpt",
        'var_list':surrogate_var_list,
        'weights_dic':surrogate_weights,
        'biases_dic':surrogate_biases,
        'input_data':evaluation_data,
        'input_labels':evaluation_labels,
        'input_placeholder':surrogate_x_placeholder,
        'dropout_rate_placeholder':0.0,
        'output_directory':'surrogate_dir',
        'num_attack_examples':adversarial_params['num_attack_examples'],
        'dynamic_dic':surrogate_params['dynamic_dic'],
        'batch_size':surrogate_params['batch_size'],
        'save_images':adversarial_params['save_images'],
        'estimate_gradients':adversarial_params['estimate_gradients']}

	#Pass the surrogate model

	# FGM = atk.FGM_attack(surrogate_dic)
	# _, FGM_distances, FGM_adversaries_array, _ = FGM.evaluate_resistance()

	print("Using DeepFool rather than FGM for transfer generation.")
	FGM = atk.DeepFool_L2_attack(surrogate_dic)
	_, FGM_distances, FGM_adversaries_array, _ = FGM.evaluate_resistance()


	BIM_L2 = atk.BIM_L2_attack(surrogate_dic)
	_, BIM_L2_distances, BIML2_adversaries_array, _ = BIM_L2.evaluate_resistance()



	# FGSM = atk.FGSM_attack(surrogate_dic)
	# _, FGSM_distances, FGSM_adversaries_array, _ = FGSM.evaluate_resistance()

	FGSM = atk.DeepFool_LInf_attack(surrogate_dic)
	_, FGSM_distances, FGSM_adversaries_array, _ = FGSM.evaluate_resistance()

	BIM_LInf = atk.BIM_Linfinity_attack(surrogate_dic)
	_, BIM_LInf_distances, BIM_LInf_adversaries_array, _ = BIM_LInf.evaluate_resistance()


	if os.path.exists('transfer_images/') == 0:
		try:
			os.mkdir('transfer_images/')
		except OSError:
			pass

	print("Providing results for adversaries generated on *surrogate* model...")
	print("Median FGM distance is " + str(np.median(FGM_distances)))
	print("Median BIM-L2 distance is " + str(np.median(BIM_L2_distances)))
	print("Median FGSM distance is " + str(np.median(FGSM_distances)))
	print("Median BIM-LInf distance is " + str(np.median(BIM_LInf_distances)))

	np.save('transfer_images/FGM', FGM_adversaries_array, allow_pickle=True)
	np.save('transfer_images/BIM_L2', BIML2_adversaries_array, allow_pickle=True)
	np.save('transfer_images/FGSM', FGSM_adversaries_array, allow_pickle=True)
	np.save('transfer_images/BIM_LInf', BIM_LInf_adversaries_array, allow_pickle=True)


def carry_out_attacks(model_params, adversarial_params, pred_function, input_data, input_labels, x_placeholder, var_list, weights, biases, network_name_str, iter_num, dynamic_dic):

	#Create directory for storing images
	if os.path.exists('adversarial_images/') == 0:
		os.mkdir('adversarial_images/')

	attack_dic = {'model_prediction_function':pred_function,
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
                'save_images':adversarial_params['save_images'],
                'estimate_gradients':adversarial_params['estimate_gradients']}

	network_dic_distances = {} #Hold the attack specific distances results for a given network
	network_dic_adver_accuracies = {} #As above but for accuracies on adversarial examples

	adver_pred_dic = {}
	adver_data_dic = {}

	L0_running_min = [] #Keep track of the minimum adversarial perturbation required for each image
	LInf_running_min = [] #Keep track of the minimum adversarial perturbation required for each image
	L2_running_min = [] #Keep track of the minimum adversarial perturbation required for each image

	print("\n\n***Performing L-0 Distance Attacks***\n")

	attack_name = 'pointwise_L0'
	pointwise_L0 = atk.pointwise_attack_L0(attack_dic)
	adversary_found, adversary_distance, adversaries_array, adversary_labels = pointwise_L0.evaluate_resistance()
	network_dic_distances[attack_name + '_distances'], network_dic_adver_accuracies[attack_name + '_accuracy'] = analysis(adversary_distance, adversary_labels, 
		perturbation_threshold=adversarial_params['perturbation_threshold']['L0'])
	L0_running_min = adversary_distance
	np.savetxt('adversarial_images/' + network_name_str + '/' + attack_name + '_distances.txt', adversary_distance)

	attack_name = 'salt'
	salt = atk.salt_pepper_attack(attack_dic)
	adversary_found, adversary_distance, adversaries_array, adversary_labels = salt.evaluate_resistance()
	network_dic_distances[attack_name + '_distances'], network_dic_adver_accuracies[attack_name + '_accuracy'] = analysis(adversary_distance, adversary_labels, 
		perturbation_threshold=adversarial_params['perturbation_threshold']['L0'])
	L0_running_min = np.minimum(L0_running_min, adversary_distance) #Take element-wise minimum
	np.savetxt('adversarial_images/' + network_name_str + '/' + attack_name + '_distances.txt', adversary_distance)

	print("\nResults across all L0 attacks...")
	attack_name = 'all_L0'
	network_dic_distances[attack_name + '_distances'], network_dic_adver_accuracies[attack_name + '_accuracy'] = analysis(L0_running_min, adversary_labels, 
		perturbation_threshold=adversarial_params['perturbation_threshold']['L0'])


	print("\n\n***Performing L-Inf Distance Attacks***\n")

	attack_name = 'FGSM'
	FGSM = atk.FGSM_attack(attack_dic)
	adversary_found, adversary_distance, adversaries_array, adversary_labels = FGSM.evaluate_resistance()
	network_dic_distances[attack_name + '_distances'], network_dic_adver_accuracies[attack_name + '_accuracy'] = analysis(adversary_distance, adversary_labels, 
		perturbation_threshold=adversarial_params['perturbation_threshold']['LInf'])
	LInf_running_min = adversary_distance
	np.savetxt('adversarial_images/' + network_name_str + '/' + attack_name + '_distances.txt', adversary_distance)

	adver_data_dic[attack_name] = adversaries_array
	adver_pred_dic[attack_name] = adversary_labels

	#LInf transfer attack
	FGSM_adversaries_array = np.load('transfer_images/FGSM.npy') #Load starting adversaries created against a surrogate CNN model
	BIM_LInf_adversaries_array = np.load('transfer_images/BIM_LInf.npy')
	starting_adversaries = np.asarray((FGSM_adversaries_array, BIM_LInf_adversaries_array))
	attack_name = 'transfer_LInf'
	transferLInf = atk.transfer_attack_LInf(attack_dic, starting_adversaries)
	adversary_distance = transferLInf.evaluate_resistance()
	adversary_labels = []
	network_dic_distances[attack_name + '_distances'], network_dic_adver_accuracies[attack_name + '_accuracy'] = analysis(adversary_distance, adversary_labels, 
		perturbation_threshold=adversarial_params['perturbation_threshold']['LInf'])
	LInf_running_min = np.minimum(LInf_running_min, adversary_distance)
	np.savetxt('adversarial_images/' + network_name_str + '/' + attack_name + '_distances.txt', adversary_distance)

	attack_name = 'BIM_LInf'
	BIM_LInf = atk.BIM_Linfinity_attack(attack_dic)
	adversary_found, adversary_distance, adversaries_array, adversary_labels = BIM_LInf.evaluate_resistance()
	network_dic_distances[attack_name + '_distances'], network_dic_adver_accuracies[attack_name + '_accuracy'] = analysis(adversary_distance, adversary_labels, 
		perturbation_threshold=adversarial_params['perturbation_threshold']['LInf'])
	LInf_running_min = np.minimum(LInf_running_min, adversary_distance)
	np.savetxt('adversarial_images/' + network_name_str + '/' + attack_name + '_distances.txt', adversary_distance)
	adver_data_dic[attack_name] = adversaries_array
	adver_pred_dic[attack_name] = adversary_labels

	attack_name = 'DeepFool_LInf'
	DeepFool_LInf = atk.DeepFool_LInf_attack(attack_dic)
	adversary_found, adversary_distance, adversaries_array, adversary_labels = DeepFool_LInf.evaluate_resistance()
	network_dic_distances[attack_name + '_distances'], network_dic_adver_accuracies[attack_name + '_accuracy'] = analysis(adversary_distance, adversary_labels, 
		perturbation_threshold=adversarial_params['perturbation_threshold']['LInf'])
	LInf_running_min = np.minimum(LInf_running_min, adversary_distance)
	np.savetxt('adversarial_images/' + network_name_str + '/' + attack_name + '_distances.txt', adversary_distance)
	
	attack_name = 'MIM'
	MIM = atk.MIM_attack(attack_dic)
	adversary_found, adversary_distance, adversaries_array, adversary_labels = MIM.evaluate_resistance()
	network_dic_distances[attack_name + '_distances'], network_dic_adver_accuracies[attack_name + '_accuracy'] = analysis(adversary_distance, adversary_labels, 
		perturbation_threshold=adversarial_params['perturbation_threshold']['LInf'])
	LInf_running_min = np.minimum(LInf_running_min, adversary_distance)
	np.savetxt('adversarial_images/' + network_name_str + '/' + attack_name + '_distances.txt', adversary_distance)
	
	# attack_name = 'hop_skip_LInf'
	# hop_skip_LInf = atk.hop_skip_attack_LInf(attack_dic,
 #                num_iterations=adversarial_params['boundary_attack_iterations'],
 #                log_every_n_steps=adversarial_params['boundary_attack_log_steps'])
	# adversary_found, adversary_distance, adversaries_array, adversary_labels = hop_skip_LInf.evaluate_resistance()
	# network_dic_distances[attack_name + '_distances'], network_dic_adver_accuracies[attack_name + '_accuracy'] = analysis(adversary_distance, adversary_labels, 
	# 	perturbation_threshold=adversarial_params['perturbation_threshold']['LInf'])
	# LInf_running_min = np.minimum(LInf_running_min, adversary_distance)
	# np.savetxt('adversarial_images/' + network_name_str + '/' + attack_name + '_distances.txt', adversary_distance)


	print("\nResults across all LInf attacks...")
	attack_name = 'all_LInf'
	network_dic_distances[attack_name + '_distances'], network_dic_adver_accuracies[attack_name + '_accuracy'] = analysis(LInf_running_min, adversary_labels, 
		perturbation_threshold=adversarial_params['perturbation_threshold']['LInf'])

	print("\n\n***Performing L-2 Distance Attacks***\n")

	attack_name = 'gaussian'
	gaussian = atk.gaussian_noise_attack(attack_dic)
	adversary_found, adversary_distance, adversaries_array, adversary_labels = gaussian.evaluate_resistance()
	network_dic_distances[attack_name + '_distances'], network_dic_adver_accuracies[attack_name + '_accuracy'] = analysis(adversary_distance, adversary_labels, 
		perturbation_threshold=adversarial_params['perturbation_threshold']['L2'])
	L2_running_min = adversary_distance
	np.savetxt('adversarial_images/' + network_name_str + '/' + attack_name + '_distances.txt', adversary_distance)
	
	adver_data_dic[attack_name] = adversaries_array
	adver_pred_dic[attack_name] = adversary_labels

	#L2 transfer attack
	FGM_adversaries_array = np.load('transfer_images/FGM.npy') #Load starting adversaries created against a surrogate CNN model
	BIML2_adversaries_array = np.load('transfer_images/BIM_L2.npy')
	starting_adversaries = np.asarray((FGM_adversaries_array, BIML2_adversaries_array))
	attack_name = 'transfer_L2'
	transferL2 = atk.transfer_attack_L2(attack_dic, starting_adversaries)
	adversary_distance = transferL2.evaluate_resistance()
	adversary_labels = []
	network_dic_distances[attack_name + '_distances'], network_dic_adver_accuracies[attack_name + '_accuracy'] = analysis(adversary_distance, adversary_labels, 
		perturbation_threshold=adversarial_params['perturbation_threshold']['L2'])
	L2_running_min = np.minimum(L2_running_min, adversary_distance)
	np.savetxt('adversarial_images/' + network_name_str + '/' + attack_name + '_distances.txt', adversary_distance)

	attack_name = 'pointwise_L2'
	pointwise_L2 = atk.pointwise_attack_L2(attack_dic)
	adversary_found, adversary_distance, adversaries_array, adversary_labels = pointwise_L2.evaluate_resistance()
	network_dic_distances[attack_name + '_distances'], network_dic_adver_accuracies[attack_name + '_accuracy'] = analysis(adversary_distance, adversary_labels, 
		perturbation_threshold=adversarial_params['perturbation_threshold']['L2'])
	L2_running_min = np.minimum(L2_running_min, adversary_distance)
	np.savetxt('adversarial_images/' + network_name_str + '/' + attack_name + '_distances.txt', adversary_distance)

	attack_name = 'FGM'
	FGM = atk.FGM_attack(attack_dic)
	adversary_found, adversary_distance, adversaries_array, adversary_labels = FGM.evaluate_resistance()
	network_dic_distances[attack_name + '_distances'], network_dic_adver_accuracies[attack_name + '_accuracy'] = analysis(adversary_distance, adversary_labels, 
		perturbation_threshold=adversarial_params['perturbation_threshold']['L2'])
	L2_running_min = np.minimum(L2_running_min, adversary_distance)
	np.savetxt('adversarial_images/' + network_name_str + '/' + attack_name + '_distances.txt', adversary_distance)

	attack_name = 'BIM_L2'
	BIM_L2 = atk.BIM_L2_attack(attack_dic)
	adversary_found, adversary_distance, adversaries_array, adversary_labels = BIM_L2.evaluate_resistance()
	network_dic_distances[attack_name + '_distances'], network_dic_adver_accuracies[attack_name + '_accuracy'] = analysis(adversary_distance, adversary_labels, 
		perturbation_threshold=adversarial_params['perturbation_threshold']['L2'])
	L2_running_min = np.minimum(L2_running_min, adversary_distance)
	np.savetxt('adversarial_images/' + network_name_str + '/' + attack_name + '_distances.txt', adversary_distance)

	attack_name = 'DeepFool_L2'
	DeepFool_L2 = atk.DeepFool_L2_attack(attack_dic)
	adversary_found, adversary_distance, adversaries_array, adversary_labels = DeepFool_L2.evaluate_resistance()
	network_dic_distances[attack_name + '_distances'], network_dic_adver_accuracies[attack_name + '_accuracy'] = analysis(adversary_distance, adversary_labels, 
		perturbation_threshold=adversarial_params['perturbation_threshold']['L2'])
	L2_running_min = np.minimum(L2_running_min, adversary_distance)
	np.savetxt('adversarial_images/' + network_name_str + '/' + attack_name + '_distances.txt', adversary_distance)

	# # # attack_name = 'BrendelBethge'
	# # # BrendelBethge = atk.brendel_bethge_attack_L2(attack_dic)
	# # # adversary_found, adversary_distance, adversaries_array, adversary_labels = BrendelBethge.evaluate_resistance()
	# # # network_dic_distances[attack_name + '_distances'], network_dic_adver_accuracies[attack_name + '_accuracy'] = analysis(adversary_distance, adversary_labels, 
	# # # 	perturbation_threshold=adversarial_params['perturbation_threshold']['L2'])
	# # # L2_running_min = np.minimum(L2_running_min, adversary_distance)
	# # # np.savetxt('adversarial_images/' + network_name_str + '/' + attack_name + '_distances.txt', adversary_distance)

	# attack_name = 'boundary'
	# boundary = atk.boundary_attack(attack_dic,
 #                num_iterations=adversarial_params['boundary_attack_iterations'],
 #                log_every_n_steps=adversarial_params['boundary_attack_log_steps'])
	# adversary_found, adversary_distance, adversaries_array, adversary_labels = boundary.evaluate_resistance()
	# network_dic_distances[attack_name + '_distances'], network_dic_adver_accuracies[attack_name + '_accuracy'] = analysis(adversary_distance, adversary_labels, 
	# 	perturbation_threshold=adversarial_params['perturbation_threshold']['L2'])
	# L2_running_min = np.minimum(L2_running_min, adversary_distance)
	# np.savetxt('adversarial_images/' + network_name_str + '/' + attack_name + '_distances.txt', adversary_distance)

	# adver_data_dic[attack_name] = adversaries_array
	# adver_pred_dic[attack_name] = adversary_labels

	# attack_name = 'hop_skip_L2'
	# #Uses the same number of iterations and log-steps as the boundary attack
	# hop_skip_L2 = atk.hop_skip_attack_L2(attack_dic,
 #                num_iterations=adversarial_params['boundary_attack_iterations'],
 #                log_every_n_steps=adversarial_params['boundary_attack_log_steps'])
	# adversary_found, adversary_distance, adversaries_array, adversary_labels = hop_skip_L2.evaluate_resistance()
	# network_dic_distances[attack_name + '_distances'], network_dic_adver_accuracies[attack_name + '_accuracy'] = analysis(adversary_distance, adversary_labels, 
	# 	perturbation_threshold=adversarial_params['perturbation_threshold']['L2'])
	# L2_running_min = np.minimum(L2_running_min, adversary_distance)
	# np.savetxt('adversarial_images/' + network_name_str + '/' + attack_name + '_distances.txt', adversary_distance)


	print("\nResults across all L2 attacks...")
	attack_name = 'all_L2'
	network_dic_distances[attack_name + '_distances'], network_dic_adver_accuracies[attack_name + '_accuracy'] = analysis(L2_running_min, adversary_labels, 
		perturbation_threshold=adversarial_params['perturbation_threshold']['L2'])


	return {**network_dic_distances, **network_dic_adver_accuracies}, adver_pred_dic, adver_data_dic

#Find the proportion of examples above a given threshold
def threshold_accuracy(perturbation_threshold, adversary_distance):
	#Boolean mask where adversary distances is above the threshold
	threshold_exceeded = (adversary_distance > perturbation_threshold)
	threshold_acc = np.sum(threshold_exceeded)/len(adversary_distance)
	return threshold_acc

def analysis(adversary_distance, adversary_labels, perturbation_threshold):

	median_distance = np.median(adversary_distance)
	threshold_acc = threshold_accuracy(perturbation_threshold, adversary_distance)
	print("The median distance is " + str(median_distance))
	print("The classiciation accuracy at a threshold of " + str(perturbation_threshold) + " is " + str(threshold_acc))

	return median_distance, threshold_acc


if __name__ == '__main__':

	#Parameters determining the networks that will be trained and evaluated
	#dynamic_var allows for temporary changes to a model's forward pass, such as adding stochasticity or ablating layers; see CNN_module for details
	#Options for dynamic_var are:
		# "Add_logit_noise"
		# "Ablate_unpooling"
		# "Ablate_gradient_unpooling"
		# "Ablate_binding"
		# "Ablate_maxpooling"
		# "Ablate_max&gradient"
		# "Ablate_max&unpool"
		# "kwinner_activations" # In gradient unpooling, use the k-largest activations rather than deriving the mask from gradients
		# "kloser_gradients" # In gradient unpooling, use the k-smallest gradients for deriving the mask
	#analysis_var allows for special contrainsts and measures to be placed on the forward pass
	#Options for analysis_var are:
		# "Activations_across_layers" (taking a batch of clean and adversarial examples, measure how different activations are across the layers of a LeNet-5 CNN)
	#sparsification_kwinner determines the sparsity of gradient unpooling activations (select a value between 0 and 1); note the actual
	# binding activation sparsity may differ (specifically be more sparse) depending on how sparse the activations are before the gradient unpooling operation
	#sparsification_dropout applies a dropout rate to binding layers at all times (i.e. including during testing) to measure the effect of activation sparsity
	#meta_architecture determines whether a standard CNN or e.g. an auto-encoder ('SAE') architecture is to be used
	#adver_trained, if True, will train or load an adversarially trained model
	#Gaussian noise - set to either None or the value of desired std to add noise to training data
	#predictive weighting - scales how important supervised prediction is vs. unsupervised reconstruction in the SAE cost function
	#He_modifier: multiplicative factor by which the initialized weights are scaled; note the scalar value selected here is multiplied
	# by 2.0 (the value used in the original publication) before scaling the weights; therefore use 1.0 for default
	#shift_range determines shifting used in CIFAR-10 data-augmentationl; for a typical value, use 0.1
	#L1 regularization has two separate constants:
		# For a LeNet model, activations1 refers to the penultimate fully-connected layer (as in Guo et al), and activations2 has no effect
		# For a control model, activations1 refers to unpooling, activations2 to gradient unpooling
		# For any other models, these activations have no regularizing effect
	model_params = {
		'architecture':'BindingCNN',
		'dynamic_dic':{'dynamic_var':'None', 'analysis_var':'None', 
			'sparsification_kwinner':0.4, 'sparsification_dropout':0.0},
		'dataset':'mnist',
	    'learning_rate':0.001,
		'meta_architecture':'CNN',
		'train_new_network':True,
		'adver_trained':False,
		'crossval_bool':True,
		'check_stochasticity':False,
		'num_network_duplicates':15,
	    'training_epochs':45,
	    'Gaussian_noise':None,
	    'salt&pepper_noise':None,
	    'dropout_rate':0.25,
	    'predictive_weighting':0.01,
	    'He_modifier':1.0,
	    'shift_range':0.1,
	    'label_smoothing':0.1,
	    'MLP_layer_1_dim':120,
	    'MLP_layer_2_dim':84,
	    'L1_regularization_activations1':0.0,
	    'L1_regularization_activations2':0.0,
	    'batch_size':128
		}

	#Parameters determining the adversarial attacks
	#perturbation_threshold defines the threshold of the L-0, L-inf, and L-2 distances at which accuracy is evaluated
	adversarial_params = {
		'num_attack_examples':512,
		'transfer_attack_setup':False,
		'estimate_gradients':False,
	    'boundary_attack_iterations':1000,
	    'boundary_attack_log_steps':1000,
	    'perturbation_threshold':{'L0':12, 'LInf':0.3,  'L2':1.5},
	    'save_images':True
	    }

	#Confirm chosen hyperparameters
	print("\n Model hyper-parameters:")
	print(model_params)

	#Confirm chosen hyperparameters
	print("\n Adversarial hyper-parameters:")
	print(adversarial_params)

	#Specify training and cross-validation data
	(training_data, training_labels, testing_data, testing_labels, crossval_data, crossval_labels) = CNN.data_setup(model_params)

	#Option to run the training and adversarial testing with a cross-validation data-set, rather than the true test dataset
	if model_params['crossval_bool'] == True:
		evaluation_data = crossval_data
		evaluation_labels = crossval_labels
	else:
		evaluation_data = testing_data
		evaluation_labels = testing_labels

	if adversarial_params['transfer_attack_setup'] == True:
		transfer_attack_setup(adversarial_params, evaluation_data, evaluation_labels, training_data, training_labels)

	else:
		pre = time.time()
		iterative_evaluation(model_params, adversarial_params, training_data, training_labels, evaluation_data, evaluation_labels)
		print("Elapsed time is " + str(time.time() - pre))

