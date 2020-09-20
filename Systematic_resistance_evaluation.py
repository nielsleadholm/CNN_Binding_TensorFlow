#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import os
import tfCore_adversarial_attacks as atk
import mltest
import math
import yaml
import pandas as pd
import time
import foolbox
import CNN_module as CNN
import Adversarial_training

def iterative_evaluation(model_params, adversarial_params, training_data, training_labels, evaluation_data, evaluation_labels):

	all_results_df = pd.DataFrame({})

	for iter_num in range(model_params['num_network_duplicates']):

		iter_dic = {}
		
		training_pretime = time.time()

		if (model_params['train_new_network'] == True) and (model_params['adver_trained'] == True):

			print("\nUsing adversarial training...")
			eval_accuracy, network_name_str = Adversarial_training.adver_training(model_params, iter_num, training_data, 
				training_labels, evaluation_data, evaluation_labels)
			
			iter_dic.update({'testing_accuracy':float(eval_accuracy)})
		
		#Note Adversarial_training uses it's own initializer
		x_placeholder, y_placeholder, dropout_rate_placeholder, var_list, weights, biases = CNN.initializer_fun(model_params, training_data, training_labels)

		if (model_params['train_new_network'] == True) and (model_params['adver_trained'] == False):
			print("\nTraining new network...")
			training_accuracy, eval_accuracy, network_name_str, eval_sparsity = CNN.network_train(model_params, iter_num, var_list, training_data, 
				training_labels, evaluation_data, evaluation_labels, weights, biases, x_placeholder=x_placeholder, 
			    y_placeholder=y_placeholder, dropout_rate_placeholder=dropout_rate_placeholder)

			testing_sparsity_float = [dict([key, float(value)] for key, value in eval_sparsity.items())] 
			iter_dic.update({'training_accuracy':float(training_accuracy), 'testing_accuracy':float(eval_accuracy),
				'testing_sparsity':testing_sparsity_float})
		
			training_total_time = time.time() - training_pretime
			iter_dic.update({'training_time':training_total_time})

		#Evaluate the accuracy of a pre-trained model and optionally run mltest suite
		elif (model_params['train_new_network'] == False) or (model_params['adver_trained'] == True):

			network_name_str = str(iter_num) + model_params['architecture']

			testing_writer = tf.compat.v1.summary.FileWriter('tensorboard_data/tb_' + network_name_str + '/testing')
			merged = tf.compat.v1.summary.merge_all()

			#Evaluate the accuracy of the loaded model
			predictions, scalar_dic = getattr(CNN, model_params['architecture'] + '_predictions')(x_placeholder, dropout_rate_placeholder, weights, biases, model_params['dynamic_dic'])
			correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(y_placeholder, 1))
			total_accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.float32)) #Used to store batched accuracy for evaluating the test dataset

			#Used for the mltest suite:
			dummy_cost = tf.reduce_mean(tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(logits=predictions, labels=y_placeholder))
			dummy_op = tf.compat.v1.train.AdamOptimizer(learning_rate=model_params['learning_rate']).minimize(dummy_cost)

			saver = tf.train.Saver(var_list)
			with tf.Session() as sess:
				saver.restore(sess, ("network_weights_data/" + network_name_str + ".ckpt"))

				if model_params['test_suite_bool'] == True:
					print("\nRunning mltest for NaN/Inf values on a single batch...")
					mltest.test_suite(
						predictions,
						train_op=dummy_op,
						feed_dict={x_placeholder: testing_data[0:model_params['batch_size']], 
							y_placeholder: testing_labels[0:model_params['batch_size']], dropout_rate_placeholder : 0.0}, 
						var_list=var_list,
						test_all_inputs_dependent=False,
						test_other_vars_dont_change=False,
						test_output_range=False,
						test_nan_vals=True,
						test_inf_vals=True)
					print("--> passed mltest check.")

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

				#On small-memory models, check layer-wise sparsity on all testing data
				if model_params['dataset']!='cifar10':
					testing_sparsity = sess.run(scalar_dic, feed_dict={x_placeholder: testing_data, y_placeholder: testing_labels, dropout_rate_placeholder : 0.0})

					print("\nLayer-wise sparsity:")
					print(testing_sparsity)

					iter_dic.update(testing_sparsity)


		# Optional check for stochasticity in a model's predictions
		if model_params['check_stochasticity'] == True:
			stochast_dic = {'model_prediction_function':getattr(CNN, model_params['architecture'] + '_predictions'),
		        'model_weights':"network_weights_data/" + network_name_str + ".ckpt",
		        'var_list':var_list,
		        'weights_dic':weights,
		        'biases_dic':biases,
		        'input_data':evaluation_data,
		        'input_labels':evaluation_labels,
		        'input_placeholder':x_placeholder,
		        'dropout_rate_placeholder':0.0,
		        'output_directory':'stochastic_dir',
		        'num_attack_examples':adversarial_params['num_attack_examples'],
		        'dynamic_dic':model_params['dynamic_dic'],
		        'batch_size':128,
		        'save_images':False,
		        'estimate_gradients':False}

			stoch_check = atk.check_stochasticity(stochast_dic)
			stoch_check.perform_check()

		update_dic = carry_out_attacks(model_params, adversarial_params, getattr(CNN, model_params['architecture'] + '_predictions'), evaluation_data, evaluation_labels, 
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

#Create the necessary adversarial examples on a surrogate model to enable performing a transfer attack
def transfer_attack_setup(model_params, adversarial_params, evaluation_data, evaluation_labels, training_data, training_labels):

	surrogate_params = model_params

	(surrogate_x_placeholder, surrogate_y_placeholder, surrogate_dropout_rate_placeholder, 
		surrogate_var_list, surrogate_weights, surrogate_biases) = CNN.initializer_fun(surrogate_params, training_data, training_labels)

	#Note the surrogate network is indexed with 'iter_num' 999
	if surrogate_params['train_new_network'] == True:
		print("\n Training a surrogate network for transfer attacks...")
		iter_num = 999
		training_accuracy, eval_accuracy, network_name_str, eval_sparsity = CNN.network_train(surrogate_params, iter_num, surrogate_var_list, training_data, 
			training_labels, evaluation_data, evaluation_labels, surrogate_weights, surrogate_biases, x_placeholder=surrogate_x_placeholder, 
		    y_placeholder=surrogate_y_placeholder, dropout_rate_placeholder=surrogate_dropout_rate_placeholder)

	else:
		network_name_str = str(999) + surrogate_params['architecture']

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

	FGM = atk.FGM_attack(surrogate_dic)
	_, FGM_distances, FGM_adversaries_array, _ = FGM.evaluate_resistance()

	BIM_L2 = atk.BIM_L2_attack(surrogate_dic)
	_, BIM_L2_distances, BIML2_adversaries_array, _ = BIM_L2.evaluate_resistance()



	FGSM = atk.FGSM_attack(surrogate_dic)
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
                'estimate_gradients':False}

	network_dic_distances = {} #Hold the attack specific distances results for a given network
	network_dic_adver_accuracies = {} #As above but for accuracies on adversarial examples

	L0_running_min = [] #Keep track of the minimum adversarial perturbation required for each image
	LInf_running_min = [] 
	L2_running_min = []

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
	np.savetxt('adversarial_images/' + network_name_str + '/' + attack_name + '_distances.txt', L0_running_min)


	print("\n\n***Performing L-Inf Distance Attacks***\n")

	attack_name = 'FGSM'
	FGSM = atk.FGSM_attack(attack_dic)
	adversary_found, adversary_distance, adversaries_array, adversary_labels = FGSM.evaluate_resistance()
	network_dic_distances[attack_name + '_distances'], network_dic_adver_accuracies[attack_name + '_accuracy'] = analysis(adversary_distance, adversary_labels, 
		perturbation_threshold=adversarial_params['perturbation_threshold']['LInf'])
	LInf_running_min = adversary_distance
	np.savetxt('adversarial_images/' + network_name_str + '/' + attack_name + '_distances.txt', adversary_distance)

	if adversarial_params['estimate_gradients'] == True:
		attack_name = 'FGSM_estimated'
		attack_dic['estimate_gradients'] = True #set boolean for passing into attack
		
		FGSM = atk.FGSM_attack(attack_dic)
		adversary_found, adversary_distance, adversaries_array, adversary_labels = FGSM.evaluate_resistance()
		network_dic_distances[attack_name + '_distances'], network_dic_adver_accuracies[attack_name + '_accuracy'] = analysis(adversary_distance, adversary_labels, 
			perturbation_threshold=adversarial_params['perturbation_threshold']['LInf'])
		LInf_running_min = adversary_distance
		np.savetxt('adversarial_images/' + network_name_str + '/' + attack_name + '_distances.txt', adversary_distance)

		attack_dic['estimate_gradients'] = False #reset

	#LInf transfer attack
	starting_adversaries = np.asarray((np.load('transfer_images/standard_network/FGSM.npy'), 
		np.load('transfer_images/standard_network/BIM_LInf.npy'), 
		np.load('transfer_images/binding_network/FGSM.npy'),
		np.load('transfer_images/binding_network/BIM_LInf.npy')))
	attack_name = 'transfer_LInf'
	transferLInf = atk.transfer_attack_LInf(attack_dic, starting_adversaries)
	adversary_distance = transferLInf.evaluate_resistance()
	adversary_labels = [] #dummy variable
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

	if adversarial_params['estimate_gradients'] == True:
		attack_name = 'BIM_LInf_estimated'
		attack_dic['estimate_gradients'] = True #set boolean for passing into attack

		BIM_LInf = atk.BIM_Linfinity_attack(attack_dic)
		adversary_found, adversary_distance, adversaries_array, adversary_labels = BIM_LInf.evaluate_resistance()
		network_dic_distances[attack_name + '_distances'], network_dic_adver_accuracies[attack_name + '_accuracy'] = analysis(adversary_distance, adversary_labels, 
			perturbation_threshold=adversarial_params['perturbation_threshold']['LInf'])
		LInf_running_min = np.minimum(LInf_running_min, adversary_distance)
		np.savetxt('adversarial_images/' + network_name_str + '/' + attack_name + '_distances.txt', adversary_distance)

		attack_dic['estimate_gradients'] = False

	attack_name = 'DeepFool_LInf'
	DeepFool_LInf = atk.DeepFool_LInf_attack(attack_dic)
	adversary_found, adversary_distance, adversaries_array, adversary_labels = DeepFool_LInf.evaluate_resistance()
	network_dic_distances[attack_name + '_distances'], network_dic_adver_accuracies[attack_name + '_accuracy'] = analysis(adversary_distance, adversary_labels, 
		perturbation_threshold=adversarial_params['perturbation_threshold']['LInf'])
	LInf_running_min = np.minimum(LInf_running_min, adversary_distance)
	np.savetxt('adversarial_images/' + network_name_str + '/' + attack_name + '_distances.txt', adversary_distance)
	
	if adversarial_params['estimate_gradients'] == True:
		attack_name = 'DeepFool_LInf_estimated'
		attack_dic['estimate_gradients'] = True #set boolean for passing into attack

		DeepFool_LInf = atk.DeepFool_LInf_attack(attack_dic)
		adversary_found, adversary_distance, adversaries_array, adversary_labels = DeepFool_LInf.evaluate_resistance()
		network_dic_distances[attack_name + '_distances'], network_dic_adver_accuracies[attack_name + '_accuracy'] = analysis(adversary_distance, adversary_labels, 
			perturbation_threshold=adversarial_params['perturbation_threshold']['LInf'])
		LInf_running_min = np.minimum(LInf_running_min, adversary_distance)
		np.savetxt('adversarial_images/' + network_name_str + '/' + attack_name + '_distances.txt', adversary_distance)
		
		attack_dic['estimate_gradients'] = False

	attack_name = 'MIM'
	MIM = atk.MIM_attack(attack_dic)
	adversary_found, adversary_distance, adversaries_array, adversary_labels = MIM.evaluate_resistance()
	network_dic_distances[attack_name + '_distances'], network_dic_adver_accuracies[attack_name + '_accuracy'] = analysis(adversary_distance, adversary_labels, 
		perturbation_threshold=adversarial_params['perturbation_threshold']['LInf'])
	LInf_running_min = np.minimum(LInf_running_min, adversary_distance)
	np.savetxt('adversarial_images/' + network_name_str + '/' + attack_name + '_distances.txt', adversary_distance)

	if adversarial_params['estimate_gradients'] == True:
		attack_name = 'MIM_estimated'
		attack_dic['estimate_gradients'] = True #set boolean for passing into attack

		MIM = atk.MIM_attack(attack_dic)
		adversary_found, adversary_distance, adversaries_array, adversary_labels = MIM.evaluate_resistance()
		network_dic_distances[attack_name + '_distances'], network_dic_adver_accuracies[attack_name + '_accuracy'] = analysis(adversary_distance, adversary_labels, 
			perturbation_threshold=adversarial_params['perturbation_threshold']['LInf'])
		LInf_running_min = np.minimum(LInf_running_min, adversary_distance)
		np.savetxt('adversarial_images/' + network_name_str + '/' + attack_name + '_distances.txt', adversary_distance)

		attack_dic['estimate_gradients'] = False

	print("\nResults across all LInf attacks...")
	attack_name = 'all_LInf'
	network_dic_distances[attack_name + '_distances'], network_dic_adver_accuracies[attack_name + '_accuracy'] = analysis(LInf_running_min, adversary_labels, 
		perturbation_threshold=adversarial_params['perturbation_threshold']['LInf'])
	np.savetxt('adversarial_images/' + network_name_str + '/' + attack_name + '_distances.txt', LInf_running_min)


	print("\n\n***Performing L-2 Distance Attacks***\n")

	attack_name = 'gaussian'
	gaussian = atk.gaussian_noise_attack(attack_dic)
	adversary_found, adversary_distance, adversaries_array, adversary_labels = gaussian.evaluate_resistance()
	network_dic_distances[attack_name + '_distances'], network_dic_adver_accuracies[attack_name + '_accuracy'] = analysis(adversary_distance, adversary_labels, 
		perturbation_threshold=adversarial_params['perturbation_threshold']['L2'])
	L2_running_min = adversary_distance
	np.savetxt('adversarial_images/' + network_name_str + '/' + attack_name + '_distances.txt', adversary_distance)
	
	#L2 transfer attack
	starting_adversaries = np.asarray((np.load('transfer_images/standard_network/FGM.npy'), 
		np.load('transfer_images/standard_network/BIM_L2.npy'), 
		np.load('transfer_images/binding_network/FGM.npy'),
		np.load('transfer_images/binding_network/BIM_L2.npy')))
	attack_name = 'transfer_L2'
	transferL2 = atk.transfer_attack_L2(attack_dic, starting_adversaries)
	adversary_distance = transferL2.evaluate_resistance()
	adversary_labels = [] #dummy variable
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

	if adversarial_params['estimate_gradients'] == True:
		attack_name = 'FGM_estimated'
		attack_dic['estimate_gradients'] = True #set boolean for passing into attack

		FGM = atk.FGM_attack(attack_dic)
		adversary_found, adversary_distance, adversaries_array, adversary_labels = FGM.evaluate_resistance()
		network_dic_distances[attack_name + '_distances'], network_dic_adver_accuracies[attack_name + '_accuracy'] = analysis(adversary_distance, adversary_labels, 
			perturbation_threshold=adversarial_params['perturbation_threshold']['L2'])
		L2_running_min = np.minimum(L2_running_min, adversary_distance)
		np.savetxt('adversarial_images/' + network_name_str + '/' + attack_name + '_distances.txt', adversary_distance)

		attack_dic['estimate_gradients'] = False

	attack_name = 'BIM_L2'
	BIM_L2 = atk.BIM_L2_attack(attack_dic)
	adversary_found, adversary_distance, adversaries_array, adversary_labels = BIM_L2.evaluate_resistance()
	network_dic_distances[attack_name + '_distances'], network_dic_adver_accuracies[attack_name + '_accuracy'] = analysis(adversary_distance, adversary_labels, 
		perturbation_threshold=adversarial_params['perturbation_threshold']['L2'])
	L2_running_min = np.minimum(L2_running_min, adversary_distance)
	np.savetxt('adversarial_images/' + network_name_str + '/' + attack_name + '_distances.txt', adversary_distance)

	if adversarial_params['estimate_gradients'] == True:
		attack_name = 'BIM_L2_estimated'
		attack_dic['estimate_gradients'] = True #set boolean for passing into attack

		BIM_L2 = atk.BIM_L2_attack(attack_dic)
		adversary_found, adversary_distance, adversaries_array, adversary_labels = BIM_L2.evaluate_resistance()
		network_dic_distances[attack_name + '_distances'], network_dic_adver_accuracies[attack_name + '_accuracy'] = analysis(adversary_distance, adversary_labels, 
			perturbation_threshold=adversarial_params['perturbation_threshold']['L2'])
		L2_running_min = np.minimum(L2_running_min, adversary_distance)
		np.savetxt('adversarial_images/' + network_name_str + '/' + attack_name + '_distances.txt', adversary_distance)

		attack_dic['estimate_gradients'] = False

	attack_name = 'DeepFool_L2'
	DeepFool_L2 = atk.DeepFool_L2_attack(attack_dic)
	adversary_found, adversary_distance, adversaries_array, adversary_labels = DeepFool_L2.evaluate_resistance()
	network_dic_distances[attack_name + '_distances'], network_dic_adver_accuracies[attack_name + '_accuracy'] = analysis(adversary_distance, adversary_labels, 
		perturbation_threshold=adversarial_params['perturbation_threshold']['L2'])
	L2_running_min = np.minimum(L2_running_min, adversary_distance)
	np.savetxt('adversarial_images/' + network_name_str + '/' + attack_name + '_distances.txt', adversary_distance)

	if adversarial_params['estimate_gradients'] == True:
		attack_name = 'DeepFool_L2_estimated'
		attack_dic['estimate_gradients'] = True #set boolean for passing into attack

		DeepFool_L2 = atk.DeepFool_L2_attack(attack_dic)
		adversary_found, adversary_distance, adversaries_array, adversary_labels = DeepFool_L2.evaluate_resistance()
		network_dic_distances[attack_name + '_distances'], network_dic_adver_accuracies[attack_name + '_accuracy'] = analysis(adversary_distance, adversary_labels, 
			perturbation_threshold=adversarial_params['perturbation_threshold']['L2'])
		L2_running_min = np.minimum(L2_running_min, adversary_distance)
		np.savetxt('adversarial_images/' + network_name_str + '/' + attack_name + '_distances.txt', adversary_distance)

		attack_dic['estimate_gradients'] = False

	attack_name = 'boundary'
	boundary = atk.boundary_attack(attack_dic,
                num_iterations=adversarial_params['boundary_attack_iterations'],
                log_every_n_steps=adversarial_params['boundary_attack_log_steps'])
	adversary_found, adversary_distance, adversaries_array, adversary_labels = boundary.evaluate_resistance()
	network_dic_distances[attack_name + '_distances'], network_dic_adver_accuracies[attack_name + '_accuracy'] = analysis(adversary_distance, adversary_labels, 
		perturbation_threshold=adversarial_params['perturbation_threshold']['L2'])
	L2_running_min = np.minimum(L2_running_min, adversary_distance)
	np.savetxt('adversarial_images/' + network_name_str + '/' + attack_name + '_distances.txt', adversary_distance)

	print("\nResults across all L2 attacks...")
	attack_name = 'all_L2'
	network_dic_distances[attack_name + '_distances'], network_dic_adver_accuracies[attack_name + '_accuracy'] = analysis(L2_running_min, adversary_labels, 
		perturbation_threshold=adversarial_params['perturbation_threshold']['L2'])
	np.savetxt('adversarial_images/' + network_name_str + '/' + attack_name + '_distances.txt', L2_running_min)


	return {**network_dic_distances, **network_dic_adver_accuracies}

#Find the proportion of examples above a given threshold
def threshold_accuracy(perturbation_threshold, adversary_distance):

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
	
	with open('config_BindingCNNs.yaml') as f:
		params = yaml.load(f, Loader=yaml.FullLoader)

	model_params = params['model_params']
	adversarial_params = params['adversarial_params']

	print("\n Model hyper-parameters:")
	print(model_params)
	print("\n Adversarial hyper-parameters:")
	print(adversarial_params)

	#Specify training and cross-validation/test data
	(training_data, training_labels, testing_data, testing_labels, crossval_data, crossval_labels) = CNN.data_setup(model_params)

	#Option to run the training with a cross-validation data-set
	if model_params['crossval_bool'] == True:
		evaluation_data = crossval_data
		evaluation_labels = crossval_labels
	else:
		evaluation_data = testing_data
		evaluation_labels = testing_labels

	if adversarial_params['transfer_attack_setup'] == True:
		#Note the parameters of the surrogate architecture will use 'model_params'
		transfer_attack_setup(model_params, adversarial_params, evaluation_data, evaluation_labels, training_data, training_labels)

	else:
		pre = time.time()
		iterative_evaluation(model_params, adversarial_params, training_data, training_labels, evaluation_data, evaluation_labels)
		print("Elapsed time is " + str(time.time() - pre))

