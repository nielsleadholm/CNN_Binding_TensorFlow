#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import os
import foolbox
import scipy
import matplotlib.pyplot as plt
from PIL import Image


#Utilizes the FoolBox Python library (link) to implement a variety 
#of adversarial attacks against deep-learning models implimented in TensorFlow's low-level Core API

class parent_attack:
        def __init__(self, model_prediction_function,
                        model_weights,
                        var_list,
                        weights_dic,
                        biases_dic,
                        input_data,
                        input_labels,
                        input_placeholder,
                        dropout_rate_placeholder,
                        output_directory,
                        num_attack_examples=3, 
                        criterion=foolbox.criteria.Misclassification(), 
                        dropout_rate=0.0):
                self.model_prediction_function = model_prediction_function
                self.model_weights = model_weights
                self.var_list = var_list
                self.weights_dic = weights_dic
                self.biases_dic = biases_dic
                self.input_data = input_data
                self.input_labels = input_labels
                self.input_placeholder = input_placeholder
                self.dropout_rate_placeholder = dropout_rate_placeholder
                self.output_directory = output_directory
                self.num_attack_examples = num_attack_examples
                self.criterion = criterion #note by default this is simply foolbox's Misclassification criterion
                self.dropout_rate = dropout_rate #also provided with default value above

        #Define the class attribute, attack_method, to be the Blended Uniform Noise attack by default
        attack_method = foolbox.attacks.BlendedUniformNoiseAttack 
        foolbox_distance_metric = foolbox.distances.MeanSquaredDistance
        attack_type_dir = 'Parent_*_not_advised_*_'

        def evaluate_resistance(self):

            logits = self.model_prediction_function(self.input_placeholder, self.dropout_rate_placeholder, self.weights_dic, self.biases_dic)
            saver = tf.train.Saver(self.var_list) #Define saver object for use later when loading the model weights

            with tf.Session() as session:
                    saver.restore(session, self.model_weights) #Note when restoring weights its important not to run init on the same
                    #variables, as this will over-write the learned weights with randomly initialized ones
                    #Define the foolbox model
                    fmodel = foolbox.models.TensorFlowModel(self.input_placeholder, logits, (0,1)) 

                    print("\nPerforming " + self.attack_type_dir + " attack(s)")
                    print("Evaluating " + str(self.num_attack_examples) + " adversarial example(s)")

                    #Arrays for storing results of the evaluation
                    adversary_found = np.zeros([self.num_attack_examples]) #array of booleans that indicates if an adversary was found for a particular image
                    adversary_distance = np.zeros([self.num_attack_examples])
                    #print("Input data shape is " + str(np.shape(self.input_data)))
                    adversary_arrays = np.zeros([self.num_attack_examples, self.input_data.shape[1], self.input_data.shape[2], 1])
                    #print("Storage shape is " + str(np.shape(adversary_arrays)))
                    perturb_list = []

                    for ii in range(self.num_attack_examples):

                            execution_data = self.input_data[ii, :, :]
                            execution_label = np.argmax(self.input_labels[ii,:])

                            #Check the predicted label of the network prior to carrying out the attack isn't already incorrect
                            # print(np.shape(execution_data))
                            #execution_data[None, :, :, :] = execution_data
                            execution_data_reshape = np.expand_dims(execution_data, axis=0)
                            # print(np.shape(execution_data_reshape))
                            pre_label = np.argmax(fmodel.forward(execution_data_reshape))
                            #pre_label = np.argmax(fmodel.predictions(execution_data[:, :, :]))
                            if (pre_label != execution_label):
                                print("The model predicted a " + str(pre_label) + " when the ground-truth label is " + str(execution_label))

                            #Carry out the attack
                            print("Beginning attack, ground truth label is " + str(execution_label))
                            self.attack_fmodel = self.attack_method(model=fmodel, criterion=self.criterion, distance=self.foolbox_distance_metric)
                            adversarial_image_fmodel = self.create_adversarial(execution_data, execution_label)
                            
                            #Check the output of the adversarial attack

                            # print("*type* Executation datatype is " + str(execution_data.dtype))
                            # print("Adversarial datatype is " + str(adversarial_image_fmodel.dtype))

                            adversarial_image_fmodel_reshape = np.expand_dims(adversarial_image_fmodel, axis=0)
                            adver_pred = fmodel.forward(adversarial_image_fmodel_reshape)


                            if np.any(adversarial_image_fmodel == None):
                                print("\n\n *** No adversarial image found - attack returned None *** \n\n")


                            #If the image is still correctly classified, iteratively perturb it by increasing the adversarial mask until the image is misclassified
                            elif np.argmax(adver_pred) == execution_label:
                                print("\n *** The model correctly predicted " + str(np.argmax(adver_pred)) + " with a ground truth of " + str(execution_label))
                                print("Iteratively enhancing perturbation until misclassified again...")
                                multiplier = 1.001 #Initialize the multiplier
                                #Evaluate with the initial perturbation
                                adversarial_image_perturbed = execution_data_reshape + multiplier*(adversarial_image_fmodel_reshape - execution_data_reshape)
                                adver_pred_perturbed = fmodel.forward(adversarial_image_perturbed)

                                while np.argmax(adver_pred_perturbed) == execution_label:
                                    multiplier += 0.001
                                    adversarial_image_perturbed = execution_data_reshape + multiplier*(adversarial_image_fmodel_reshape - execution_data_reshape) #Check for 'correct' classification due to numerical issues
                                    adver_pred_perturbed = fmodel.forward(adversarial_image_perturbed)

                                print("Perturbed classification is " + str(np.argmax(adver_pred_perturbed)) + " following additional perturbation of " +str(multiplier))
                                #print(np.shape(adversarial_image_fmodel))
                                adversarial_image_fmodel = adversarial_image_perturbed[0, :, : , :] #update the adversarial image to reflect the genuinely misclassified image
                                #print(np.shape(adversarial_image_fmodel))
                                #perturb_list.append(multiplier)
                                # if np.argmax(adver_pred_perturbed) != execution_label:
                                #     print("However, after slight perturbation, classificaiton is now " + str(np.argmax(adver_pred_perturbed)))
                                adversary_found, adversary_distance, adversary_arrays = self.store_data(adversary_found, adversary_distance, adversary_arrays,
                                 execution_data, execution_label, adversarial_image_fmodel, ii, fmodel)

                            #     # print("\n Forward pass logits are : " + str(fmodel.forward(adversarial_image_fmodel[:, :, :])))
                            #     print("\n Predictions gave the label : " + str(np.argmax(fmodel.predictions(adversarial_image_fmodel[:, :, :]))))
                            #     print("\n Predictions logits are : " + str(fmodel.predictions(adversarial_image_fmodel[:, :, :])))
                            #     print("\n Original classification of " + str(execution_label))
                            else:
                                adversary_found, adversary_distance, adversary_arrays = self.store_data(adversary_found, adversary_distance, adversary_arrays,
                                    execution_data, execution_label, adversarial_image_fmodel, ii, fmodel)
                                    

                            
                    return adversary_found, adversary_distance, adversary_arrays, perturb_list


        def create_adversarial(self, execution_data, execution_label):

            adversarial_image_fmodel = self.attack_fmodel(execution_data, execution_label)
            return adversarial_image_fmodel

        def store_data(self, adversary_found, adversary_distance, adversary_arrays, execution_data, execution_label, adversarial_image_fmodel, ii, fmodel):

            if os.path.exists('adversarial_images/' + self.output_directory + '/' + self.attack_type_dir + '/') == 0:
                try:
                    os.mkdir('adversarial_images/' + self.output_directory + '/')
                except OSError:
                    pass
                try:
                    os.mkdir('adversarial_images/' + self.output_directory + '/' + self.attack_type_dir + '/')
                except OSError:
                    pass
            
            plt.imsave('adversarial_images/' + self.output_directory + '/' + 
                self.attack_type_dir + '/num' + str(ii) + '.png', adversarial_image_fmodel[:,:,0], cmap='gray')

            # print("ii is " + str(ii))
            # print("Shape is " + str(np.shape(adversarial_image_fmodel)))
            # print("Storage shape is " + str(np.shape(adversary_arrays)))
            adversary_arrays[ii, :, :] = adversarial_image_fmodel

            adversarial_image_fmodel_reshape = np.expand_dims(adversarial_image_fmodel, axis=0)
            adver_pred = fmodel.forward(adversarial_image_fmodel_reshape)

            print("The classification label following attack is " + str(np.argmax(adver_pred)) + " from an original classification of " + str(execution_label))
            # print("The classification label following attack is " + str(np.argmax(fmodel.predictions(adversarial_image_fmodel[:, :, :]))) 
            #         + " from an original classification of " + str(execution_label))
            distance, distance_name = self.distance_metric(execution_data.flatten(), adversarial_image_fmodel.flatten())
            print("The " + distance_name + " distance of the adversary is " + str(distance))
            adversary_found[ii] = 1
            adversary_distance[ii] = distance

            return adversary_found, adversary_distance, adversary_arrays

        def distance_metric(self, vector1, vector2):
            distance = scipy.spatial.distance.euclidean(vector1, vector2)
            distance_name = 'Euclidean (L-2)'
            return distance, distance_name



class blended_noise_attack(parent_attack):
        attack_type_dir = 'Blended_noise'
        #As the parent attack already uses the blended uniform noise attack by default, no changes are necessary

class pointwise_attack(parent_attack):
        attack_method = foolbox.attacks.PointwiseAttack
        foolbox_distance_metric = foolbox.distances.L0 #This is the distance metric used during optimization by FoolBox attacks
        attack_type_dir = 'Pointwise'

        #This is the distance metric used to evalute the final distances of the returned images from the original
        def distance_metric(self, vector1, vector2):
            distance = scipy.spatial.distance.hamming(vector1, vector2)
            distance_name = 'Hamming (L-0)'
            return distance, distance_name

class boundary_attack(parent_attack):
        #Overwite parent constructor for two additional attributes : num_iterations and log_every_n_steps
        #As it is overwritten, it needs to be explicitly called here
        def __init__(self, model_prediction_function,
                        model_weights,
                        var_list,
                        weights_dic,
                        biases_dic,
                        input_data,
                        input_labels,
                        input_placeholder,
                        dropout_rate_placeholder,
                        output_directory,
                        num_attack_examples=3, 
                        criterion=foolbox.criteria.Misclassification(), 
                        dropout_rate=0.0,
                        num_iterations=50,
                        log_every_n_steps=50):
                parent_attack.__init__(self, model_prediction_function,
                        model_weights,
                        var_list,
                        weights_dic,
                        biases_dic,
                        input_data,
                        input_labels,
                        input_placeholder,
                        dropout_rate_placeholder,
                        output_directory,
                        num_attack_examples, 
                        criterion, 
                        dropout_rate)
                self.num_iterations = num_iterations
                self.log_every_n_steps = log_every_n_steps
        
        attack_method = foolbox.attacks.BoundaryAttack
        attack_type_dir = 'Boundary'

        #Overwrite the execute attack method, as the boundary attack requires a specified number of iterations
        def create_adversarial(self, execution_data, execution_label):

                adversarial_image_fmodel = self.attack_fmodel(execution_data, execution_label, iterations=self.num_iterations, 
                    log_every_n_steps=self.log_every_n_steps, verbose=False)

                return adversarial_image_fmodel

class BIM_L2_attack(parent_attack):
        def __init__(self, model_prediction_function,
                        model_weights,
                        var_list,
                        weights_dic,
                        biases_dic,
                        input_data,
                        input_labels,
                        input_placeholder,
                        dropout_rate_placeholder,
                        output_directory,
                        num_attack_examples=3, 
                        criterion=foolbox.criteria.Misclassification(), 
                        dropout_rate=0.0,
                        epsilon=0.3):
                parent_attack.__init__(self, model_prediction_function,
                        model_weights,
                        var_list,
                        weights_dic,
                        biases_dic,
                        input_data,
                        input_labels,
                        input_placeholder,
                        dropout_rate_placeholder,
                        output_directory,
                        num_attack_examples, 
                        criterion, 
                        dropout_rate)
                self.epsilon = epsilon

        #Inherit the attributes of the BIM_Linfinity_attack class, then overwite the attack method
        attack_method = foolbox.attacks.L2BasicIterativeAttack
        attack_type_dir = 'BIM2'

        def create_adversarial(self, execution_data, execution_label):

                adversarial_image_fmodel = self.attack_fmodel(execution_data, execution_label, epsilon=self.epsilon)
                return adversarial_image_fmodel


class BIM_Linfinity_attack(BIM_L2_attack):
        #Inherit the attributes of the BIM_L2_attack class, then overwite the attack method and summary distance metric
        attack_method = foolbox.attacks.LinfinityBasicIterativeAttack
        foolbox_distance_metric = foolbox.distances.Linfinity
        attack_type_dir = 'BIMInf'

        def distance_metric(self, vector1, vector2):

            distance = scipy.spatial.distance.chebyshev(vector1, vector2)
            distance_name = 'Chebyshev (L-Inf)'
            return distance, distance_name






