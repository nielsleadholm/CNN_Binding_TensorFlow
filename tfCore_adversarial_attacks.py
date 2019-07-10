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
        distance_metric = scipy.spatial.distance.euclidean
        attack_type_dir = 'ParentClass_attack_images'

        def evaluate_resistance(self, distance_metric=distance_metric, attack_type_dir=attack_type_dir):

                logits = self.model_prediction_function(self.input_placeholder, self.dropout_rate_placeholder, self.weights_dic, self.biases_dic)
                saver = tf.train.Saver(self.var_list) #Define saver object for use later when loading the model weights

                with tf.Session() as session:
                        saver.restore(session, self.model_weights) #Note when restoring weights its important not to run init on the same
                        #variables, as this will over-write the learned weights with randomly initialized ones
                        #Define the foolbox model
                        fmodel = foolbox.models.TensorFlowModel(self.input_placeholder, logits, (0,1)) 

                        print("Evaluating " + str(self.num_attack_examples) + " adversarial example(s)")

                        #Arrays for storing results of the evaluation
                        adversary_found = np.zeros([self.num_attack_examples]) #array of booleans that indicates if an adversary was found for a particular image
                        adversary_distance = np.zeros([self.num_attack_examples])
                        adversary_arrays = np.zeros([self.num_attack_examples, self.input_data.shape[1], self.input_data.shape[2], 1])

                        for ii in range(self.num_attack_examples):

                                execution_data = self.input_data[ii, :, :]
                                execution_label = np.argmax(self.input_labels[ii,:])

                                #Check the predicted label of the network prior to carrying out the attack isn't already incorrect
                                pre_label = np.argmax(fmodel.predictions(execution_data[:, :, :]))
                                if (pre_label != execution_label):
                                    print("The model predicted a " + str(pre_label) + " when the ground-truth label is " + str(execution_label))

                                #Carry out the attack
                                self.attack_fmodel = self.attack_method(model=fmodel, criterion=self.criterion)
                                adversarial_image_fmodel = self.__create_adversarial(execution_data, execution_label)


                                
                                #Check the output of the adversarial attack

                                if np.any(adversarial_image_fmodel == None):
                                    print("\n\n *** No adversarial image found *** \n\n")
                                elif np.argmax(fmodel.predictions(adversarial_image_fmodel[:, :, :])) == execution_label:
                                    print("\n\n *** No adversarial image found *** \n\n")
                                else:
                                    if os.path.exists('adversarial_images/' + self.output_directory + '/' + attack_type_dir + '/') == 0:
                                        os.mkdir('adversarial_images/' + self.output_directory + '/')
                                        os.mkdir('adversarial_images/' + self.output_directory + '/' + attack_type_dir + '/')

                                    plt.imsave('adversarial_images/' + self.output_directory + '/' + 
                                        attack_type_dir + '/num' + str(ii) + '.png', adversarial_image_fmodel[:,:,0], cmap='gray')
                                    
                                    adversary_arrays[ii, :, :] = adversarial_image_fmodel

                                    print("The classification label following attack is " + str(np.argmax(fmodel.predictions(adversarial_image_fmodel[:, :, :]))) 
                                            + " from an original classification of " + str(execution_label))
                                    distance = distance_metric(execution_data.flatten(), adversarial_image_fmodel.flatten())
                                    print("The " + str(distance_metric) + " distance of the adversary is " + str(distance))
                                    adversary_found[ii] = 1
                                    adversary_distance[ii] = distance
                                
                        return adversary_found, adversary_distance, adversary_arrays


        def __create_adversarial(self, execution_data, execution_label):

                adversarial_image_fmodel = self.attack_fmodel(execution_data, execution_label)
                return adversarial_image_fmodel


class blended_noise_attack(parent_attack):
        attack_type_dir = 'Blended_noise_attack_images'
        #As the parent attack already uses the blended uniform noise attack by default, no changes are necessary

# class pointwise_attack(parent_attack):
#         attack_method = foolbox.attacks.PointwiseAttack
#         distance_metric = scipy.spatial.distance.hamming
#         attack_type_dir = 'Pointwise_attack_images'

# class boundary_attack(parent_attack):
#         #Overwite parent constructor for two additional attributes : num_iterations and log_every_n_steps
#         #As it is overwritten, it needs to be explicitly called here
#         def __init__(self, model_prediction_function,
#                         model_weights,
#                         var_list,
#                         input_data,
#                         input_labels,
#                         input_placeholder,
#                         num_attack_examples=1, 
#                         criterion=foolbox.criteria.Misclassification(), 
#                         dropout_rate=0.0,
#                         num_iterations=50,
#                         log_every_n_steps=50):
#                 parent_attack.__init__(self, model_prediction_function,
#                         model_weights,
#                         var_list,
#                         input_data,
#                         input_labels,
#                         input_placeholder,
#                         criterion=foolbox.criteria.Misclassification(), 
#                         dropout_rate=0.0)
#                 self.num_iterations = num_iterations
#                 self.log_every_n_steps = log_every_n_steps
        
#         attack_method = foolbox.attacks.BoundaryAttack
#         attack_type_dir = 'Boundary_attack_images'

#         #Overwrite the execute attack method, as the boundary attack requires a specified number of iterations
#         def __create_adversarial(self, execution_data, execution_label):
#                 adversarial_image_fmodel = self.attack_fmodel(self.input_data, self.input_labels, iterations=self.num_iterations, log_every_n_steps=self.log_every_n_steps)

#                 return adversarial_image_fmodel

# class BIM_L2_attack(parent_attack):
#         def __init__(self, model_prediction_function,
#                         model_weights,
#                         var_list,
#                         input_data,
#                         input_labels,
#                         input_placeholder,
#                         num_attack_examples=1, 
#                         criterion=foolbox.criteria.Misclassification(), 
#                         dropout_rate=0.0,
#                         epsilon=0.3):
#                 parent_attack.__init__(self, model_prediction_function,
#                         model_weights,
#                         var_list,
#                         input_data,
#                         input_labels,
#                         input_placeholder,
#                         criterion=foolbox.criteria.Misclassification(), 
#                         dropout_rate=0.0)
#                 self.epsilon = epsilon

#         #Inherit the attributes of the BIM_Linfinity_attack class, then overwite the attack method
#         attack_method = foolbox.attacks.L2BasicIterativeAttack
#         attack_type_dir = 'BIM2_attack_images'

#         def __create_adversarial(self, execution_data, execution_label):
#                 adversarial_image_fmodel = self.attack_fmodel(self.input_data, self.input_labels, epsilon=self.epsilon)

#                 return adversarial_image_fmodel


# class BIM_Linfinity_attack(BIM_L2_attack):
#         #Inherit the attributes of the BIM_L2_attack class, then overwite the attack method and summary distance metric
#         attack_method = foolbox.attacks.LinfinityBasicIterativeAttack
#         distance_metric = scipy.spatial.distance.chebyshev
#         attack_type_dir = 'BIMInf_attack_images'






