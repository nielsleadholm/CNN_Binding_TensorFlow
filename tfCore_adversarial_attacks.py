#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import os
import copy
import math
import foolbox
import scipy
import matplotlib.pyplot as plt
from PIL import Image

#Utilizes the FoolBox Python library (https://github.com/bethgelab/foolbox) to implement a variety 
#of adversarial attacks against deep-learning models implimented in TensorFlow's Core API

class parent_attack:
    def __init__(self, attack_dic, 
                    criterion=foolbox.criteria.Misclassification()):
            self.model_prediction_function = attack_dic['model_prediction_function']
            self.model_weights = attack_dic['model_weights']
            self.var_list = attack_dic['var_list']
            self.weights_dic = attack_dic['weights_dic']
            self.biases_dic = attack_dic['biases_dic']
            self.input_data = attack_dic['input_data']
            self.input_labels = attack_dic['input_labels']
            self.input_placeholder = attack_dic['input_placeholder']
            self.dropout_rate_placeholder = attack_dic['dropout_rate_placeholder']
            self.output_directory = attack_dic['output_directory']
            self.num_attack_examples = attack_dic['num_attack_examples']
            self.dynamic_var = attack_dic['dynamic_var'] #Determines if e.g. a network section is ablated, or noise is added to the logits
            self.batch_size = attack_dic['batch_size']
            self.save_images = attack_dic['save_images']
            self.criterion = criterion #note by default this is simply foolbox's Misclassification criterion

    #Define the class attribute, attack_method, to be the Blended Uniform Noise attack by default
    attack_method = foolbox.attacks.BlendedUniformNoiseAttack 
    foolbox_distance_metric = foolbox.distances.MeanSquaredDistance
    attack_type_dir = 'Parent_*_not_advised_*_'

    def evaluate_resistance(self):

        logits, _, _, _ = self.model_prediction_function(self.input_placeholder, self.dropout_rate_placeholder, self.weights_dic, self.biases_dic, self.dynamic_var)
        saver = tf.train.Saver(self.var_list) #Define saver object for use later when loading the model weights
        self.mk_dir()

        with tf.Session() as session:
            saver.restore(session, self.model_weights) #Note when restoring weights its important not to run init on the same
            #variables, as this will over-write the learned weights with randomly initialized ones

            #Define the foolbox model
            fmodel = foolbox.models.TensorFlowModel(self.input_placeholder, logits, (0,1)) 

            print("\nPerforming " + self.attack_type_dir + " attack")
            print("Evaluating " + str(self.num_attack_examples) + " adversarial example(s)")

            #Arrays for storing results of the evaluation
            adversary_found = np.zeros([self.num_attack_examples]) #array of booleans that indicates if an adversary was found for a particular image
            adversary_distance = np.zeros([self.num_attack_examples])
            adversaries_array = np.zeros([self.num_attack_examples, self.input_data.shape[1], self.input_data.shape[2], self.input_data.shape[3]])
            perturb_list = []

            self.attack_fmodel = self.attack_method(model=fmodel, criterion=self.criterion, distance=self.foolbox_distance_metric)

            for batch_iter in range(math.ceil(self.num_attack_examples/self.batch_size)):

                execution_batch_data = self.input_data[batch_iter*self.batch_size:min((batch_iter+1)*self.batch_size, self.num_attack_examples), :, :, :]
                execution_batch_labels = np.argmax(self.input_labels[batch_iter*self.batch_size:min((batch_iter+1)*self.batch_size, self.num_attack_examples), :], axis=1)

                #Carry out the attack
                adversarial_images = self.create_adversarial(execution_batch_data, execution_batch_labels)

                #Process results of the batched attack
                for example_iter in range(execution_batch_data.shape[0]):

                    if np.any(adversarial_images[example_iter] == None) or np.all(np.isnan(adversarial_images[example_iter])):
                        print("\nNo adversarial image found - attack returned None or array of NaNs\n")
                        #As in Schott, 2019 et al, the distance of an unsuccessful attack is recorded as infinity
                        adversary_distance[batch_iter*self.batch_size + example_iter] = np.inf

                    #Ocasionally the image returned by FoolBox will be correctly classified by the network when passed through again
                    #If the image is still correctly classified, iteratively perturb it by increasing the adversarial mask until the image is misclassified
                    # elif np.argmax(fmodel.forward(adversarial_images[None, example_iter])) == execution_batch_labels[example_iter]:
                    #     print("\n *The model correctly predicted " + str(np.argmax(fmodel.forward(adversarial_images[None, example_iter]))) + " with a ground truth of " + str(execution_batch_labels[example_iter]))
                    #     print("Iteratively enhancing perturbation until misclassified again...")
                    #     multiplier = 1.0001 #Initialize the multiplier
                    #     #Evaluate with the initial perturbation
                    #     adversarial_image_perturbed = execution_batch_data[example_iter] + multiplier*(adversarial_images[example_iter] - execution_batch_data[example_iter])
                    #     #Clip the pixel values to lie within the given range
                    #     adversarial_image_perturbed = np.clip(adversarial_image_perturbed, 0, 1)
                    #     adver_pred_perturbed = fmodel.forward(adversarial_image_perturbed[None, :, :, :])

                    #     while np.argmax(adver_pred_perturbed) == execution_batch_labels[example_iter]:
                    #         multiplier += 0.001
                    #         adversarial_image_perturbed = execution_batch_data[example_iter] + multiplier*(adversarial_images[example_iter] - execution_batch_data[example_iter])
                    #         adversarial_image_perturbed = np.clip(adversarial_image_perturbed, 0, 1)
                    #         adver_pred_perturbed = fmodel.forward(adversarial_image_perturbed[None, :, :, :])

                    #         #Check if all the values of the image are maximally perturbed, in which case break
                    #         if np.all((adversarial_image_perturbed==0) | (adversarial_image_perturbed==1))==True:
                    #             print("\n\n***Maximally perturbed returned image, but still not misclassified\n\n")
                    #             print(adversarial_image_perturbed)
                    #             adversary_distance[batch_iter*self.batch_size + example_iter] = np.inf
                    #             break

                    #     #Check it wasn't a maximally perturbed image:
                    #     if np.all((adversarial_image_perturbed==0) | (adversarial_image_perturbed==1))==False:
                    #         print("Perturbed classification is " + str(np.argmax(adver_pred_perturbed)) + " following additional perturbation of " +str(multiplier))

                    #         adversarial_images[example_iter] = adversarial_image_perturbed[:, :, :] #update the adversarial image to reflect the genuinely misclassified image
                    #         adversary_found, adversary_distance, adversaries_array = self.store_data(adversary_found, adversary_distance, adversaries_array,
                    #          execution_batch_data[example_iter], execution_batch_labels[example_iter], adversarial_images[example_iter], batch_iter*self.batch_size + example_iter, fmodel)

                    else:
                        adversary_found, adversary_distance, adversaries_array = self.store_data(adversary_found, adversary_distance, adversaries_array,
                            execution_batch_data[example_iter], execution_batch_labels[example_iter], adversarial_images[example_iter], batch_iter*self.batch_size + example_iter, fmodel)

            return adversary_found, adversary_distance, adversaries_array, perturb_list

    #Make the attack directory for storing results
    def mk_dir(self):
        if os.path.exists('adversarial_images/' + self.output_directory + '/' + self.attack_type_dir + '/') == 0:
            try:
                os.mkdir('adversarial_images/' + self.output_directory + '/')
            except OSError:
                pass
            try:
                os.mkdir('adversarial_images/' + self.output_directory + '/' + self.attack_type_dir + '/')
            except OSError:
                pass

    def create_adversarial(self, execution_data, execution_label):

        adversarial_images = self.attack_fmodel(execution_data, execution_label)
        return adversarial_images

    def store_data(self, adversary_found, adversary_distance, adversaries_array, execution_data, execution_label, adversarial_image, results_iter, fmodel):
        
        adversaries_array[results_iter, :, :, :] = adversarial_image

        if self.save_images == True:
            if adversarial_image.shape[2] == 3:
                image_to_png = adversarial_image
                print(image_to_png.shape)
            elif adversarial_image.shape[2] == 1:
                #cmap=plt.cm.gray
                image_to_png = np.squeeze(adversarial_image, axis=2) #Remove last dimension if saving to greyscale
                print(image_to_png.shape)

            plt.imsave('adversarial_images/' + self.output_directory + '/' + 
                self.attack_type_dir + '/AttackNum' + str(results_iter) + '_Predicted' + str(np.argmax(fmodel.forward(adversarial_image[None, :, :, :]))) + 
                '_GroundTruth' + str(execution_label) + '.png', image_to_png)

        print("The classification label following attack is " + str(np.argmax(fmodel.forward(adversarial_image[None, :, :, :]))) + " from an original classification of " + str(execution_label))
        distance, distance_name = self.distance_metric(execution_data.flatten(), adversarial_image.flatten())
        print("The " + distance_name + " distance of the adversary is " + str(distance))
        adversary_found[results_iter] = 1
        adversary_distance[results_iter] = distance

        return adversary_found, adversary_distance, adversaries_array

    def distance_metric(self, vector1, vector2):
        distance = scipy.spatial.distance.euclidean(vector1, vector2)
        distance_name = 'Euclidean (L-2)'
        return distance, distance_name

class check_stochasticity(parent_attack):
    #Performs checks to ensure there are no unintended stochastic elements (e.g. due to floating point changes) in a models predictions in foolbox

    def perform_check(self):

            logits, _, _, _ = self.model_prediction_function(self.input_placeholder, self.dropout_rate_placeholder, self.weights_dic, self.biases_dic)
            saver = tf.train.Saver(self.var_list) #Define saver object for use later when loading the model weights

            with tf.Session() as session:
                saver.restore(session, self.model_weights) #Note when restoring weights its important not to run init on the same
                #variables, as this will over-write the learned weights with randomly initialized ones
                #Define the foolbox model
                fmodel = foolbox.models.TensorFlowModel(self.input_placeholder, logits, (0,1)) 

                print('Checking the models performance on multiple runs of the same images')

                for example_iter in range(self.num_attack_examples):

                    execution_data = self.input_data[example_iter, :, :]
                    execution_data_reshape = execution_data #np.expand_dims(execution_data, axis=0)

                    logits_list = []
                    labels_list = []

                    #Check the same image with multiple runs
                    for ii in range(10):
                        #Return the logits and label of the model
                        predicted_logits = fmodel.forward(execution_data_reshape)
                        print(predicted_logits)
                        # print(predicted_logits[0][0])
                        # print(np.dtype(predicted_logits[0][0]))
                        predicted_label = np.argmax(predicted_logits)
                        logits_list.append(predicted_logits)
                        labels_list.append(predicted_label)
                        
                    #Check every element is equivalent to the most recent prediction
                    assert np.all(logits_list == np.asarray(predicted_logits)), "***Some of the logits are changing stochastically!***"
                    assert np.all(labels_list == np.asarray(predicted_label)), "***Some of the labels are changing stochastically!***"

                    print("No stochastic elements identified")
                    

# class transfer_attack(parent_attack):
#     #Overwrite parent constructor for three additional attributes:
#     # model_under_attack is the model architecture that the transfer attack is being applied to
#     # model_adversarial_gen and attack_type_dir is the model and attack-type that were used to actually generate the adversarial examples
#     # distance_range determines if an upper and/or lower bound on the distance of the adversaries will be set before conducting the transfer attack;
#     # note that for this, the distance metric will have been defined when the adversary_distances array was first made
#     def __init__(self, attack_dic, 
#                     return_distance_image=0,
#                     criterion=foolbox.criteria.Misclassification(), 
#                     model_under_attack=None,
#                     model_adversarial_gen=None,
#                     attack_type_dir=None,
#                     transfer_distance_metric=None):
#             parent_attack.__init__(self, attack_dic, 
#                     return_distance_image,
#                     criterion)
#             self.model_under_attack = model_under_attack
#             self.model_adversarial_gen = model_adversarial_gen
#             self.attack_type_dir = attack_type_dir
#             self.transfer_distance_metric = transfer_distance_metric


#     def evaluate_resistance(self):

#         logits, _, _, _ = self.model_prediction_function(self.input_placeholder, self.dropout_rate_placeholder, self.weights_dic, self.biases_dic)
#         saver = tf.train.Saver(self.var_list) #Define saver object for use later when loading the model weights

#         with tf.Session() as session:
        
#             saver.restore(session, self.model_weights)
#             fmodel = foolbox.models.TensorFlowModel(self.input_placeholder, logits, (0,1)) 
        
#             #Iterate through all the models that are not of the kind being attacked
#             rootdir = 'adversarial_images/transfer_images/'
#             _, model_dirs, _ = next(os.walk(rootdir))
#             for key in model_dirs:
#                 if not key.endswith(self.model_under_attack):
#                     print(key)

#             #Iterate through all the attack types of a given distance metric


#             print("\nPerforming " + self.transfer_distance_metric + " transfer attack with " + self.attack_type_dir + 
#                 " attack examples generated on a " + self.model_adversarial_gen + " model")
#             print("Evaluating " + str(self.num_attack_examples) + " adversarial example(s)")

#             assert self.model_adversarial_gen != self.model_under_attack, "*** Trying to use transfer attack with the model's own adversaries ***"

#             #Arrays for storing results of the evaluation
#             adversary_found, adversary_distance = []
#             adversary_distance = [] 

#             #Note that attack_type_dir now determines the type of adversarial examples (i.e. how they were generated) that shoulbe *loaded*
#             transfer_images_file = ('adversarial_images/transfer_images/' + self.model_adversarial_gen + '/' + 
#                 self.attack_type_dir + '/adversaries_array.npy')
#             ground_truth_file = ('adversarial_images/transfer_images/' + self.model_adversarial_gen + '/' + 
#                 self.attack_type_dir + '/ground_truth_labels.csv')
#             distances_file = ('adversarial_images/transfer_images/' + self.model_adversarial_gen + '/' + 
#                 self.attack_type_dir + '/adversary_distances.csv')

#             #Load transfer_attack_images and reshape
#             transfer_images = np.load(transfer_images_file, mmap_mode='r')
#             transfer_images = np.reshape(transfer_images, [transfer_images.shape[0], 28, 28, 1])
#             image_distances = np.genfromtxt(distances_file, delimiter=',')

#             ground_truth_labels = np.genfromtxt(ground_truth_file, delimiter=',')

#             #Check that the attack isn't trying to be performed on more images than are available
#             assert self.num_attack_examples <= transfer_images.shape[0], "*** Requested attack with more examples than are available in transfer attack dataset ***"

#             for example_iter in range(self.num_attack_examples):

#                 execution_data = transfer_images[example_iter, :, :, :]
#                 execution_data = execution_data #np.expand_dims(execution_data, axis=0) #Reshape for 'forward' method used later
#                 execution_label = ground_truth_labels[example_iter]

#                 adver_pred = fmodel.forward(execution_data)

#                 if np.argmax(adver_pred) == execution_label:
#                     print("The model correctly predicted the ground truth label of " + str(execution_label))
#                     adversary_found.append(0)

#                 else:
#                     print("The classification label following attack is " + str(np.argmax(adver_pred)) + " from a ground-truth classification of " + str(execution_label))
#                     adversary_found.append(1)
#                     #Take absolute value of image_distance to account for images that didn't actually fool the original model, and are hence stored negative
#                     adversary_distance.append(abs(image_distances[example_iter]))
            
#             adversary_found = np.asarray(adversary_found)
            
#             return adversary_found, adversary_distance



#*** L-0 Distance Attacks ***

class pointwise_attack_L0(parent_attack):
    attack_method = foolbox.attacks.PointwiseAttack
    foolbox_distance_metric = foolbox.distances.L0 #This is the distance metric used during optimization by FoolBox attacks
    attack_type_dir = 'Pointwise_L0'

    #This is the distance metric used to evalute the final distances of the returned images from the original
    def distance_metric(self, vector1, vector2):
        distance = scipy.spatial.distance.hamming(vector1, vector2)*len(vector1)
        distance_name = 'Hamming (L-0)'
        return distance, distance_name

class salt_pepper_attack(pointwise_attack_L0):
    #Inherit the attributes of the pointwise_attack_L0 class, then overwrite the attack method and summary distance metric
    attack_method = foolbox.attacks.SaltAndPepperNoiseAttack
    attack_type_dir = 'Salt_and_Pepper'



#*** L-Inf Distance Attacks ***

class FGSM_attack(parent_attack):
    attack_method = foolbox.attacks.GradientSignAttack
    attack_type_dir = 'FGSM'
    foolbox_distance_metric = foolbox.distances.Linfinity

    def distance_metric(self, vector1, vector2):
        distance = scipy.spatial.distance.chebyshev(vector1, vector2)
        distance_name = 'Chebyshev (L-Inf)'
        return distance, distance_name

class BIM_Linfinity_attack(FGSM_attack):
    #Inherit the attributes of the FGSM_attack class, then overwrite the attack method and summary distance metric
    attack_method = foolbox.attacks.LinfinityBasicIterativeAttack
    attack_type_dir = 'BIM_LInf'

class DeepFool_LInf_attack(FGSM_attack):
    attack_method = foolbox.attacks.DeepFoolLinfinityAttack
    attack_type_dir = 'DeepFool_LInf'

class MIM_attack(FGSM_attack):
    attack_method = foolbox.attacks.MomentumIterativeAttack
    attack_type_dir = 'MIM'



#*** L-2 Distance Attacks ***

class blended_noise_attack(parent_attack):
    attack_type_dir = 'Blended_noise'
    #As the parent attack already uses the blended uniform noise attack by default, no changes are necessary

class gaussian_noise_attack(parent_attack):
    attack_method = foolbox.attacks.AdditiveGaussianNoiseAttack
    attack_type_dir = 'Gaussian_noise'

class pointwise_attack_L2(parent_attack):
    #Note this version of the point-wise attack inherits the L2 distance metric from the parent class
    attack_method = foolbox.attacks.PointwiseAttack
    attack_type_dir = 'Pointwise_L2'

class FGM_attack(parent_attack):
    attack_method = foolbox.attacks.GradientAttack
    attack_type_dir = 'FGM'

class BIM_L2_attack(parent_attack):
    attack_method = foolbox.attacks.L2BasicIterativeAttack
    attack_type_dir = 'BIM_L2'

class DeepFool_L2_attack(parent_attack):
    attack_method = foolbox.attacks.DeepFoolL2Attack
    attack_type_dir = 'DeepFool_L2'

class local_search_attack(parent_attack):
    attack_method = foolbox.attacks.LocalSearchAttack
    attack_type_dir = 'LocalSearch'

class gaussian_blur_attack(parent_attack):
    attack_method = foolbox.attacks.GaussianBlurAttack
    attack_type_dir = 'GaussianBlur'

class spatial_attack(parent_attack):
    attack_method = foolbox.attacks.SpatialAttack
    attack_type_dir = 'Spatial'

class boundary_attack(parent_attack):
    #Overwite parent constructor for two additional attributes : num_iterations and log_every_n_steps
    #As it is overwritten, it needs to be explicitly called here
    def __init__(self, attack_dic,
                    criterion=foolbox.criteria.Misclassification(), 
                    num_iterations=50,
                    log_every_n_steps=50):
            parent_attack.__init__(self, attack_dic,
                    criterion)
            self.num_iterations = num_iterations
            self.log_every_n_steps = log_every_n_steps
    
    attack_method = foolbox.attacks.BoundaryAttack
    attack_type_dir = 'Boundary'

    #Overwrite the execute attack method, as the boundary attack requires a specified number of iterations
    def create_adversarial(self, execution_data, execution_label):

            adversarial_images = self.attack_fmodel(execution_data, execution_label, iterations=self.num_iterations, 
                log_every_n_steps=self.log_every_n_steps, verbose=False)

            return adversarial_images
