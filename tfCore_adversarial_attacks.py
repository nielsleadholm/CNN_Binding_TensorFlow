#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import os
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
            self.dynamic_dic = attack_dic['dynamic_dic'] #Determines if e.g. a network section is ablated, or noise is added to the logits
            self.batch_size = attack_dic['batch_size']
            self.save_images = attack_dic['save_images']
            self.estimate_gradients = attack_dic['estimate_gradients']
            self.adver_model = attack_dic['adver_model']
            self.adver_checkpoint = attack_dic['adver_checkpoint']
            self.criterion = criterion #note by default this is simply foolbox's Misclassification criterion

    #Define the class attribute, attack_method, to be the Blended Uniform Noise attack by default
    attack_method = foolbox.attacks.BlendedUniformNoiseAttack 
    foolbox_distance_metric = foolbox.distances.MeanSquaredDistance
    attack_type_dir = 'Parent_*_not_advised_*_'

    def evaluate_resistance(self):

        if self.adver_model == None:
            logits, _ = self.model_prediction_function(self.input_placeholder, self.dropout_rate_placeholder, self.weights_dic, self.biases_dic, self.dynamic_dic)
            saver = tf.train.Saver(self.var_list) #Define saver object for use later when loading the model weights
   
        else:
            saver = tf.train.Saver()

        self.mk_dir()

        with tf.Session() as session:
            #Define the foolbox model 
            if self.adver_model == None: 
                print("\nEvaluating a non-adversarially trained model")
                saver.restore(session, self.model_weights) #Note when restoring weights its important not to run init on the same
                    #variables, as this will over-write the learned weights with randomly initialized ones
                fmodel = foolbox.models.TensorFlowModel(self.input_placeholder, logits, (0,1)) 

            else:
                print("\nEvaluating an adversarially trained model")
                saver.restore(session, self.adver_checkpoint)
                fmodel = foolbox.models.TensorFlowModel(self.input_placeholder, self.adver_model.pre_softmax, (0,1))

            #Wrap the model to enable estimated gradients if desired
            if self.estimate_gradients == True:
                print("\nUsing a model with *estimated* gradients.")
                estimator = foolbox.gradient_estimators.CoordinateWiseGradientEstimator(epsilon=0.01)
                fmodel = foolbox.models.ModelWithEstimatedGradients(fmodel, gradient_estimator=estimator)
                #The default CoordinateWiseGradientEstimator estimator is the same used in the Schott et al, 2019 ABS paper from ICLR

            print("\nPerforming " + self.attack_type_dir + " attack")
            print("Evaluating " + str(self.num_attack_examples) + " adversarial example(s)")

            #Arrays for storing results of the evaluation
            adversary_found = np.zeros([self.num_attack_examples]) #array of booleans that indicates if an adversary was found for a particular image
            adversary_distance = np.zeros([self.num_attack_examples])

            adversaries_array = np.zeros(np.concatenate(([self.num_attack_examples], self.input_data.shape[1:])))
            adversary_labels = []

            self.attack_specification(fmodel)

            for batch_iter in range(math.ceil(self.num_attack_examples/self.batch_size)):

                execution_batch_data = self.input_data[batch_iter*self.batch_size:min((batch_iter+1)*self.batch_size, self.num_attack_examples)]
                execution_batch_labels = np.argmax(self.input_labels[batch_iter*self.batch_size:min((batch_iter+1)*self.batch_size, self.num_attack_examples)], axis=1)

                #Carry out the attack
                adversarial_images, batch_adversary_labels = self.create_adversarial(execution_batch_data, execution_batch_labels)
                adversary_labels.extend(batch_adversary_labels)

                #Process results of the batched attack
                for example_iter in range(execution_batch_data.shape[0]):

                    if np.any(adversarial_images[example_iter] == None) or np.all(np.isnan(adversarial_images[example_iter])):
                        print("\nNo adversarial image found - attack returned None or array of NaNs\n")
                        #As in Schott, 2019 et al, the distance of an unsuccessful attack is recorded as infinity
                        adversary_distance[batch_iter*self.batch_size + example_iter] = np.inf

                    else:
                        adversary_found, adversary_distance, adversaries_array = self.store_data(adversary_found, adversary_distance, adversaries_array,
                            execution_batch_data[example_iter], execution_batch_labels[example_iter], adversarial_images[example_iter], batch_iter*self.batch_size + example_iter, fmodel)

            adversary_labels = np.asarray(adversary_labels)

            return adversary_found, adversary_distance, adversaries_array, adversary_labels

    def attack_specification(self, fmodel):
        self.attack_fmodel = self.attack_method(model=fmodel, criterion=self.criterion, distance=self.foolbox_distance_metric)

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

        adversarials = self.attack_fmodel(execution_data, execution_label, unpack=False)

        adversary_labels = np.asarray([a.adversarial_class for a in adversarials])
        adversarial_images = np.asarray([a.perturbed for a in adversarials])

        return adversarial_images, adversary_labels

    def store_data(self, adversary_found, adversary_distance, adversaries_array, execution_data, execution_label, adversarial_image, results_iter, fmodel):
        
        adversaries_array[results_iter] = adversarial_image

        #Note only 10 adversarial images are saved for a given attack to reduce memory issues
        if self.save_images == True and (results_iter<10):
            if adversarial_image.shape[2] == 3:
                image_to_png = adversarial_image
            elif adversarial_image.shape[2] == 1:
                image_to_png = np.squeeze(adversarial_image, axis=2) #Remove last dimension if saving to greyscale

            plt.imsave('adversarial_images/' + self.output_directory + '/' + 
                self.attack_type_dir + '/AttackNum' + str(results_iter) + '_Predicted' + str(np.argmax(fmodel.forward(adversarial_image[None, :, :, :]))) + 
                '_GroundTruth' + str(execution_label) + '.png', image_to_png)

        print("The classification label following attack is " + str(np.argmax(fmodel.forward(adversarial_image[None]))) + " from an original classification of " + str(execution_label))
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
    #Performs checks to ensure there are no unintended stochastic elements (e.g. due to numerical issues) in a models predictions in foolbox

    def perform_check(self):

        logits, _ = self.model_prediction_function(self.input_placeholder, self.dropout_rate_placeholder, self.weights_dic, self.biases_dic, self.dynamic_dic)
        saver = tf.train.Saver(self.var_list)

        with tf.Session() as session:
            saver.restore(session, self.model_weights) 
            fmodel = foolbox.models.TensorFlowModel(self.input_placeholder, logits, (0,1)) 

            print('Checking the models performance on multiple runs of the same images')

            for example_iter in range(self.num_attack_examples):

                execution_data = self.input_data[example_iter, :, :, :]

                logits_list = []
                labels_list = []

                #Check the same image with multiple runs
                for ii in range(10):
                    #Return the logits and label of the model
                    predicted_logits = fmodel.forward(execution_data[None,:,:,:])
                    logits_list.extend(predicted_logits)
                    
                #Check every element is equivalent to the most recent prediction
                assert np.all(logits_list == np.asarray(predicted_logits)), "***Some of the logits are changing stochastically***"

                print("No stochastic elements identified")
                

class transfer_attack_L2(parent_attack):
    #Overwrite parent constructor for two additional attributes : starting_adversaries, epsilon_step_size, and max_iterations
    def __init__(self, attack_dic,
                    starting_adversaries, 
                    epsilon_step_size=0.01,
                    max_iterations=1000):
            parent_attack.__init__(self, attack_dic)
            self.starting_adversaries = starting_adversaries
            self.epsilon_step_size = epsilon_step_size
            self.max_iterations = max_iterations

    attack_type_dir = 'Transfer_L2'
    
    #Overwrite evaluate_resistance method with one that finds minimal transfer-attack images
    def evaluate_resistance(self):

        if self.adver_model == None:
            logits, _ = self.model_prediction_function(self.input_placeholder, self.dropout_rate_placeholder, self.weights_dic, self.biases_dic, self.dynamic_dic)
            saver = tf.train.Saver(self.var_list) #Define saver object for use later when loading the model weights
   
        else:
            saver = tf.train.Saver()

        self.mk_dir()

        with tf.Session() as session:
            #Define the foolbox model 
            if self.adver_model == None: 
                print("\nEvaluating a non-adversarially trained model")
                saver.restore(session, self.model_weights) #Note when restoring weights its important not to run init on the same
                    #variables, as this will over-write the learned weights with randomly initialized ones
                fmodel = foolbox.models.TensorFlowModel(self.input_placeholder, logits, (0,1)) 

            else:
                print("\nEvaluating an adversarially trained model")
                saver.restore(session, self.adver_checkpoint)
                fmodel = foolbox.models.TensorFlowModel(self.input_placeholder, self.adver_model.pre_softmax, (0,1))

            print("\nPerforming a Transfer attack")
            print("Evaluating " + str(self.num_attack_examples) + " adversarial example(s)")

            #Arrays for storing results of the evaluation
            adversary_distance = np.zeros([4, self.num_attack_examples])

            for example_iter in range(self.num_attack_examples):

                print("Transfer attack number " + str(example_iter))
                
                #Iterate through the four different starting points for generating adversaries (two different gradient 
                # based attacks for each of the two main architecture types --> binding or not binding); the minimally-perturbed attack will be returned
                for base_method_iter in range(4):
                
                    adversary_distance = self.iterative_perturbation(fmodel, adversary_distance, example_iter, base_method_iter, unperturbed_image=self.input_data[example_iter], 
                        ground_truth_label=self.input_labels[example_iter], starting_adversary=self.starting_adversaries[base_method_iter, example_iter])
                    print("Method " + str(base_method_iter) + " distance is " + str(adversary_distance[base_method_iter, example_iter]))

            #Of all images genereated from the base attack types, select the minimally perturbed image for each example
            adversary_distance = adversary_distance.min(axis=0)

            return adversary_distance

    def iterative_perturbation(self, fmodel, adversary_distance, example_iter, base_method_iter, unperturbed_image, ground_truth_label, starting_adversary):

        epsilon = 0.0
        current_iteration = 1

        #First check if the base attack method failed on the surrogate model
        #If so, see if the target model correctly classifies it, in which case it is a failed attack, or otherwise it is a successful attack with distance 0
        if np.any(starting_adversary == None) or np.all(np.isnan(starting_adversary)):
            if (np.argmax(fmodel.forward(unperturbed_image[None])) == np.argmax(ground_truth_label)):
                print("Base attack failed, and target model correctly classified image.")
                adversary_distance[base_method_iter, example_iter] = np.inf
            else:
                print("Base attack failed, but target model misclassified image.")
                adversary_distance[base_method_iter, example_iter] = 0

        else:
            #Begin with an *unperturbed* image, as this may already be enough to fool the target model
            transfer_perturbed = unperturbed_image
            
            print("Original classification is " + str(np.argmax(fmodel.forward(transfer_perturbed[None]))))
            print("Ground truth label is " + str(np.argmax(ground_truth_label)))


            # Binary search for transfer attack as used in Schott et al; based on code provided by Jonas Rauber (Bethge Lab)
            direction = starting_adversary - unperturbed_image
            bad = 0
            good = None
            epsilon_binary = 1
            k = 10

            #Rapidly identify starting point for binary search
            for _ in range(k):
                transfer_perturbed = unperturbed_image + epsilon_binary * direction
                transfer_perturbed = np.clip(transfer_perturbed, 0, 1)
                
                print("Epsilon is " + str(epsilon_binary))
                if (np.argmax(fmodel.forward(transfer_perturbed[None])) != np.argmax(ground_truth_label)):
                    good = epsilon_binary
                    break
                else:
                    bad = epsilon_binary

                epsilon_binary *= 2

            print("After exponential binary search, the classification is " + str(np.argmax(fmodel.forward(transfer_perturbed[None]))))
 
            if np.argmax(fmodel.forward(transfer_perturbed[None])) == np.argmax(ground_truth_label):
                print("Exponential search failed")
                adversary_distance[base_method_iter, example_iter] = np.inf
                print("The distance is " + str(adversary_distance[base_method_iter, example_iter]))

            else:
                for _ in range(k):
                    epsilon_binary = (good + bad) / 2.
                    transfer_perturbed = unperturbed_image + epsilon_binary * direction
                    transfer_perturbed = np.clip(transfer_perturbed, 0, 1)

                    if (np.argmax(fmodel.forward(transfer_perturbed[None])) != np.argmax(ground_truth_label)):
                        good = epsilon_binary
                    else:
                        bad = epsilon_binary

                adversary_distance[base_method_iter, example_iter], _ = self.distance_metric(unperturbed_image.flatten(), transfer_perturbed.flatten())
                print("After standard binary search, the classification is " + str(np.argmax(fmodel.forward(transfer_perturbed[None]))))
                print("The distance is " + str(adversary_distance[base_method_iter, example_iter]))

        return adversary_distance


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
    #Inherit the attributes of the pointwise_attack_L0 class, then overwrite the attack method
    attack_method = foolbox.attacks.SaltAndPepperNoiseAttack
    attack_type_dir = 'Salt_and_Pepper'


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
    attack_method = foolbox.attacks.GradientSignAttack
    attack_type_dir = 'FGM'    
    # attack_method = foolbox.attacks.GradientAttack
    # attack_type_dir = 'FGM'

class BIM_L2_attack(parent_attack):
    attack_method = foolbox.attacks.L2BasicIterativeAttack
    attack_type_dir = 'BIM_L2'

class DeepFool_L2_attack(parent_attack):
    attack_method = foolbox.attacks.DeepFoolL2Attack
    attack_type_dir = 'DeepFool_L2'

class boundary_attack(parent_attack):
    #Overwrite parent constructor for two additional attributes : num_iterations and log_every_n_steps
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

    #Overwrite create adversarial method, as the boundary attack takes a specified number of iterations
    def create_adversarial(self, execution_data, execution_label):

        adversarials = self.attack_fmodel(execution_data, execution_label, iterations=self.num_iterations, 
            log_every_n_steps=self.log_every_n_steps, verbose=False, unpack=False)

        adversary_labels = np.asarray([a.adversarial_class for a in adversarials])
        adversarial_images = np.asarray([a.perturbed for a in adversarials])

        return adversarial_images, adversary_labels


#*** L-Inf Distance Attacks ***

class transfer_attack_LInf(transfer_attack_L2):
    attack_type_dir = 'Transfer_LInf'

    def distance_metric(self, vector1, vector2):
        distance = scipy.spatial.distance.chebyshev(vector1, vector2)
        distance_name = 'Chebyshev (L-Inf)'
        return distance, distance_name

class FGSM_attack(parent_attack):
    attack_method = foolbox.attacks.GradientSignAttack
    attack_type_dir = 'FGSM'
    foolbox_distance_metric = foolbox.distances.Linfinity

    def distance_metric(self, vector1, vector2):
        distance = scipy.spatial.distance.chebyshev(vector1, vector2)
        distance_name = 'Chebyshev (L-Inf)'
        return distance, distance_name

class BIM_Linfinity_attack(FGSM_attack):
    attack_method = foolbox.attacks.LinfinityBasicIterativeAttack
    attack_type_dir = 'BIM_LInf'

class DeepFool_LInf_attack(FGSM_attack):
    attack_method = foolbox.attacks.DeepFoolLinfinityAttack
    attack_type_dir = 'DeepFool_LInf'

class MIM_attack(FGSM_attack):
    attack_method = foolbox.attacks.MomentumIterativeAttack
    attack_type_dir = 'MIM'


