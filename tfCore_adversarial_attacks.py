#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import os
import foolbox
import scipy
import matplotlib.pyplot as plt
from PIL import Image

#Utilizes the FoolBox Python library (https://github.com/bethgelab/foolbox) to implement a variety 
#of adversarial attacks against deep-learning models implimented in TensorFlow's Core API

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
                        return_distance_image=0,
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
                self.return_distance_image = return_distance_image
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

                print("\nPerforming " + self.attack_type_dir + " attack")
                print("Evaluating " + str(self.num_attack_examples) + " adversarial example(s)")

                #Arrays for storing results of the evaluation
                adversary_found = np.zeros([self.num_attack_examples]) #array of booleans that indicates if an adversary was found for a particular image
                adversary_distance = np.zeros([self.num_attack_examples])
                adversaries_array = np.zeros([self.num_attack_examples, self.input_data.shape[1], self.input_data.shape[2], 1])
                perturb_list = []

                for example_iter in range(self.num_attack_examples):

                    execution_data = self.input_data[example_iter, :, :]
                    execution_label = np.argmax(self.input_labels[example_iter,:])

                    #Check the predicted label of the network prior to carrying out the attack isn't already incorrect
                    execution_data_reshape = np.expand_dims(execution_data, axis=0)
                    pre_label = np.argmax(fmodel.forward(execution_data_reshape))
                    if (pre_label != execution_label):
                        print("The model predicted a " + str(pre_label) + " when the ground-truth label is " + str(execution_label))

                    #Carry out the attack
                    print("Beginning attack, ground truth label is " + str(execution_label))
                    self.attack_fmodel = self.attack_method(model=fmodel, criterion=self.criterion, distance=self.foolbox_distance_metric)
                    adversarial_image_fmodel = self.create_adversarial(execution_data, execution_label)
                    
                    #Check the output of the adversarial attack
                    adversarial_image_fmodel_reshape = np.expand_dims(adversarial_image_fmodel, axis=0)


                    if np.any(adversarial_image_fmodel == None):
                        print("\n\n *** No adversarial image found - attack returned None *** \n\n")


                    #If the image is still correctly classified, iteratively perturb it by increasing the adversarial mask until the image is misclassified
                    elif np.argmax(fmodel.forward(adversarial_image_fmodel_reshape)) == execution_label:
                        print("\n *** The model correctly predicted " + str(np.argmax(fmodel.forward(adversarial_image_fmodel_reshape))) + " with a ground truth of " + str(execution_label))
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
                        adversarial_image_fmodel = adversarial_image_perturbed[0, :, : , :] #update the adversarial image to reflect the genuinely misclassified image
                        adversary_found, adversary_distance, adversaries_array = self.store_data(adversary_found, adversary_distance, adversaries_array,
                         execution_data, execution_label, adversarial_image_fmodel, example_iter, fmodel)

                    else:
                        adversary_found, adversary_distance, adversaries_array = self.store_data(adversary_found, adversary_distance, adversaries_array,
                            execution_data, execution_label, adversarial_image_fmodel, example_iter, fmodel)
                
                self.store_transfer_attack_data(adversaries_array, adversary_distance)

                return adversary_found, adversary_distance, adversaries_array, perturb_list


        def create_adversarial(self, execution_data, execution_label):

            adversarial_image_fmodel = self.attack_fmodel(execution_data, execution_label)
            return adversarial_image_fmodel

        def store_data(self, adversary_found, adversary_distance, adversaries_array, execution_data, execution_label, adversarial_image_fmodel, example_iter, fmodel):

            if os.path.exists('adversarial_images/' + self.output_directory + '/' + self.attack_type_dir + '/') == 0:
                try:
                    os.mkdir('adversarial_images/' + self.output_directory + '/')
                except OSError:
                    pass
                try:
                    os.mkdir('adversarial_images/' + self.output_directory + '/' + self.attack_type_dir + '/')
                except OSError:
                    pass
            
            adversaries_array[example_iter, :, :] = adversarial_image_fmodel

            adversarial_image_fmodel_reshape = np.expand_dims(adversarial_image_fmodel, axis=0)
            adver_pred = fmodel.forward(adversarial_image_fmodel_reshape)

            plt.imsave('adversarial_images/' + self.output_directory + '/' + 
                self.attack_type_dir + '/AttackNum' + str(example_iter) + '_Predicted' + str(np.argmax(adver_pred)) + 
                '_GroundTruth' + str(execution_label) + '.png', adversarial_image_fmodel[:,:,0], cmap='gray')

            print("The classification label following attack is " + str(np.argmax(adver_pred)) + " from an original classification of " + str(execution_label))
            distance, distance_name = self.distance_metric(execution_data.flatten(), adversarial_image_fmodel.flatten())
            print("The " + distance_name + " distance of the adversary is " + str(distance))
            adversary_found[example_iter] = 1
            adversary_distance[example_iter] = distance

            #If the user would like an example image of a specified distance from the original (to e.g. show what a typical mean distance corresponds to)
            # then they should pass a non-zero value giving the desired distance
            if self.return_distance_image != 0:
                #Checks if the currently generated image is within a certain tolerance for the desired distance
                upper_bound = self.return_distance_image*1.05
                lower_bound = self.return_distance_image*0.95
                if (distance < upper_bound) and (distance > lower_bound):
                    print("\n *** An image has been found with the desired distance...")
                    plt.imsave('adversarial_images/' + self.output_directory + '/' + 
                            self.attack_type_dir + '/DesiredImage_AttackNum' + str(example_iter) + '_Predicted' + str(np.argmax(adver_pred)) + 
                            '_GroundTruth' + str(execution_label) + distance_name + 'DistanceOf' + str(distance) + '.png', adversarial_image_fmodel[:,:,0], cmap='gray')


            return adversary_found, adversary_distance, adversaries_array

        def store_transfer_attack_data(self, adversaries_array, adversary_distance):
           #Note the save directory is slightly different for the Transfer Attack data
            transfer_output_directory = ((self.output_directory).split('_'))[0]
            
            if os.path.exists('adversarial_images/transfer_images/' + transfer_output_directory + '/' + self.attack_type_dir + '/') == 0:
                try: 
                    os.mkdir('adversarial_images/transfer_images')
                except OSError:
                    pass
                try:
                    os.mkdir('adversarial_images/transfer_images/' + transfer_output_directory + '/')
                except OSError:
                    pass
                try:
                    os.mkdir('adversarial_images/transfer_images/' + transfer_output_directory + '/' + self.attack_type_dir + '/')
                except OSError:
                    pass

            #Save the generated adversaries and ground-truth labels for use later in transfer attacks
            np.savetxt('adversarial_images/transfer_images/' + transfer_output_directory + '/' + self.attack_type_dir + '/adversaries_array.csv', 
                np.reshape(adversaries_array, [adversaries_array.shape[0], adversaries_array.shape[1]*adversaries_array.shape[2]]), 
                delimiter=',')
            np.savetxt('adversarial_images/transfer_images/' + transfer_output_directory + '/' + self.attack_type_dir + '/ground_truth_labels.csv', 
                np.argmax(self.input_labels[0:self.num_attack_examples,:], axis=1),
                delimiter=',')
            #Save the distances of all the images generated
            np.savetxt('adversarial_images/transfer_images/' + transfer_output_directory + '/' + self.attack_type_dir + '/adversary_distances.csv', 
                adversary_distance, 
                delimiter=',')

        def distance_metric(self, vector1, vector2):
            distance = scipy.spatial.distance.euclidean(vector1, vector2)
            distance_name = 'Euclidean (L-2)'
            return distance, distance_name

class check_stochasticity(parent_attack):
    #Performs checks to ensure there are no unintended stochastic elements (e.g. due to floating point changes) in a models predictions in foolbox

    def perform_check(self):

            logits = self.model_prediction_function(self.input_placeholder, self.dropout_rate_placeholder, self.weights_dic, self.biases_dic)
            saver = tf.train.Saver(self.var_list) #Define saver object for use later when loading the model weights

            with tf.Session() as session:
                saver.restore(session, self.model_weights) #Note when restoring weights its important not to run init on the same
                #variables, as this will over-write the learned weights with randomly initialized ones
                #Define the foolbox model
                fmodel = foolbox.models.TensorFlowModel(self.input_placeholder, logits, (0,1)) 

                print('Checking the models performance on multiple runs of the same images')

                for example_iter in range(self.num_attack_examples):

                    execution_data = self.input_data[example_iter, :, :]
                    execution_data_reshape = np.expand_dims(execution_data, axis=0)

                    logits_list = []
                    labels_list = []

                    #Check the same image with multiple runs
                    for ii in range(10):
                        #Return the logits and label of the model
                        predicted_logits = fmodel.forward(execution_data_reshape)
                        predicted_label = np.argmax(predicted_logits)
                        logits_list.append(predicted_logits)
                        labels_list.append(predicted_label)
                        
                    #Check every element is equivalent to the most recent prediction
                    assert np.all(logits_list == np.asarray(predicted_logits)), "***Some of the logits are changing stochastically!***"
                    assert np.all(labels_list == np.asarray(predicted_label)), "***Some of the labels are changing stochastically!***"

                    print("No stochastic elements identified")
                    

class transfer_attack(parent_attack):
        #Overwite parent constructor for three additional attributes:
        # model_under_attack is the model architecture that the transfer attack is being applied to
        # model_adversarial_gen and attack_type_dir is the model and attack-type that were used to actually generate the adversarial examples
        # distance_range determines if an upper and/or lower bound on the distance of the adversaries will be set before conducting the transfer attack;
        # note that for this, the distance metric will have been defined when the adversary_distances array was first made
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
                        return_distance_image=0,
                        criterion=foolbox.criteria.Misclassification(), 
                        dropout_rate=0.0,
                        model_under_attack=None,
                        model_adversarial_gen=None,
                        attack_type_dir=None,
                        distance_range=None):
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
                        return_distance_image,
                        criterion, 
                        dropout_rate)
                self.model_under_attack = model_under_attack
                self.model_adversarial_gen = model_adversarial_gen
                self.attack_type_dir = attack_type_dir
                self.distance_range = distance_range


        def evaluate_resistance(self):

            logits = self.model_prediction_function(self.input_placeholder, self.dropout_rate_placeholder, self.weights_dic, self.biases_dic)
            saver = tf.train.Saver(self.var_list) #Define saver object for use later when loading the model weights

            with tf.Session() as session:
            
                saver.restore(session, self.model_weights)
                fmodel = foolbox.models.TensorFlowModel(self.input_placeholder, logits, (0,1)) 
            
                print("\nPerforming transfer attack with " + self.attack_type_dir + 
                    " attack examples generated on a " + self.model_adversarial_gen + " model")
                print("Evaluating " + str(self.num_attack_examples) + " adversarial example(s)")

                assert self.model_adversarial_gen != self.model_under_attack, "*** Trying to use transfer attack with the model's own adversaries ***"

                #Arrays for storing results of the evaluation
                adversary_found = []

                #Note that attack_type_dir now determines the type of adversarial examples (i.e. how they were generated) that shoulbe *loaded*
                transfer_images_file = ('adversarial_images/transfer_images/' + self.model_adversarial_gen + '/' + 
                    self.attack_type_dir + '/adversaries_array.csv')
                ground_truth_file = ('adversarial_images/transfer_images/' + self.model_adversarial_gen + '/' + 
                    self.attack_type_dir + '/ground_truth_labels.csv')
                distances_file = ('adversarial_images/transfer_images/' + self.model_adversarial_gen + '/' + 
                    self.attack_type_dir + '/adversary_distances.csv')

                #Load transfer_attack_images and reshape
                transfer_images = np.genfromtxt(transfer_images_file, delimiter=',')
                transfer_images = np.reshape(transfer_images, [transfer_images.shape[0], 28, 28, 1])

                ground_truth_labels = np.genfromtxt(ground_truth_file, delimiter=',')

                #Filter the transfer images by the upper and lower bounds of perturbation, if set
                satisfy_bounds = np.ones(self.num_attack_examples) #Create a default array of all true if distance range not specified
                if self.distance_range != None:
                    print("Using a lower and upper bound on transfer attack perturbations of " + str(self.distance_range[0]) + " and " + str(self.distance_range[1]))
                    image_distances = np.genfromtxt(distances_file, delimiter=',')
                    satisfy_bounds = (image_distances > self.distance_range[0]) & (image_distances < self.distance_range[1])


                #Check that the attack isn't trying to be performed on more images than are available
                assert self.num_attack_examples <= transfer_images.shape[0], "*** Requested attack with more examples than are available in transfer attack dataset ***"

                for example_iter in range(self.num_attack_examples):

                    if satisfy_bounds[example_iter] == 1:
                        execution_data = transfer_images[example_iter, :, :, :]
                        execution_data = np.expand_dims(execution_data, axis=0) #Reshape for 'forward' method used later
                        execution_label = ground_truth_labels[example_iter]

                        adver_pred = fmodel.forward(execution_data)

                        if np.argmax(adver_pred) == execution_label:
                            print("The model correctly predicted the ground truth label of " + str(execution_label))
                            adversary_found.append(0)

                        else:
                            print("The classification label following attack is " + str(np.argmax(adver_pred)) + " from a ground-truth classification of " + str(execution_label))
                            adversary_found.append(1)
                
                adversary_found = np.asarray(adversary_found)
                
                return adversary_found


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
                        return_distance_image=0,
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
                        return_distance_image,
                        criterion, 
                        dropout_rate)
                self.num_iterations = num_iterations
                self.log_every_n_steps = log_every_n_steps
        
        attack_method = foolbox.attacks.BoundaryAttack
        attack_type_dir = 'Boundary'

        #Overwrite the execute attack method, as the boundary attack requires a specified number of iterations
        def create_adversarial(self, execution_data, execution_label):

                adversarial_image_fmodel = self.attack_fmodel(execution_data, execution_label, iterations=self.num_iterations, 
                    log_every_n_steps=self.log_every_n_steps, verbose=True)

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
                        return_distance_image=0,
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
                        return_distance_image,
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


