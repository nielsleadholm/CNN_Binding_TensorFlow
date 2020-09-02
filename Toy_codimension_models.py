#!/usr/bin/env python3

import numpy as np
import math
import os
import pandas as pd
import yaml
import tensorflow as tf
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from bokeh.plotting import output_notebook, figure, show
from bokeh.models import ColumnDataSource, Segment
from bokeh.io import export_png, save
from Systematic_resistance_evaluation import carry_out_attacks

#Build and evaluate an MLP on a toy-dataset to examine the effect of co-dimension on adversarial robustness
#All boundary visualization code is from Data Visualizations tutorial by Gaurav-Kaushik (https://github.com/gaurav-kaushik/Data-Visualizations-Medium)

# Suppress unecessary logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

def randomize_order(x, y):
    #Randomize data-sample orders
    for_shuffle = list(zip(x, y))
    np.random.shuffle(for_shuffle)
    x, y = zip(*for_shuffle)

    #Get y back to an np array
    y = np.array(list(map(list, y)))

    return x,y

def generate_toy_data(data_size, additional_features_dimension, training_data_bool):

    #Data-set is multi-dimensional Gaussian in which two dimensions can be easily linearly seperated, as well as
    # an additional number of dimensions that are not as easily seperable, but which still carry some class information

    manifold_noise = 0.15

    #Note the difference in means
    base_features_zero_class = np.random.normal(0, scale=manifold_noise, size=(int(data_size/2),2)) #The base two features
    base_features_one_class = np.random.normal(1, scale=manifold_noise, size=(int(data_size/2),2))

    #Note the change in means and the use of additional_features_dimension
    imperfect_features_zero_class = np.random.normal(0, scale=manifold_noise, size=(int(data_size/2),additional_features_dimension))
    imperfect_features_one_class = np.random.normal(0.65, scale=manifold_noise, size=(int(data_size/2),additional_features_dimension))

    zero_class_data = np.concatenate((base_features_zero_class, imperfect_features_zero_class), axis=1)
    one_class_data = np.concatenate((base_features_one_class, imperfect_features_one_class), axis=1)

    x_data = np.concatenate((zero_class_data, one_class_data), axis=0)

    one_hot_labels = np.concatenate(
        (np.concatenate((np.ones(shape=(int(data_size/2),1)), np.zeros(shape=(int(data_size/2),1))), axis=1),
        np.concatenate((np.zeros(shape=(int(data_size/2),1)), np.ones(shape=(int(data_size/2),1))), axis=1)), 
        axis=0)

    x_data, one_hot_labels = randomize_order(x=x_data, y=one_hot_labels)

    return x_data, one_hot_labels


def toy_initializer(network_iter, model_params):
    
    tf.reset_default_graph()

    input_dim = 2+model_params['additional_features_dimension']+model_params['additional_zero_dimensions']

    x_placeholder = tf.compat.v1.placeholder(tf.float32, [None, input_dim])
    y_placeholder = tf.compat.v1.placeholder(tf.int32, [None, 2])
    dropout_rate_placeholder = tf.compat.v1.placeholder(tf.float32) #note no dropout is used but it is an expected argument
        # in some of the functions used

    initializer = tf.contrib.layers.variance_scaling_initializer()

    with tf.name_scope('Network_' + str(network_iter)):
        weights = {
            'w1' : tf.compat.v1.get_variable('w1', shape=(input_dim,8), initializer=initializer),
            'w2' : tf.compat.v1.get_variable('w2', shape=(8,2), initializer=initializer)
        }

        biases = {
            'b1' : tf.compat.v1.get_variable('b1', shape=(8), initializer=initializer),
            'b2' : tf.compat.v1.get_variable('b2', shape=(2), initializer=initializer)
        }

    var_list = [weights['w1'], weights['w2'], biases['b1'], biases['b2']]

    return x_placeholder, y_placeholder, dropout_rate_placeholder, weights, biases, var_list

def MLP_predictions(x_input, dropout_rate_placeholder, weights, biases, dynamic_dic):

    layer_1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(tf.dtypes.cast(x_input, dtype=tf.float32), weights['w1']), biases['b1']))

    logits = tf.nn.bias_add(tf.matmul(layer_1, weights['w2']), biases['b2'])

    scalar_dic = {} #Used in CNN_module but not required here

    return logits, scalar_dic

def train_toy(pred_function, x_placeholder, y_placeholder, dropout_rate_placeholder, training_data, training_labels, 
        testing_data, testing_labels, mesh, weights, biases, var_list, model_params, network_iter):
    
    predictions, _ = pred_function(x_placeholder, dropout_rate_placeholder, weights, biases, dynamic_dic=[])

    cost = tf.reduce_mean(tf.compat.v1.losses.sigmoid_cross_entropy(logits=predictions, 
        multi_class_labels=y_placeholder))
    correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(y_placeholder, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=model_params['learning_rate']).minimize(cost)

    saver = tf.compat.v1.train.Saver(var_list)

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())

        for epoch in range(model_params['training_epochs']):
            run_optim = sess.run(optimizer, feed_dict = {x_placeholder: training_data, y_placeholder: training_labels})
            loss, training_acc = sess.run([cost, accuracy], feed_dict = {x_placeholder: training_data, y_placeholder: training_labels})
            testing_acc = sess.run(accuracy, feed_dict = {x_placeholder: testing_data, y_placeholder: testing_labels})

            print("At iteration " + str(epoch) + ", Loss = " + \
                 "{:.4f}".format(loss) + ", Training Accuracy = " + \
                                "{:.4f}".format(training_acc) + ", Testing Accuracy = " + \
                                "{:.4f}".format(testing_acc))

        print("\nTraining complete.")

        save_path = saver.save(sess, "network_weights_data/" + str(network_iter) + "_MLP.ckpt")

        # Get predictions for each point in the mesh; this will enable later visualization of the decision boundary
        mesh_pred = sess.run(predictions, feed_dict = {x_placeholder: mesh})
        mesh_pred = np.argmax(mesh_pred, axis=1)

        return training_acc, testing_acc, mesh_pred

#From Gaurav-Kaushik (https://github.com/gaurav-kaushik/Data-Visualizations-Medium)
def create_mesh(matrix_2D, bound=.1, step=.02):
    """
    create_mesh will generate a mesh grid for a given matrix
    
    matrix_2D: input matrix (numpy)
    bound:     boundary around matrix (absolute value)
    step:      step size between each point in the mesh
    """

    # set bound as % of average of ranges for x and y
    bound = bound*np.average(np.ptp(matrix_2D, axis=0))
    
    # set step size as % of the average of ranges for x and y 
    step = step*np.average(np.ptp(matrix_2D, axis=0))

    # get boundaries
    x_min = matrix_2D[:,0].min() - bound
    x_max = matrix_2D[:,0].max() + bound
    y_min = matrix_2D[:,1].min() - bound 
    y_max = matrix_2D[:,1].max() + bound
    
    # create and return mesh
    mesh = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))
    return mesh

#Adapted from Gaurav-Kaushik (https://github.com/gaurav-kaushik/Data-Visualizations-Medium)
def plot_decision_boundaries(X_2D, mesh_pred, colormap_,
                            labelmap_, network_iter,
                            step_=0.02, title_="", xlabel_="", ylabel_=""):
    """
    X_2D:        2D numpy array of all data (training and testing)
    colormap_:   map of target:color (e.g. {'0': 'red', ...} )
    labelmap_:   map of target:label (e.g. {'0': 'setosa', ...} )
    step_:       step size for mesh (e.g lower = higher resolution)
    title_:      plot title
    xlabel_:     x-axis label
    ylabel_:     y-axis label
    """
    
    # Create a mesh
    mesh_ = create_mesh(X_2D, step=step_)

    # create color vectors [assume targets are last index]
    colors_mesh = [colormap_[str(int(m))] for m in mesh_pred.ravel()]


    """ Create ColumnDataSources  """
    source_mesh = ColumnDataSource(data=dict(mesh_x=mesh_[0].ravel(),
                                             mesh_y=mesh_[1].ravel(),
                                             colors_mesh=colors_mesh))    

    # Initiate Plot
    tools_ = ['crosshair', 'zoom_in', 'zoom_out', 'save', 'reset', 'tap', 'box_zoom']
    p = figure(title=title_, tools=tools_)
    p.xaxis.axis_label = xlabel_
    p.yaxis.axis_label = ylabel_

    # Plot mesh
    p_mesh = p.square('mesh_x', 'mesh_y', fill_color='colors_mesh',
               size = 13, line_alpha=0, fill_alpha=0.05, 
               source=source_mesh)
    
    save(obj=p, filename='./Boundary_visual_' + str(network_iter) + '.html', title="Decision_boundary")

    return

def normalize(training_data, testing_data):
    data_min = np.minimum(np.amin(training_data, axis=0), np.amin(testing_data, axis=0))
    data_max = np.maximum(np.amax(training_data, axis=0), np.amax(testing_data, axis=0))

    training_data = (training_data - data_min)/(data_max - data_min)
    testing_data = (testing_data - data_min)/(data_max - data_min)

    return training_data, testing_data

#Add additional (uninformative) feature dimensions
def dimension_augment(dim, x_data, reverse_bool=False):

    if reverse_bool == False:
        x_data = np.concatenate((x_data, np.zeros((np.shape(x_data)[0], 
            dim))), axis=1)
    else:
        x_data = np.concatenate((np.zeros((np.shape(x_data)[0], 
            dim)), x_data), axis=1)


    return x_data

def generate_toy_visual(model_params, adversarial_params, network_iter, all_results_df):

    iter_dic = {} #Store results 

    training_data, training_labels = generate_toy_data(model_params['data_size'], 
        model_params['additional_features_dimension'], training_data_bool=True)

    testing_data, testing_labels = generate_toy_data(adversarial_params['num_attack_examples'], 
        model_params['additional_features_dimension'], training_data_bool=False)

    training_data, testing_data = normalize(training_data, testing_data)

    x_placeholder, y_placeholder, dropout_rate_placeholder, weights, biases, var_list = toy_initializer(network_iter, model_params)


    #mesh_pred will determine how the mesh-data-squares are labeled, so which dimension one chooses to pass to the
    # plot mesh function (e.g. one of the two core dimensions, or one of the additional ones) is up to the user
    #It is only mesh_all_data that determines which dimensions are actually plotted

    #**Note mesh data for plotting (rather than predictions) is *not* expanded beyond the two non-zero dimensions of the input, in order to enable visualization
    


    #create_mesh essentially just takes the range of values of the input distribution, and using the step size
    # creates a mesh of appropriate range and grade/number of partitions 

    #the returned mesh_, therefore doesn't really encode any information about what the 'origin' dimension was
    #At this point it's a e.g. 2 x 121 x 121 array

    #I then expand the dimensionality of mesh_ to also have the zero and non-zero feature dimensions; this is done alongside
    # flattening the samples from the mesh, so that it becomes a e.g. 121*121= 14641 x 22 x 22 array

    #I then pass this long array of 'inputs' so that after training, the network's predictions for all of these data-points are evaluated
    #The returned labels are therefore e.g. a 14641 x 1 array

    #It's the fact that the 14641 inputs don't correspond to a random arrangement, but actually the 'movement' along
    # the mesh grid that essentially encodes the information which I'm subsequently seeing in the decision boundary

    #If I was to create the first two dimensions of the mesh grid as e.g. two extra zero-features ones,
    # then I believe I would get the decision boundary I want - because think about it, as you move along 
    # these dimensions, the network prediction might e.g. always be 1




    # the key is that I'm defining the mesh steps to be related to the first two dimensions, which based on my network 
    #and what it's been trained on, are the two main informative dimensions; this is all specified by how I join the 
    # mesh_ vector (which could really correspond to *any* of the two features) with the other features; if for example
    # I was to concatenate it *after* the zero dimensions, then it would be the first two zero dimensions that would be used





    mesh_all_data = np.concatenate((training_data, testing_data), axis=0)

    mesh_ = create_mesh(mesh_all_data, step=model_params['step'])


    print(training_data.shape)

    # Create the additional zero-valued, d-2 dimensional data to assess the effect of co-dimension (see Khoury, 2019 et al)
    # These features are *always* included when the network is making predictions (train, test, adversarial)
    training_data = dimension_augment(model_params['additional_zero_dimensions'], training_data)
    testing_data = dimension_augment(model_params['additional_zero_dimensions'], testing_data)

    print("Mesh shape is " + str(np.shape(mesh_)))
    mesh_ = dimension_augment(model_params['additional_zero_dimensions'], np.c_[mesh_[0].ravel(), mesh_[1].ravel()], reverse_bool=False)
    print("Mesh shape is " + str(np.shape(mesh_)))

    # *** NEED TO REFACTOR ***
    
    #Perform a second dimension augmentation to account for the semi-informative features that are 
    # not included when the actual mesh is created

    #** bear in mind above that the zero dimensions are added last on the training/test data

    mesh_ = dimension_augment(model_params['additional_features_dimension'], mesh_, reverse_bool=True)
    print("Mesh shape is " + str(np.shape(mesh_)))

    print(training_data.shape)


    functions = globals().copy()
    functions.update(locals())
    pred_function = functions.get('MLP_predictions')

    training_acc, testing_acc, mesh_pred = train_toy(pred_function, x_placeholder, y_placeholder, dropout_rate_placeholder,
        training_data, training_labels, testing_data, testing_labels, mesh_, weights, biases, var_list, model_params, network_iter)

    print("mesh_pred shape is " + str(np.shape(mesh_pred)))

    # iter_dic.update({'co_dim': model_params['additional_zero_dimensions'], 'training_accuracy':float(training_acc), 'testing_accuracy':float(testing_acc)})

    # update_dic = carry_out_attacks(model_params=model_params, adversarial_params=adversarial_params, 
    #     pred_function=pred_function, input_data=testing_data, input_labels=testing_labels, 
    #     x_placeholder=x_placeholder, var_list=var_list, weights=weights, biases=biases, 
    #     network_name_str=str(network_iter) + "_MLP", 
    #     iter_num=network_iter, dynamic_dic={})

    # iter_dic.update(update_dic)

    # print("\n\nThe cumulative results are...\n")
    # print(iter_dic)
    # iter_df = pd.DataFrame(data=iter_dic, index=[network_iter], dtype=np.float32)
    # all_results_df = all_results_df.append(iter_df)
    # all_results_df.to_pickle('Results.pkl')
    # all_results_df.to_csv('Results.csv')

    toy_colormap = {'0': 'red', '1': 'dodgerblue'} 
    toy_labelmap = {'0': 'negative', '1': 'positive'} 
    

    #Plot decision boundary
    plot_decision_boundaries(X_2D=mesh_all_data, mesh_pred=mesh_pred, step_=model_params['step'], colormap_=toy_colormap, 
                            labelmap_=toy_labelmap, network_iter=network_iter)

    return all_results_df

if __name__ == '__main__':

    with open('config_toy.yaml') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    model_params = params['model_params']

    adversarial_params = params['adversarial_params']

    print(model_params)
    print(adversarial_params)

    all_results_df=pd.DataFrame({})

    total_dim = 20

    for codim_iter in range(total_dim+1):


        #temporarily disable to test plotting function***
        # model_params['additional_features_dimension']=codim_iter
        # model_params['additional_zero_dimensions']=total_dim-codim_iter

        for network_iter in range(model_params['num_networks']):

            all_results_df = generate_toy_visual(model_params, adversarial_params, network_iter, all_results_df)
