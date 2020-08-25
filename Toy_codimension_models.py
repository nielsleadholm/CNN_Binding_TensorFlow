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

    return x_placeholder, y_placeholder, weights, biases, var_list

def MLP_predictions(x_input, weights, biases):

    layer_1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(tf.dtypes.cast(x_input, dtype=tf.float32), weights['w1']), biases['b1']))

    logits = tf.nn.bias_add(tf.matmul(layer_1, weights['w2']), biases['b2'])

    scalar_dic = {} #Used in CNN_module but not required here

    return logits, scalar_dic

def train_toy(pred_function, x_placeholder, y_placeholder, training_data, training_labels, 
        testing_data, testing_labels, mesh, weights, biases, var_list, model_params, network_iter):
    
    predictions, _ = pred_function(x_placeholder, weights, biases)

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

        # Get predictions for each point in the mesh, for test data, and for all data
        mesh_pred = sess.run(predictions, feed_dict = {x_placeholder: mesh})
        test_pred = sess.run(predictions, feed_dict = {x_placeholder: testing_data})
        data_pred = sess.run(predictions, feed_dict = {x_placeholder: np.concatenate((training_data, testing_data), axis=0)})

        mesh_pred = np.argmax(mesh_pred, axis=1)
        test_pred = np.argmax(test_pred, axis=1)
        data_pred = np.argmax(data_pred, axis=1)

        return training_acc, testing_acc, mesh_pred, test_pred, data_pred

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
def plot_decision_boundaries(X_2D, targets, labels, X_test_, 
                            mesh_pred, test_pred, data_pred, X_adver_, adver_pred,
                             colormap_, colormap_adver_, labelmap_, network_iter, boundary_iter,
                            step_=0.02, title_="", xlabel_="", ylabel_=""):
    """
    X_2D:        2D numpy array of all data (training and testing)
    targets:     array of target data (e.g 0's and 1's)
    labels:      array of labels for target data (e.g. 'positive' and 'negative')
    X_test_:     test data taken (e.g. from test_train_split())
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
    colors = [colormap_[str(t)] for t in targets]
    colors_test_pred = [colormap_[str(p)] for p in test_pred]
    colors_adver_pred = [colormap_adver_[str(p)] for p in adver_pred]
    colors_pred_data = [labelmap_[str(x)] for x in data_pred]
    colors_mesh = [colormap_[str(int(m))] for m in mesh_pred.ravel()]


    """ Create ColumnDataSources  """
    source_data = ColumnDataSource(data=dict(X=X_2D[:,0], 
                                             Y=X_2D[:,1],
                                             colors=colors, 
                                             colors_legend=labels,
                                             colors_pred_data=colors_pred_data))


    source_test = ColumnDataSource(data=dict(X_test=X_test_[:,0],
                                                    Y_test=X_test_[:,1],
                                                    colors_test_pred=colors_test_pred))

    source_adver = ColumnDataSource(data=dict(X_adver=X_adver_[:,0],
                                                    Y_adver=X_adver_[:,1],
                                                    colors_adver_pred=colors_adver_pred))

    source_mesh = ColumnDataSource(data=dict(mesh_x=mesh_[0].ravel(),
                                             mesh_y=mesh_[1].ravel(),
                                             colors_mesh=colors_mesh))    

    print("Half-value")
    half=int(len(X_test_)/2)
    print(half)
    
    if boundary_iter == 0:
        segment_color='navy'
        source_adver_shift = ColumnDataSource(data=dict(
            x=X_adver_[:,0],
            y=X_adver_[:,1],
            xm01=X_test_[:half,0],
            ym01=X_test_[:half,1]
        )
    )
    else:
        segment_color='pink'
        source_adver_shift = ColumnDataSource(data=dict(
            x=X_adver_[:,0],
            y=X_adver_[:,1],
            xm01=X_test_[half:,0],
            ym01=X_test_[half:,1]
        )
    )


    # Initiate Plot
    tools_ = ['crosshair', 'zoom_in', 'zoom_out', 'save', 'reset', 'tap', 'box_zoom']
    p = figure(title=title_, tools=tools_)
    p.xaxis.axis_label = xlabel_
    p.yaxis.axis_label = ylabel_

    # plot all data
    # p_data = p.circle('X', 'Y', fill_color='colors',
    #               size=10, alpha=0.5, line_alpha=0, 
    #               source=source_data, name='Data')
    
    # plot thick outline around predictions on test data
    p_test = p.circle('X_test', 'Y_test', line_color='colors_test_pred',
                  size=8, alpha=0.75, line_width=3, fill_alpha=0,
                  source=source_test, legend_label='Test Data')

    # plot mesh
    p_mesh = p.square('mesh_x', 'mesh_y', fill_color='colors_mesh',
               size = 13, line_alpha=0, fill_alpha=0.05, 
               source=source_mesh)
    

    #plot predictions on adversarial data
    p_data = p.circle('X_adver', 'Y_adver', fill_color='colors_adver_pred',
                  size=12, alpha=0.75, line_alpha=0, 
                  source=source_adver, legend_label='Adversarial Images')


    adver_shift = Segment(x0="x", y0="y", x1="xm01", y1="ym01", line_color=segment_color, line_width=3)
    p.add_glyph(source_adver_shift, adver_shift)


    # add hovertool
    # hover_1 = HoverTool(names=['Data'], 
    #                     tooltips=[("truth", "@colors_legend"), ("prediction", "@colors_pred_data")], 
    #                     renderers=[p_data])
    # p.add_tools(hover_1)

    #show(p)
    save(obj=p, filename='./Boundary_visual_' + str(network_iter) + 'boundary_' + str(boundary_iter) + '.html', title="Decision_boundary")

    return

def normalize(training_data, testing_data):
    #Normalize data
    data_min = np.minimum(np.amin(training_data, axis=0), np.amin(testing_data, axis=0))
    data_max = np.maximum(np.amax(training_data, axis=0), np.amax(testing_data, axis=0))

    training_data = (training_data - data_min)/(data_max - data_min)
    testing_data = (testing_data - data_min)/(data_max - data_min)

    return training_data, testing_data

#Add additional (uninformative) feature dimensions
def dimension_augment(dim, x_data):

    x_data = np.concatenate((x_data, np.zeros((np.shape(x_data)[0], 
        dim))), axis=1)

    return x_data

def generate_toy_visual(model_params, adversarial_params, network_iter, attack_for_visual, all_results_df):

    iter_dic = {} #Store results 

    training_data, training_labels = generate_toy_data(model_params['data_size'], 
        model_params['additional_features_dimension'], training_data_bool=True)

    testing_data, testing_labels = generate_toy_data(adversarial_params['num_attack_examples'], 
        model_params['additional_features_dimension'], training_data_bool=False)

    training_data, testing_data = normalize(training_data, testing_data)

    x_placeholder, y_placeholder, weights, biases, var_list = toy_initializer(network_iter, model_params)


    #Note mesh data for plotting (rather than predictions) is *not* expanded beyond the two non-zero dimensions of the input, in order to enable visualization
    mesh_all_data = np.concatenate((training_data, testing_data), axis=0)
    mesh_all_labels = (np.concatenate((training_labels, testing_labels), axis=0)[:, 1]).astype(int)

    mesh_ = create_mesh(mesh_all_data, step=model_params['step'])


    print(training_data.shape)

    # Create the additional zero-valued, d-2 dimensional data to assess the effect of co-dimension (see Khoury, 2019 et al)
    # These features are *always* included when the network is making predictions (train, test, adversarial)
    training_data = dimension_augment(model_params['additional_zero_dimensions'], training_data)
    testing_data = dimension_augment(model_params['additional_zero_dimensions'], testing_data)

    print("Mesh shape is " + str(np.shape(mesh_)))
    mesh_ = dimension_augment(model_params['additional_zero_dimensions'], np.c_[mesh_[0].ravel(), mesh_[1].ravel()])
    print("Mesh shape is " + str(np.shape(mesh_)))

    # *** NEED TO REFACTOR ***
    
    #Perform a second dimension augmentation to account for the semi-informative features that are 
    # not included when the actual mesh is created
    mesh_ = dimension_augment(model_params['additional_features_dimension'], mesh_)
    print("Mesh shape is " + str(np.shape(mesh_)))

    print(training_data.shape)


    functions = globals().copy()
    functions.update(locals())
    pred_function = functions.get('MLP_predictions')

    training_acc, testing_acc, mesh_pred, test_pred, data_pred = train_toy(pred_function, x_placeholder, y_placeholder, 
        training_data, training_labels, testing_data, testing_labels, mesh_,
     weights, biases, var_list, model_params, network_iter)


    iter_dic.update({'co_dim': model_params['additional_zero_dimensions'], 'training_accuracy':float(training_acc), 'testing_accuracy':float(testing_acc)})

    update_dic = carry_out_attacks(model_params=model_params, adversarial_params=adversarial_params, 
        pred_function=pred_function, input_data=testing_data, input_labels=testing_labels, 
        x_placeholder=x_placeholder, var_list=var_list, weights=weights, biases=biases, 
        network_name_str=str(network_iter) + "_MLP", 
        iter_num=network_iter)


    iter_dic.update(update_dic)

    print("\n\nThe cumulative results are...\n")
    print(iter_dic)
    iter_df = pd.DataFrame(data=iter_dic, index=[network_iter], dtype=np.float32)
    all_results_df = all_results_df.append(iter_df)
    print(all_results_df)
    all_results_df.to_pickle('Results.pkl')
    all_results_df.to_csv('Results.csv')


    toy_colormap = {'0': 'red', '1': 'dodgerblue'} 
    toy_labelmap = {'0': 'negative', '1': 'positive'} 
    toy_labels = [toy_labelmap[str(x)] for x in mesh_all_labels]

    # adver_colormap = {'0': 'pink', '1': 'navy'} 

    # #Note that Carry_out_attacks will return a None (under class/prediction) and array of zeros where the attack
    # #was unsucessful; this is key to how tfCore_adversarial e.g. calculates distances, so rather than change this
    # #we use the locations of where an attack was unsuccessful to retrieve the original (true) label and the associated input features-data
    # #Note therefore that due to the label these will be clearly distinguishable from unaltered data that was misclassified

    # failed_adver_indices = np.nonzero(adver_pred_dic[attack_for_visual] == None)

    # print(np.shape(adver_pred_dic[attack_for_visual]))
    # print(adver_pred_dic[attack_for_visual])
    # print(np.shape(testing_labels))
    # print(failed_adver_indices)

    # adver_pred_dic[attack_for_visual][failed_adver_indices] = ((testing_labels[:,1]==1)[failed_adver_indices]).astype(int)
    # adver_data_dic[attack_for_visual][failed_adver_indices, :] = testing_data[failed_adver_indices, :]
    # adver_pred = adver_pred_dic[attack_for_visual]
    # adver_data = adver_data_dic[attack_for_visual]

    # half = int(adversarial_params['num_attack_examples']/2)

    # print(np.shape(test_pred))
    # print(np.shape(adver_pred))

    # print("\n *** adversarial predictions ***")
    # print(test_pred[0:20])
    # print(adver_pred[0:20])

    # #Plot test samples on decision boundary
    # plot_decision_boundaries(X_2D=mesh_all_data, targets=mesh_all_labels, labels=toy_labels, X_test_=testing_data, 
    #                         mesh_pred=mesh_pred, test_pred=test_pred, data_pred=data_pred, X_adver_=adver_data[0:half], 
    #                         adver_pred=adver_pred[0:half],
    #                          step_=model_params['step'], colormap_=toy_colormap, colormap_adver_=adver_colormap,
    #                           labelmap_=toy_labelmap, network_iter=network_iter, boundary_iter=0)

    # plot_decision_boundaries(X_2D=mesh_all_data, targets=mesh_all_labels, labels=toy_labels, X_test_=testing_data, 
    #                         mesh_pred=mesh_pred, test_pred=test_pred, data_pred=data_pred, X_adver_=adver_data[half:], 
    #                         adver_pred=adver_pred[half:],
    #                          step_=model_params['step'], colormap_=toy_colormap, colormap_adver_=adver_colormap,
    #                           labelmap_=toy_labelmap, network_iter=network_iter, boundary_iter=1)

    return all_results_df

if __name__ == '__main__':

    with open('config_toy.yaml') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    attack_for_visual = 'boundary'
    model_params = params['model_params']

    adversarial_params = params['adversarial_params']

    print(model_params)
    print(adversarial_params)

    all_results_df=pd.DataFrame({})

    total_dim = 20

    for codim_iter in range(total_dim+1):

        model_params['additional_features_dimension']=codim_iter
        model_params['additional_zero_dimensions']=total_dim-codim_iter

        for network_iter in range(model_params['num_networks']):

            all_results_df = generate_toy_visual(model_params, adversarial_params, network_iter, attack_for_visual, all_results_df)
