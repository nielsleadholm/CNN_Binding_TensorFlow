#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import math
import pandas as pd
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from bokeh.plotting import output_notebook, figure, show
from bokeh.models import ColumnDataSource
from bokeh.io import export_png, save
import os
from Systematic_resistance_evaluation import carry_out_attacks


# Disable unecessary logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

#Boundary visualization code from Data Visualizations tutorial by Gaurav-Kaushik (https://github.com/gaurav-kaushik/Data-Visualizations-Medium)

def generate_toy_data(data_size, data_set, noise=None):

    if data_set == 'linear':

        x_data = np.random.uniform(size=(data_size,2))
        one_hot_labels = np.transpose(np.asarray((np.sum(x_data, axis=1)<1, np.sum(x_data, axis=1)>1)))

    elif data_set == 'circle':
        radius_1 = 1
        radius_2 = 3

        #Uniform distribution of angle
        uniform = np.random.uniform(low=0, high=2*math.pi, size=(data_size, 2))

        #Identify cartesian coordinates given desired radius
        inner_circle = np.asarray((radius_1*np.cos(uniform[:,0]), radius_1*np.sin(uniform[:,0])))

        outer_circle = np.asarray((radius_2*np.cos(uniform[:,0]), radius_2*np.sin(uniform[:,0])))
     
        x_data = np.transpose(np.concatenate((inner_circle, outer_circle), axis=1))

        if noise != None:
            print("Adding Gaussian noise to training data")
            x_data = x_data + np.random.normal(0, scale=noise, 
                size=np.shape(x_data))

        np.random.shuffle(x_data)
        
        #Not efficient label generation but useful sanity check
        one_hot_labels = np.transpose(np.asarray((np.sqrt(np.square(x_data[:,0]) + np.square(x_data[:,1])) < 2, 
            np.sqrt(np.square(x_data[:,0]) + np.square(x_data[:,1])) > 2)))


    return x_data, one_hot_labels

def toy_initializer(network_iter, model_params):
    
    tf.reset_default_graph()

    x_placeholder = tf.compat.v1.placeholder(tf.float32, [None, 2])
    y_placeholder = tf.compat.v1.placeholder(tf.int32, [None, 2])
    dropout_rate_placeholder = tf.compat.v1.placeholder(tf.float32)

    initializer = tf.contrib.layers.variance_scaling_initializer()

    with tf.name_scope('Network_' + str(network_iter)):
        weights = {
            'w1' : tf.compat.v1.get_variable('w1', shape=(2,model_params['network_width']), initializer=initializer),
            'w2' : tf.compat.v1.get_variable('w2', shape=(model_params['network_width'],model_params['network_width']), initializer=initializer),
            'w3' : tf.compat.v1.get_variable('w3', shape=(model_params['network_width'],2), initializer=initializer)
        }

        biases = {
            'b1' : tf.compat.v1.get_variable('b1', shape=(model_params['network_width']), initializer=initializer),
            'b2' : tf.compat.v1.get_variable('b2', shape=(model_params['network_width']), initializer=initializer),
            'b3' : tf.compat.v1.get_variable('b3', shape=(2), initializer=initializer)
        }

    var_list = [weights['w1'], weights['w2'], weights['w3'], biases['b1'], biases['b2'], biases['b3']]

    if (model_params['architecture'] == 'BindingMLP'):
        weights['w1_binding'] = tf.compat.v1.get_variable('w1_binding', shape=(model_params['network_width'], 2), initializer=initializer)

        var_list.append(weights['w1_binding'])


    return x_placeholder, y_placeholder, dropout_rate_placeholder, weights, biases, var_list

def MLP_predictions(x_input, dropout_rate_placeholder, weights, biases, dynamic_dic):

    layer_1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(tf.dtypes.cast(x_input, dtype=tf.float32), weights['w1']), biases['b1']))
    layer_2 = tf.nn.relu(tf.nn.bias_add(tf.matmul(layer_1, weights['w2']), biases['b2']))

    logits = tf.nn.bias_add(tf.matmul(layer_2, weights['w3']), biases['b3'])

    dummy = {}

    return logits, {}, {}, 0.0, 0.0

def BindingMLP_predictions(x_input, dropout_rate_placeholder, weights, biases, dynamic_dic):

    layer_1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(tf.dtypes.cast(x_input, dtype=tf.float32), weights['w1']), biases['b1']))
    layer_2 = tf.nn.relu(tf.nn.bias_add(tf.matmul(layer_1, weights['w2']), biases['b2']))

    binding_layer = gradient_unpooling_sequence(layer_2, layer_1, low_flat_shape=[-1,dynamic_dic['binding_width']])

    logits = tf.nn.bias_add(tf.add(tf.matmul(layer_2, weights['w3']), 
        tf.matmul(binding_layer, weights['w1_binding'])), biases['b3'])

    return logits, {}, {}, 0.0, 0.0

def gradient_unpooling_sequence(high_level, low_level, low_flat_shape):

    #Extract binding information for low-level neurons that are driving critical (i.e. max-pooled) mid-level neurons
    binding_grad = tf.squeeze(tf.gradients(high_level, low_level, unconnected_gradients=tf.UnconnectedGradients.ZERO), 0) #Squeeze removes the dimension of the gradient tensor that stores dtype
    binding_grad_flat = tf.reshape(binding_grad, low_flat_shape)

    #Use k-th largest value as a threshold for getting a boolean mask
    values, _ = tf.math.top_k(binding_grad_flat, k=round(low_flat_shape[1]*0.25))
    kth = tf.reduce_min(values, axis=1)
    mask = tf.greater_equal(binding_grad_flat, tf.expand_dims(kth, -1))
    low_level_flat = tf.reshape(low_level, low_flat_shape) 
    gradient_unpool_binding_activations = tf.multiply(low_level_flat, tf.dtypes.cast(mask, dtype=tf.float32)) #Apply the Boolean mask element-wise

    return gradient_unpool_binding_activations

def train_toy(x_placeholder, y_placeholder, dropout_rate_placeholder, training_data, training_labels, 
        testing_data, testing_labels, mesh, weights, biases, var_list, model_params, network_iter):
    
    if model_params['architecture'] == 'MLP':
        print("\nTraining standard MLP")
        predictions, _, _, _, _ = MLP_predictions(x_placeholder, dropout_rate_placeholder, weights, biases, model_params['dynamic_dic'])
    elif model_params['architecture'] == 'BindingMLP':
        print("\nTraining Binding MLP")
        predictions, _, _, _, _ = BindingMLP_predictions(x_placeholder, dropout_rate_placeholder, weights, biases, dynamic_dic=model_params['dynamic_dic'])

    cost = tf.reduce_mean(tf.compat.v1.losses.sigmoid_cross_entropy(logits=predictions, 
        multi_class_labels=y_placeholder, label_smoothing=model_params['smoothing_coefficient']))
    correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(y_placeholder, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=0.1).minimize(cost)

    saver = tf.compat.v1.train.Saver(var_list)

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())

        for epoch in range(model_params['training_epochs']):
            run_optim = sess.run(optimizer, feed_dict = {x_placeholder: training_data, dropout_rate_placeholder: None, y_placeholder: training_labels})
            loss, training_acc = sess.run([cost, accuracy], feed_dict = {x_placeholder: training_data, dropout_rate_placeholder: None, y_placeholder: training_labels})

            testing_acc = sess.run(accuracy, feed_dict = {x_placeholder: testing_data, dropout_rate_placeholder: None, y_placeholder: testing_labels})

            print("At iteration " + str(epoch) + ", Loss = " + \
                 "{:.4f}".format(loss) + ", Training Accuracy = " + \
                                "{:.4f}".format(training_acc) + ", Testing Accuracy = " + \
                                "{:.4f}".format(testing_acc))

        print("\nTraining complete.")

        save_path = saver.save(sess, "network_weights_data/" + str(network_iter) + '_' + model_params['architecture'] + str(model_params['network_width']) + ".ckpt")

        # Get predictions for each point in the mesh
        mesh_pred = sess.run(predictions, feed_dict = {x_placeholder: np.c_[mesh[0].ravel(), mesh[1].ravel()], dropout_rate_placeholder: None, })
        
        # Get predictions for test data, and for all data
        test_pred = sess.run(predictions, feed_dict = {x_placeholder: testing_data, dropout_rate_placeholder: None, })
        data_pred = sess.run(predictions, feed_dict = {x_placeholder: np.concatenate((training_data, testing_data), axis=0), dropout_rate_placeholder: None, })

        #print(mesh_pred[0:5])
        mesh_pred = np.argmax(mesh_pred, axis=1)
        #print(mesh_pred[0:5])
        test_pred = np.argmax(test_pred, axis=1)
        data_pred = np.argmax(data_pred, axis=1)


        return training_acc, testing_acc, mesh_pred, test_pred, data_pred


def visualize_data(data, labels):
    
    
    plt.scatter(data[:, 0][np.where(labels[:,0]==1)], data[:,1][np.where(labels[:,0]==1)], color='dodgerblue')
    plt.scatter(data[:, 0][np.where(labels[:,1]==1)], data[:,1][np.where(labels[:,1]==1)], color='crimson')
    plt.savefig('visual.jpg')
    plt.show()


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


def plot_decision_boundaries(X_2D, targets, labels, X_test_, 
                            mesh_pred, test_pred, data_pred,
                             colormap_, labelmap_, network_iter, 
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

    source_mesh = ColumnDataSource(data=dict(mesh_x=mesh_[0].ravel(),
                                             mesh_y=mesh_[1].ravel(),
                                             colors_mesh=colors_mesh))    
    
    # Initiate Plot
    tools_ = ['crosshair', 'zoom_in', 'zoom_out', 'save', 'reset', 'tap', 'box_zoom']
    p = figure(title=title_, tools=tools_)
    p.xaxis.axis_label = xlabel_
    p.yaxis.axis_label = ylabel_

    # plot all data
    p_data = p.circle('X', 'Y', fill_color='colors',
                  size=10, alpha=0.5, line_alpha=0, 
                  source=source_data, name='Data')
    
    # plot thick outline around predictions on test data
    p_test = p.circle('X_test', 'Y_test', line_color='colors_test_pred',
                  size=12, alpha=1, line_width=3, fill_alpha=0,
                  source=source_test)

    # plot mesh
    p_mesh = p.square('mesh_x', 'mesh_y', fill_color='colors_mesh',
               size = 13, line_alpha=0, fill_alpha=0.05, 
               source=source_mesh)
    
    # add hovertool
    # hover_1 = HoverTool(names=['Data'], 
    #                     tooltips=[("truth", "@colors_legend"), ("prediction", "@colors_pred_data")], 
    #                     renderers=[p_data])
    # p.add_tools(hover_1)

    #show(p)
    save(obj=p, filename='./Boundary_visual_' + str(network_iter) + '.html', title="Decision_boundary")

    return

def generate_toy_visual(model_params, adversarial_params, network_iter, all_results_df):

    iter_dic = {}

    data_size = model_params['data_size']
    step = model_params['step']
    data_set = model_params['data_set']

    training_data, training_labels = generate_toy_data(data_size, data_set, noise=model_params['Gaussian_noise'])

    #smooth_training_labels = label_smoothing(training_labels, model_params['smoothing_coefficient'])

    testing_data, testing_labels = generate_toy_data(data_size, data_set, noise=None)

    #Normalize data, accounting for larger values in training data if noise added
    data_min = np.minimum(np.amin(training_data, axis=0), np.amin(testing_data, axis=0))
    data_max = np.maximum(np.amax(training_data, axis=0), np.amax(testing_data, axis=0))

    training_data = (training_data - data_min)/(data_max - data_min)
    testing_data = (testing_data - data_min)/(data_max - data_min)


    x_placeholder, y_placeholder, dropout_rate_placeholder, weights, biases, var_list = toy_initializer(network_iter, model_params)

    #visualize_data(training_data, training_labels)

    all_data = np.concatenate((training_data, testing_data), axis=0)
    all_labels = (np.concatenate((training_labels, testing_labels), axis=0)[:, 1]).astype(int)

    mesh_ = create_mesh(all_data, step=step)

    training_acc, testing_acc, mesh_pred, test_pred, data_pred = train_toy(x_placeholder, y_placeholder, dropout_rate_placeholder, training_data, training_labels, testing_data, testing_labels, mesh_,
     weights, biases, var_list, model_params, network_iter)

    iter_dic.update({'training_accuracy':float(training_acc), 'testing_accuracy':float(testing_acc)})

    functions = globals().copy()
    functions.update(locals())
    pred_function = functions.get(model_params['architecture'] + '_predictions')
    if not pred_function:
         raise NotImplementedError("Prediction function %s not implemented" % (model_params['architecture'] + '_predictions'))


    iter_dic.update(carry_out_attacks(model_params=model_params, adversarial_params=adversarial_params, 
        pred_function=pred_function, input_data=testing_data, input_labels=testing_labels, 
        x_placeholder=x_placeholder, var_list=var_list, weights=weights, biases=biases, 
        network_name_str=str(network_iter) + '_' + model_params['architecture'] + str(model_params['network_width']), 
        iter_num=network_iter, dynamic_dic=model_params['dynamic_dic']))

    print("\n\nThe cumulative results are...\n")
    print(iter_dic)
    iter_df = pd.DataFrame(data=iter_dic, index=[network_iter], dtype=np.float32)
    all_results_df = all_results_df.append(iter_df)
    print(all_results_df)
    all_results_df.to_pickle('Results.pkl')
    all_results_df.to_csv('Results.csv')

    toy_colormap = {'0': 'red', '1': 'dodgerblue'} 
    toy_labelmap = {'0': 'negative', '1': 'positive'} 
    toy_labels = [toy_labelmap[str(x)] for x in all_labels]

    plot_decision_boundaries(X_2D=all_data, targets=all_labels, labels=toy_labels, X_test_=testing_data, 
                            mesh_pred=mesh_pred, test_pred=test_pred, data_pred=data_pred,
                             step_=step, colormap_=toy_colormap, labelmap_=toy_labelmap, network_iter=network_iter)

    return all_results_df

if __name__ == '__main__':

    model_params = {
    'architecture' : 'MLP',
    'network_width' : 8,
    'dynamic_dic' : {'binding_width': 8},
    'training_epochs' : 50,
    'Gaussian_noise' : 0.3,
    'smoothing_coefficient' : 0.01,
    'step' : 0.01,
    'data_set' : 'circle',
    'num_networks' : 20,
    'data_size' : 5000,
    'batch_size' : 128
    }

    adversarial_params = {
        'num_attack_examples':128,
        'transfer_attack_setup':False,
        'estimate_gradients':False,
        'boundary_attack_iterations':100,
        'boundary_attack_log_steps':100,
        'perturbation_threshold':{'L0':12, 'LInf':0.3,  'L2':1.5},
        'save_images':False
        }

    all_results_df=pd.DataFrame({})

    for network_iter in range(model_params['num_networks']):

        all_results_df = generate_toy_visual(model_params, adversarial_params, network_iter, all_results_df)
