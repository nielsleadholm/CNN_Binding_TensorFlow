#!/usr/bin/env python3

import numpy as np
import math
import os
import pandas as pd
import yaml
import tensorflow as tf
from Systematic_resistance_evaluation import carry_out_attacks

#Build and evaluate an MLP on a toy-dataset to examine the effect of co-dimension on adversarial robustness

# Suppress unecessary logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

def generate_toy_data(data_size, additional_feature_dimensions, training_data_bool):

    #Data-set is multi-dimensional Gaussian in which two dimensions have little overlap, as well as
    # an additional number of dimensions that are not as easily seperable, but which still carry some class information

    manifold_noise = 0.15

    #Note the difference in means
    base_features_zero_class = np.random.normal(0, scale=manifold_noise, size=(int(data_size/2),2)) #The base two features
    base_features_one_class = np.random.normal(1, scale=manifold_noise, size=(int(data_size/2),2))

    #Note the change in means and the use of additional_feature_dimensions parameter
    imperfect_features_zero_class = np.random.normal(0, scale=manifold_noise, size=(int(data_size/2),additional_feature_dimensions))
    imperfect_features_one_class = np.random.normal(0.5, scale=manifold_noise, size=(int(data_size/2),additional_feature_dimensions))

    zero_class_data = np.concatenate((base_features_zero_class, imperfect_features_zero_class), axis=1)
    one_class_data = np.concatenate((base_features_one_class, imperfect_features_one_class), axis=1)

    x_data = np.concatenate((zero_class_data, one_class_data), axis=0)

    one_hot_labels = np.concatenate(
        (np.concatenate((np.ones(shape=(int(data_size/2),1)), np.zeros(shape=(int(data_size/2),1))), axis=1),
        np.concatenate((np.zeros(shape=(int(data_size/2),1)), np.ones(shape=(int(data_size/2),1))), axis=1)), 
        axis=0)

    x_data, one_hot_labels = randomize_order(x=x_data, y=one_hot_labels)

    return x_data, one_hot_labels

def randomize_order(x, y):
    #Randomize data-sample orders
    for_shuffle = list(zip(x, y))
    np.random.shuffle(for_shuffle)
    x, y = zip(*for_shuffle)

    #Get y back to an np array
    y = np.array(list(map(list, y)))

    return x,y

def toy_initializer(network_iter, model_params):
    
    tf.reset_default_graph()

    input_dim = 2+model_params['additional_feature_dimensions']+model_params['additional_zero_dimensions'] #the 2 accounts for the core, easily separable dimensions

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

    scalar_dic = {} #Used in CNN_module and therefore expected in adversarial evaluations but not required here

    return logits, scalar_dic

def train_toy(pred_function, x_placeholder, y_placeholder, dropout_rate_placeholder, training_data, training_labels, 
        testing_data, testing_labels, weights, biases, var_list, model_params, network_iter):
    
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

        return training_acc, testing_acc

def normalize(training_data, testing_data):
    data_min = np.minimum(np.amin(training_data, axis=0), np.amin(testing_data, axis=0))
    data_max = np.maximum(np.amax(training_data, axis=0), np.amax(testing_data, axis=0))

    training_data = (training_data - data_min)/(data_max - data_min)
    testing_data = (testing_data - data_min)/(data_max - data_min)

    return training_data, testing_data

#Add additional (uninformative) feature dimensions that carry no class information
def dimension_augment(dim, x_data):

    x_data = np.concatenate((np.zeros((np.shape(x_data)[0], 
        dim)), x_data), axis=1)

    return x_data

def evaluate_toy_net(model_params, adversarial_params, network_iter, codim_results_df):

    iter_dic = {} #Store results 

    training_data, training_labels = generate_toy_data(model_params['data_size'], 
        model_params['additional_feature_dimensions'], training_data_bool=True)

    testing_data, testing_labels = generate_toy_data(adversarial_params['num_attack_examples'], 
        model_params['additional_feature_dimensions'], training_data_bool=False)

    training_data, testing_data = normalize(training_data, testing_data)

    x_placeholder, y_placeholder, dropout_rate_placeholder, weights, biases, var_list = toy_initializer(network_iter, model_params)

    # Create the additional zero-valued features to assess the effect of co-dimension (see Khoury, 2019 et al)
    training_data = dimension_augment(model_params['additional_zero_dimensions'], training_data)
    testing_data = dimension_augment(model_params['additional_zero_dimensions'], testing_data)

    functions = globals().copy()
    functions.update(locals())
    pred_function = functions.get('MLP_predictions')

    training_acc, testing_acc = train_toy(pred_function, x_placeholder, y_placeholder, dropout_rate_placeholder,
        training_data, training_labels, testing_data, testing_labels, weights, biases, var_list, model_params, network_iter)

    iter_dic.update({'co_dim': model_params['additional_zero_dimensions'], 'training_accuracy':float(training_acc), 'testing_accuracy':float(testing_acc)})

    update_dic = carry_out_attacks(model_params=model_params, adversarial_params=adversarial_params, 
        pred_function=pred_function, input_data=testing_data, input_labels=testing_labels, 
        x_placeholder=x_placeholder, var_list=var_list, weights=weights, biases=biases, 
        network_name_str=str(network_iter) + "_MLP", 
        iter_num=network_iter, dynamic_dic={})

    iter_dic.update(update_dic)

    print("\n\nThe network iter results are...\n")
    print(iter_dic)

    iter_df = pd.DataFrame(data=iter_dic, index=[network_iter], dtype=np.float32)
    codim_results_df = codim_results_df.append(iter_df)

    return codim_results_df

def df_median(codim_results_df, codim_value):

    mask = (codim_results_df['co_dim'] == codim_value)
    LInf_median = np.median(codim_results_df['BIM_LInf_distances'][mask])
    L2_median = np.median(codim_results_df['BIM_L2_distances'][mask])

    return {'LInf_median':LInf_median, 'L2_median':L2_median}

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

        model_params['additional_feature_dimensions']=codim_iter
        model_params['additional_zero_dimensions']=total_dim-codim_iter

        codim_results_df=pd.DataFrame({})

        for network_iter in range(model_params['num_networks']):

            codim_results_df = evaluate_toy_net(model_params, adversarial_params, network_iter, codim_results_df)

        median_dic = df_median(codim_results_df, model_params['additional_zero_dimensions'])
        median_df = pd.DataFrame(data=median_dic, index=[model_params['additional_zero_dimensions']], dtype=np.float32)

        all_results_df = all_results_df.append(median_df)
        print("\nCumulative results:")
        print(all_results_df)
        all_results_df.to_pickle('Results.pkl')
        all_results_df.to_csv('Results.csv')
