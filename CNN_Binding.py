import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#The following program has been built to run on Colaboratory
#It implements a simple CNN based on the architecture of LeNet-5, for the MNIST dataset
#Further modifications are intended to implement 'hierarchical feature binding' in a deep learning context
#For information on hierarchical feature binding, see the paper "The Emergence of Polychronization and Feature 
#Binding in a Spiking Neural Network Model of the Primate Ventral Visual System", Eguchi, 2018 at
#(http://psycnet.apa.org/fulltext/2018-25960-001.html)

#Use the small (20k examples) MNIST dataset available in Colaboratory
training_import = np.genfromtxt('mnist_train_small.csv', delimiter=",")
testing_import = np.genfromtxt('mnist_test.csv', delimiter=',')

#Separate the training and testing images and their labels
training_data = training_import[:, 1:]
training_labels = training_import[:, 0]

testing_data = testing_import[:, 1:]
testing_labels = testing_import[:, 0]

#Rescale images to values between 0:1 and reshape so each image is 28x28
training_data = training_data/255
training_data = np.reshape(training_data, [np.shape(training_data)[0], 28, 28, 1])

testing_data = testing_data/255
testing_data = np.reshape(testing_data, [np.shape(testing_data)[0], 28, 28, 1])

#Transform the labels into one-hot encoding
num_classes = 10
training_labels = np.eye(num_classes)[training_labels.astype(int)]

testing_labels = np.eye(num_classes)[testing_labels.astype(int)]

#Define training parameters
batch_size = 128
training_epochs = 3

#Declare placeholders for the input features and labels
#The first dimension of the palceholder shape is set to None as this will later be defined by the batch size
x = tf.placeholder(training_data.dtype, [None, 28, 28, 1])
y = tf.placeholder(training_labels.dtype, [None, num_classes])

#Define weight and bias variables, initialize values, and create a variable list for holding their keys
weights_Binding = {
    'conv_W1_bind' : tf.get_variable('CW1_bind', shape=(5, 5, 1, 6), initializer=tf.contrib.layers.xavier_initializer()),
    'conv_W2_bind' : tf.get_variable('CW2_bind', shape=(5, 5, 6, 16), initializer=tf.contrib.layers.xavier_initializer()),
    'dense_W1_bind' : tf.get_variable('DW1_bind', shape=(400+1600, 120), initializer=tf.contrib.layers.xavier_initializer()),
    'dense_W2_bind' : tf.get_variable('DW2_bind', shape=(120, 84), initializer=tf.contrib.layers.xavier_initializer()),
    'output_W_bind' : tf.get_variable('OW_bind', shape=(84, num_classes), initializer=tf.contrib.layers.xavier_initializer())
}

biases_Binding = {
    'conv_b1_bind' : tf.get_variable('Cb1_bind', shape=(6), initializer=tf.contrib.layers.xavier_initializer()),
    'conv_b2_bind' : tf.get_variable('Cb2_bind', shape=(16), initializer=tf.contrib.layers.xavier_initializer()),
    'dense_b1_bind' : tf.get_variable('Db1_bind', shape=(120), initializer=tf.contrib.layers.xavier_initializer()),
    'dense_b2_bind' : tf.get_variable('Db2_bind', shape=(84), initializer=tf.contrib.layers.xavier_initializer()),
    'output_b_bind' : tf.get_variable('Ob_bind', shape=(num_classes), initializer=tf.contrib.layers.xavier_initializer())
}

var_list_Binding = [weights_Binding['conv_W1_bind'], weights_Binding['conv_W2_bind'], weights_Binding['dense_W1_bind'], 
                    weights_Binding['dense_W2_bind'], weights_Binding['output_W_bind'], biases_Binding['conv_b1_bind'], 
                    biases_Binding['conv_b2_bind'], biases_Binding['dense_b1_bind'], biases_Binding['dense_b2_bind'], 
                    biases_Binding['output_b_bind']]


#Define the convolutional model now with binding information
def cnn_binding_predictions(features, temp_batch_size):

    conv1 = tf.nn.conv2d(input=tf.dtypes.cast(features, dtype=tf.float32), filter=weights_Binding['conv_W1_bind'], 
                         strides=[1, 1, 1, 1], padding="SAME")
    conv1 = tf.nn.bias_add(conv1, biases_Binding['conv_b1_bind'])
    relu1 = tf.nn.relu(conv1)
    pool1 = tf.nn.max_pool(relu1, ksize=(1,2,2,1), strides=(1,2,2,1), padding="VALID")
    #Note in the tuple defining strides for max_pool, the first entry is always 1 as this refers to the batches/indexed images,
    #rather than the dimensions of a particular image

    conv2 = tf.nn.conv2d(pool1, weights_Binding['conv_W2_bind'], strides=[1,1,1,1], padding="VALID")
    conv2 = tf.nn.bias_add(conv2, biases_Binding['conv_b2_bind'])
    relu2 = tf.nn.relu(conv2)
    pool2 = tf.nn.max_pool(relu2, ksize=(1,2,2,1), strides=(1,2,2,1), padding="VALID")

    #Flatten Pool 2 before connecting it (fully) with the dense layers 1 and 2
    pool2_flat = tf.reshape(pool2, [-1, 5 * 5 * 16])
    
    
    #Output the indeces of the relu2 nodes that were maximally active (i.e. 'drove' the pool2 nodes), and flatten them
    _, binding_indeces = tf.nn.max_pool_with_argmax(relu2, ksize=(1,2,2,1), strides=(1,2,2,1), padding="VALID")
    binding_indeces_flat = tf.reshape(binding_indeces, [-1, 5 * 5 * 16])
    binding_indeces_flat_cast = tf.dtypes.cast(binding_indeces_flat, dtype=tf.int32)

    #Define update (containing the same number of ones as earlier indices), and shape, which form arguments of scatter_nd
    updates_batch = tf.ones_like(binding_indeces_flat_cast)
    shape = tf.constant([10 * 10 * 16])

    tensor_list = [] #Empty list to hold each row-like tensor, one of which will be formed for each image in the batch
    
    for batch_iter in range(temp_batch_size):

        #Add a dimension to enable use of scatter_nd with the flattened tensor
        indices_temp = tf.expand_dims(binding_indeces_flat_cast[batch_iter], axis=1) 
        
        #In a new tensor of shape 'shape', place a one wherever indicated by 'indices_temp'
        scatter = tf.scatter_nd(indices_temp, updates_batch[batch_iter], shape)

        tensor_list.append(scatter)

    #Stack the scatter tensors for each member of the batch into a single tensor of dimension (batch_size, 'shape')
    binding_mask = tf.stack(tensor_list)
    
    #Recast and flatten tensors as necessary
    binding_mask_cast = tf.dtypes.cast(binding_mask, dtype=tf.float32)
    relu2_flat = tf.reshape(relu2, [-1, 10 * 10 * 16])

    #Apply the boolean information about which lower level nodes were active as a mask to the actual activaiton values
    binding_activations = tf.math.multiply(relu2_flat, binding_mask_cast)
    
    dense1 = tf.add(tf.matmul(tf.concat([pool2_flat, binding_activations], axis=1), 
                              weights_Binding['dense_W1_bind']), biases_Binding['dense_b1_bind'])
    dense1 = tf.nn.relu(dense1)
    dense2 = tf.add(tf.matmul(dense1, weights_Binding['dense_W2_bind']), biases_Binding['dense_b2_bind'])
    dense2 = tf.nn.relu(dense2)

    logits = tf.add(tf.matmul(dense2, weights_Binding['output_W_bind']), biases_Binding['output_b_bind'])

    return logits


#Define the training function of the new Binding-CNN
def BindingNet_train(var_list, training_data, training_labels, testing_data, testing_labels, learning_rate):
    
    predictions = cnn_binding_predictions(x, temp_batch_size=128) #NB that x was defined earlier with tf.placeholder
    
    #Define the main Tensors (left hand) and Operations (right hand) that will be used during training
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=y))
    correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    #Create the chosen optimizer with tf.train.Adam..., then add it to the graph with .minimize
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
    #Define values to be written with the summary method for later visualization
    loss_summary = tf.summary.scalar(name="Loss_values", tensor=cost)
    accuracy_summary = tf.summary.scalar(name="Accuracy_values", tensor=accuracy)
    
    saver = tf.train.Saver(var_list)
    
    #Carry out training
    with tf.Session() as sess:
        #Initialize the new variables
        sess.run(tf.global_variables_initializer())
        

        for epoch in range(training_epochs):

            for batch in range(int(len(training_labels)/batch_size)):

                batch_x = training_data[batch*batch_size:min((batch+1)*batch_size, len(training_labels))]
                batch_y = training_labels[batch*batch_size:min((batch+1)*batch_size, len(training_labels))]
                

                #Recall that tf.Session.run is the main method for running a tf.Operation or evaluation a tf.Tensor
                #By passing or more Tensors or Operations, TensorFlow will execute the operations needed
                run_optim = sess.run(optimizer, feed_dict = {x: batch_x, y: batch_y})

                loss, acc = sess.run([cost, accuracy], feed_dict = {x: batch_x, y: batch_y})
                

            print("At iteration " + str(epoch) + ", Loss = " + \
                 "{:.6f}".format(loss) + ", Training Accuracy = " + \
                                "{:.5f}".format(acc))

        print("Training complete")
        
        save_path = saver.save(sess, "/Binding_CNN.ckpt")
        print("Model saved in Binding_CNN.ckpt")
        
        test_acc_list = []
        test_loss_list = []
        
        for batch in range(int(len(testing_labels)/batch_size)):

            batch_x = testing_data[batch*batch_size:min((batch+1)*batch_size, len(testing_labels))]
            batch_y = testing_labels[batch*batch_size:min((batch+1)*batch_size, len(testing_labels))]


            test_acc, test_l = sess.run([accuracy,cost], feed_dict={x: batch_x, y: batch_y})

            test_acc_list.append(test_acc)
            test_loss_list.append(test_l)

        return test_acc_list, test_loss_list


test_acc_list, test_loss_list = BindingNet_train(var_list_Binding, training_data, training_labels, 
                                                 testing_data, testing_labels, learning_rate=0.001)

print("Achieved an accuracy of " + str(np.mean(test_acc_list)))