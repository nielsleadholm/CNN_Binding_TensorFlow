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

#Visualise some random examples from the dataset, as well as print the associated one-hot encoding
example_iter = 8
example_img = np.reshape(training_data[example_iter, :, :], (28,28))
plt.imshow(example_img, cmap='gray')

training_labels[example_iter, :]

#Define training parameters
batch_size = 128
training_epochs = 5

#Declare placeholders for the input features and labels
#The first dimension of the palceholder shape is set to None as this will later be defined by the batch size

x = tf.placeholder(training_data.dtype, [None, 28, 28, 1])
y = tf.placeholder(training_labels.dtype, [None, num_classes])

#Define weight and bias variables, and initialize values 
#Note for example that the first convolutional weights layer has a 5x5 filter with 1 input channel, and 6 output channels
#tf.get_variable will either get an existing variable with these parameters, or otherwise create a new one
weights = {
    'conv_W1' : tf.get_variable('CW1', shape=(5, 5, 1, 6), initializer=tf.contrib.layers.xavier_initializer()),
    'conv_W2' : tf.get_variable('CW2', shape=(5, 5, 6, 16), initializer=tf.contrib.layers.xavier_initializer()),
    'dense_W1' : tf.get_variable('DW1', shape=(400, 120), initializer=tf.contrib.layers.xavier_initializer()),
    'dense_W2' : tf.get_variable('DW2', shape=(120, 84), initializer=tf.contrib.layers.xavier_initializer()),
    'output_W' : tf.get_variable('OW', shape=(84, num_classes), initializer=tf.contrib.layers.xavier_initializer())
}

biases = {
    'conv_b1' : tf.get_variable('Cb1', shape=(6), initializer=tf.contrib.layers.xavier_initializer()),
    'conv_b2' : tf.get_variable('Cb2', shape=(16), initializer=tf.contrib.layers.xavier_initializer()),
    'dense_b1' : tf.get_variable('Db1', shape=(120), initializer=tf.contrib.layers.xavier_initializer()),
    'dense_b2' : tf.get_variable('Db2', shape=(84), initializer=tf.contrib.layers.xavier_initializer()),
    'output_b' : tf.get_variable('Ob', shape=(num_classes), initializer=tf.contrib.layers.xavier_initializer())
}

#Define the convolutional model

def cnn_predictions(features):

    conv1 = tf.nn.conv2d(input=tf.dtypes.cast(features, dtype=tf.float32), filter=weights['conv_W1'], strides=[1, 1, 1, 1], padding="SAME")
    conv1 = tf.nn.bias_add(conv1, biases['conv_b1'])
    relu1 = tf.nn.relu(conv1)
    pool1 = tf.nn.max_pool(relu1, ksize=(1,2,2,1), strides=(1,2,2,1), padding="VALID")
    #Note in the tuple defining strides for max_pool, the first entry is always 1 as this refers to the batches/indexed images,
    #rather than the dimensions of a particular image

    conv2 = tf.nn.conv2d(pool1, weights['conv_W2'], strides=[1,1,1,1], padding="VALID")
    conv2 = tf.nn.bias_add(conv2, biases['conv_b2'])
    relu2 = tf.nn.relu(conv2)
    pool2 = tf.nn.max_pool(relu2, ksize=(1,2,2,1), strides=(1,2,2,1), padding="VALID")

    #Flatten Pool 2 before connecting it (fully) with the dense layers 1 and 2
    pool2_flat = tf.reshape(pool2, [-1, 5 * 5 * 16])
    dense1 = tf.add(tf.matmul(pool2_flat, weights['dense_W1']), biases['dense_b1'])
    dense1 = tf.nn.relu(dense1)
    dense2 = tf.add(tf.matmul(dense1, weights['dense_W2']), biases['dense_b2'])
    dense2 = tf.nn.relu(dense2)

    logits = tf.add(tf.matmul(dense2, weights['output_W']), biases['output_b'])
    #classes = tf.argmax(logits, axis=1)
    #probabilities = tf.nn.softmax(logits)
  
    return logits

#Model setup
def train_CNN(training_data, training_labels, testing_data, testing_labels, learning_rate):
    
    
    predictions = cnn_predictions(x)
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=y))
  
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
 

    correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(y, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_loss = []
        test_loss = []
        train_accuracy = []
        test_accuracy = []

        write_summary_to_file = tf.summary.FileWriter('./Summary', sess.graph)

        for epoch in range(training_epochs):

            for batch in range(int(len(training_labels)/batch_size)):

                batch_x = training_data[batch*batch_size:min((batch+1)*batch_size, len(training_labels))]
                batch_y = training_labels[batch*batch_size:min((batch+1)*batch_size, len(training_labels))]
                

                run_optim = sess.run(optimizer, feed_dict = {x: batch_x, y: batch_y})

                loss, acc = sess.run([cost, accuracy], feed_dict = {x: batch_x, y: batch_y})

            print("At iteration " + str(epoch) + ", Loss = " + \
                 "{:.6f}".format(loss) + ", Training Accuracy = " + \
                                "{:.5f}".format(acc))

        print("Training complete")

        test_acc, test_l = sess.run([accuracy,cost], feed_dict={x: testing_data, y: testing_labels})

        train_loss.append(loss)

        test_loss.append(test_l)

        train_accuracy.append(acc)

        test_accuracy.append(test_acc)

        print("Testing Accuracy:","{:.5f}".format(test_acc))


        write_summary_to_file.close()

    return 0

train_CNN(training_data, training_labels, testing_data, testing_labels, learning_rate=0.001)