# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 19:41:01 2017

@author: Pika
"""
import tensorflow as tf

#import MNIST DATA
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#define  parameter
lrate = 0.001
iters = 200000
batch_size = 128
display_step = 10
beta = 0.01

#Network Parameter
n_input = 784 #  data input  28*28
n_classes = 10 # classes 0-9
stride = 1

#set TF IO
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])

#layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wcon1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wcon2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bcon1': tf.Variable(tf.random_normal([32])),
    'bcon2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


#Convultion
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)
#Poolong
def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

#define model
def conv_NN(x,weights,biases):
    #ResizeDatato28*28
    x = tf.reshape(x, shape=[-1, 28, 28, 1])
    # Convolution Layer
    conv1 = conv2d(x, weights['wcon1'], biases['bcon1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wcon2'], biases['bcon2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    ##DROPOUT     fc1 = tf.nn.dropout(fc1, dropout)
    result =  tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return result 


predict = conv_NN(x,weights,biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=y))
#L1
for w in weights:
    regularizer += tf.reduce_sum(tf.abs(weights[w]))
cost = tf.reduce_mean(cost + beta * regularizer)

optimizer = tf.train.AdamOptimizer(learning_rate=lrate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(predict, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# Initializing the variables
init = tf.global_variables_initializer()
trlosses = list()
telosses = list()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y}) 
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,y: batch_y})
            print( "Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
            trlosses.append((1-acc))
            teloss, teacc = sess.run([cost, accuracy], feed_dict={x: mnist.test.images[:256],y: mnist.test.labels[:256]})
            telosses.append((1-teacc))
        step += 1
    print ("Optimization Finished!")

    # Calculate accuracy for 256 mnist test images
    print( "Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: mnist.test.images[:256],
                                      y: mnist.test.labels[:256]}))
    we_con1 = weights['wcon1'].eval();
    we_con2 = weights['wcon2'].eval();
    we_fc = weights['wd1'].eval();
    we_out = weights['out'].eval();
    
    bia_con1 = biases['bcon1'].eval();
    bia_con2 = biases['bcon2'].eval();
    bia_fc = biases['bd1'].eval();
    bia_out = biases['out'].eval();

