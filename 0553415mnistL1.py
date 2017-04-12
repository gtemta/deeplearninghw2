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
batch_size = 300
display_step = 20
beta = 0.01

#Network Parameter
n_input = 784 #  data input  28*28
n_classes = 10 # classes 0-9

#set TF IO
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])

#layers weight & bias
weights = {
    'wd1': tf.Variable(tf.random_normal([n_input, 540])),
    'wd2': tf.Variable(tf.random_normal([540, 360])),
    'out': tf.Variable(tf.random_normal([360, n_classes]))
}

biases = {
    'bd1': tf.Variable(tf.random_normal([540])),
    'bd2': tf.Variable(tf.random_normal([360])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

#define model
def DNN(x,weights,biases):
    # Fully connected layer
    fc1 = tf.add(tf.matmul(x,weights['wd1']),biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Fully connected layer2
    fc2 = tf.add(tf.matmul(fc1,weights['wd2']),biases['bd2'])
    fc2 = tf.nn.relu(fc2)
    #Outputlayer  
    result = tf.add(tf.matmul(fc2, weights['out']), biases['out'])
    return result 

# construct model
predict = DNN(x,weights,biases)
regulizer = 0
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
    loss,acc = sess.run([cost, accuracy], feed_dict={x: batch_x,y: batch_y}) 
    print( "Training Accuracy= " + \
                  "{:.5f}".format(acc))
    # Calculate accuracy for 256 mnist test images
    print( "Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: mnist.test.images[:256],
                                      y: mnist.test.labels[:256]}))
    we_fc1 = weights['wd1'].eval();
    we_fc2 = weights['wd2'].eval();
    we_out = weights['out'].eval();
    bia_fc1 = biases['bd1'].eval();
    bia_fc2 = biases['bd2'].eval();
    bia_out = biases['out'].eval();

import matplotlib.pyplot as plt
plt.gca().set_color_cycle(['red','blue'])
plt.plot(trlosses,label = 'train')
plt.plot(telosses,label = 'test')
plt.legend(['train','test'])
plt.show()

plt.hist(we_fc1.flatten(),color = 'blue',bins=50)
plt.xlabel('Value')
plt.ylabel('Numbers')
plt.show()