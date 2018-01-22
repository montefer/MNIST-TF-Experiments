# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 19:27:52 2017

@author: Fernando
"""

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


    #Import TensorFlow
import tensorflow as tf

    #Create a tensor, [] specifies the shape of the tensor
    #Note that 'None' part of the placeholder refers to the number of rows that can be present in the tensor, which is not specified,
    #so there can be any number of rows. '784' refers to the number of column entries there are.
x = tf.placeholder(tf.float32, [None, 784])

    #Creates ten tensors of dimension 784 for each digit to calculate weights
    #Note: that tensors are written as an array along a single row, with all the arrays stacked upon each other
W = tf.Variable(tf.zeros([784, 10]))

    #Creates ten tensors of size none (so a single number, not a vector or array) to determine a bias that is added to the output
b = tf.Variable(tf.zeros([10]))

    #Multiply the placeholder variable (x) by the weight (W), then add a bias (b), then perform softmax on it
y = tf.nn.softmax(tf.matmul(x, W) + b)

    #Array for the actual value of the image, stored as a 1 in the index corresponding to the number.
y_ = tf.placeholder(tf.float32, [None, 10])

    #Some weird-ass bullshit I don't actually understand at the moment - I should probably study this
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

    #Each loop, from the training set batch, we draw 100 random points to train with
for _ in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    #Determine the accuracy of the model. tf.argmax(y,1) returns the an array for the index of the highest value on each row
    #which has ten places. The array (y) [array of probabilities that the tensor is one of the numbers] has various values
    #from 0-1, pulls the largest of them and returns the index, which is then compared to the index of the actual value for the
    #image.
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

    #time for original code. hopefully this can tell me which numbers are incorrect the most, followed by what they're being confused with
#what_we_think = tf.argmax(y,1)#.index("1")
#what_it_is = tf.argmax(y_,1).index("1")
    #print(what_we_think)
    #print(what_it_is)


accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("Model Prediction Accuracy:",("{0:.2f}%".format(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})*100)))
#print(tf.TensorShape(what_we_think))
  