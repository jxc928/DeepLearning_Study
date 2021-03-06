###
# MNIST: nn-relu, weight-xavier
# deep: 1 hidden layer
# wide: 256
###
# Accuracy: 0.9743

import tensorflow as tf
import random
from tensorflow.examples.tutorials.mnist import input_data
# Check out https://www.tensorflow.org/get_started/mnist/beginners
# for more information about the mnist dataset
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

nb_classes = 10

# MNIST data image of shape 28 * 28 = 784
X = tf.placeholder(tf.float32, [None, 784])
# 0-9 digits recognition = 10 classes
Y = tf.placeholder(tf.float32, [None, nb_classes])

W1 = tf.get_variable("weight1", shape=[784, 256], initializer=tf.contrib.layers.xavier_initializer())
W2 = tf.get_variable("weight2", shape=[256, 256], initializer=tf.contrib.layers.xavier_initializer())
W3 = tf.get_variable("weight3", shape=[256, nb_classes], initializer=tf.contrib.layers.xavier_initializer())

b1 = tf.Variable(tf.random_normal([256]), name='bias1')
b2 = tf.Variable(tf.random_normal([256]), name='bias2')
b3 = tf.Variable(tf.random_normal([nb_classes]), name='bias3')

# Hypothesis
layer1 = tf.nn.relu(tf.matmul(X, W1) + b1)
layer2 = tf.nn.relu(tf.matmul(layer1, W2) + b2)
hypothesis = tf.matmul(layer2, W3) + b3

# Cross entropy - cost/loss function
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
train = optimizer.minimize(cost)

# Test model
is_correct = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

## parameters
# one epoch = one forward pass and one backward pass of all the training examples
# batch size = the number of training examples in one forward/backward pass.
training_epochs = 15
batch_size = 100

# Launch graph
with tf.Session() as sess:
    # Initialize Tensorflow variables
    sess.run(tf.global_variables_initializer())
    # Training cycle
    print("Learning Start: ")
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            c, _ = sess.run([cost, train], feed_dict={X: batch_xs, Y: batch_ys})
            avg_cost += c / total_batch

        print('Epoch: ', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
    print("Learning Finished!")

    # Test the model using test sets
    print("Accuracy: ", accuracy.eval(session=sess, feed_dict={X: mnist.test.images,
                                                               Y: mnist.test.labels}))

    # Get one and predict
    r = random.randint(0, mnist.test.num_examples - 1)
    print("Label: ", sess.run(
        tf.argmax(mnist.test.labels[r:r+1], 1)))
    print("Prediction: ", sess.run(
        tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r:r+1]}))
