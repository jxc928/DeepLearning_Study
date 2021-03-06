###
# MNIST: nn-relu, weight-xavier, dropout
# deep: 3 hidden layer
# wide: 512
###
# Accuracy: 0.9819

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
# dropout(keep_prob) rate 0.5~0.7 on training, but shold be 1 in testing
keep_prob = tf.placeholder(tf.float32)

W1 = tf.get_variable("weight1", shape=[784, 512], initializer=tf.contrib.layers.xavier_initializer())
W2 = tf.get_variable("weight2", shape=[512, 512], initializer=tf.contrib.layers.xavier_initializer())
W3 = tf.get_variable("weight3", shape=[512, 512], initializer=tf.contrib.layers.xavier_initializer())
W4 = tf.get_variable("weight4", shape=[512, 512], initializer=tf.contrib.layers.xavier_initializer())
W5 = tf.get_variable("weight5", shape=[512, nb_classes], initializer=tf.contrib.layers.xavier_initializer())

b1 = tf.Variable(tf.random_normal([512]), name='bias1')
b2 = tf.Variable(tf.random_normal([512]), name='bias2')
b3 = tf.Variable(tf.random_normal([512]), name='bias3')
b4 = tf.Variable(tf.random_normal([512]), name='bias4')
b5 = tf.Variable(tf.random_normal([nb_classes]), name='bias5')

# Hypothesis
layer1 = tf.nn.relu(tf.matmul(X, W1) + b1)
layer1 = tf.nn.dropout(layer1, keep_prob=keep_prob)
layer2 = tf.nn.relu(tf.matmul(layer1, W2) + b2)
layer2 = tf.nn.dropout(layer2, keep_prob=keep_prob)
layer3 = tf.nn.relu(tf.matmul(layer2, W3) + b3)
layer3 = tf.nn.dropout(layer3, keep_prob=keep_prob)
layer4 = tf.nn.relu(tf.matmul(layer3, W4) + b4)
layer4 = tf.nn.dropout(layer4, keep_prob=keep_prob)
hypothesis = tf.matmul(layer4, W5) + b5

# Cross entropy - cost/loss function
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
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
            c, _ = sess.run([cost, train], feed_dict={X: batch_xs,
                                                      Y: batch_ys,
                                                      keep_prob: 0.7})
            avg_cost += c / total_batch

        print('Epoch: ', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
    print("Learning Finished!")

    # Test the model using test sets
    print("Accuracy: ", accuracy.eval(session=sess, feed_dict={X: mnist.test.images,
                                                               Y: mnist.test.labels,
                                                               keep_prob: 1}))

    # Get one and predict
    r = random.randint(0, mnist.test.num_examples - 1)
    print("Label: ", sess.run(
        tf.argmax(mnist.test.labels[r:r+1], 1)))
    print("Prediction: ", sess.run(
        tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r:r+1], keep_prob: 1}))
