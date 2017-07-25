###
# same code with lab11-3-CNN_MNIST_deep.py, but in 'class' way
###
# Accuracy: 0.9939

import tensorflow as tf
import random
from tensorflow.examples.tutorials.mnist import input_data
# Check out https://www.tensorflow.org/get_started/mnist/beginners
# for more information about the mnist dataset
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# hyper parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100


class Model:

    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name):
            # dropout(keep_prob) rate 0.5~0.7 on training, but shold be 1 in testing
            self.keep_prob = tf.placeholder(tf.float32)

            # input/output placeholders
            self.X = tf.placeholder(tf.float32, [None, 784])
            X_img = tf.reshape(self.X, [-1, 28, 28, 1])  # 28x28x1(black/white); -1 ==> default
            self.Y = tf.placeholder(tf.float32, [None, 10])

            ##
            # layer1 imgIn shape=(?, 28, 28, 1)
            ##
            # (3, 3, 1, 32): filter size(3, 3) ; input filter(color) 1 ; output filters 32
            W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
            # Conv -> (?, 28, 28, 32)
            # Pool -> (?, 14, 14, 32)
            layer1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
            layer1 = tf.nn.relu(layer1)
            #  A list of ints that has length >= 4. The size of the window for each dimension of the input tensor.
            layer1 = tf.nn.max_pool(layer1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            layer1 = tf.nn.dropout(layer1, keep_prob=self.keep_prob)

            ##
            # layer2 imgIn shape=(?, 14, 14, 32)
            ##
            # (3, 3, 32, 64): filter size(3, 3) ; input filters 32 ; output filters 64
            W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
            # Conv -> (?, 14, 14, 64)
            # Pool -> (?, 7, 7, 64)
            layer2 = tf.nn.conv2d(layer1, W2, strides=[1, 1, 1, 1], padding='SAME')
            layer2 = tf.nn.relu(layer2)
            layer2 = tf.nn.max_pool(layer2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            layer2 = tf.nn.dropout(layer2, keep_prob=self.keep_prob)

            ##
            # layer3 imgIn shape=(?, 7, 7, 64)
            ##
            W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
            # Conv -> (?, 7, 7, 128)
            # Pool -> (?, 4, 4, 128)
            #       why (4, 4)?('SAME')
            #       out_height = ceil(float(in_height) / float(strides[1]))
            #       out_width = ceil(float(in_width) / float(strides[2]))
            layer3 = tf.nn.conv2d(layer2, W3, strides=[1, 1, 1, 1], padding='SAME')
            layer3 = tf.nn.relu(layer3)
            layer3 = tf.nn.max_pool(layer3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            layer3 = tf.nn.dropout(layer3, keep_prob=self.keep_prob)
            # reshape to pass to fully-connected layer
            layer3 = tf.reshape(layer3, [-1, 4 * 4 * 128])

            ##
            # first FC: 4x5x128 inputs -> 625 outputs
            ##
            W4 = tf.get_variable("W4", shape=[4 * 4 * 128, 625], initializer=tf.contrib.layers.xavier_initializer())
            b4 = tf.Variable(tf.random_normal([625]))
            layer4 = tf.nn.relu(tf.matmul(layer3, W4) + b4)
            layer4 = tf.nn.dropout(layer4, keep_prob=self.keep_prob)

            ##
            # final FC: 625 inputs -> 10 outputs
            ##
            W5 = tf.get_variable("W5", shape=[625, 10], initializer=tf.contrib.layers.xavier_initializer())
            b5 = tf.Variable(tf.random_normal([10]))
            # Hypothesis
            self.hypothesis = tf.matmul(layer4, W5) + b5

        # define cost/loss & optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.hypothesis, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)

        # Accuracy computation
        correct_prediction = tf.equal(tf.argmax(self.hypothesis, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def predict(self, x_test, keep_prop=1.0):
        return self.sess.run(self.hypothesis, feed_dict={self.X: x_test,
                                                         self.keep_prob: keep_prop})

    def get_accuracy(self, x_test, y_test, keep_prop=1.0):
        return self.sess.run(self.accuracy, feed_dict={self.X: x_test,
                                                       self.Y: y_test,
                                                       self.keep_prob: keep_prop})

    def train(self, x_data, y_data, keep_prop=0.7):
        return self.sess.run([self.cost, self.optimizer], feed_dict={self.X: x_data,
                                                                 self.Y: y_data,
                                                                 self.keep_prob: keep_prop})

# Launch graph
with tf.Session() as sess:
    m1 = Model(sess, "m1")

    # Initialize Tensorflow variables
    sess.run(tf.global_variables_initializer())
    # Training cycle
    print("Start Learning.....")
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            c, _ = m1.train(batch_xs, batch_ys)
            avg_cost += c / total_batch
        print('Epoch: ', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

    print("Learning Finished!")

    # Test the model and check accuracy
    print('Accuracy:', m1.get_accuracy(mnist.test.images, mnist.test.labels))

    # Get one and predict
    r = random.randint(0, mnist.test.num_examples - 1)
    print("Label: ", sess.run(
        tf.argmax(mnist.test.labels[r:r + 1], 1)))
    print("Prediction: ", sess.run(
        tf.argmax(m1.predict(mnist.test.images[r:r + 1]), 1)))
