###
# same code with lab11-5-CNN_MNIST_layers.py, but in 'ensemble' way
# Ensemble: create and predicted by several individual neural networks
#           and average(sum in this code) their outputs finally
###
# Accuracy: 0.9951

import tensorflow as tf
import random
import numpy as np
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
            # dropout(keep_prob) rate 0.5~0.7 on training(True), but shold be 1 in testing(False)
            # tf.layers.dropout()
            #   - rate: The dropout rate, between 0 and 1.
            #       E.g. "rate=0.1" would drop out 10% of input units.
            self.training = tf.placeholder(tf.bool)

            # input/output placeholders
            self.X = tf.placeholder(tf.float32, [None, 784])
            X_img = tf.reshape(self.X, [-1, 28, 28, 1])  # 28x28x1(black/white); -1 ==> default
            self.Y = tf.placeholder(tf.float32, [None, 10])

            # Convolutional Layer #1
            conv1 = tf.layers.conv2d(inputs=X_img, filters=32, kernel_size=[3, 3],
                                     padding='SAME', activation=tf.nn.relu)

            # Pooling Layer #1
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2],
                                            padding='SAME', strides=2)
            dropout1 = tf.layers.dropout(inputs=pool1, rate=0.3, training=self.training)

            # Convolutional Layer #2 and Pooling Layer #2
            conv2 = tf.layers.conv2d(inputs=dropout1, filters=64, kernel_size=[3, 3],
                                     padding='SAME', activation=tf.nn.relu)
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2],
                                            padding='SAME', strides=2)
            dropout2 = tf.layers.dropout(inputs=pool2, rate=0.3, training=self.training)

            # Convolutional Layer #3 and Pooling Layer #3
            conv3 = tf.layers.conv2d(inputs=dropout2, filters=128, kernel_size=[3, 3],
                                     padding='SAME', activation=tf.nn.relu)
            pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2],
                                            padding='SAME', strides=2)
            dropout3 = tf.layers.dropout(inputs=pool3, rate=0.3, training=self.training)

            # Dense Layer with Relu
            flat = tf.reshape(dropout3, [-1, 4 * 4 * 128])
            dense4 = tf.layers.dense(inputs=flat, units=625, activation=tf.nn.relu,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     bias_initializer=tf.random_normal_initializer())
            dropout4 = tf.layers.dropout(inputs=dense4, rate=0.3, training=self.training)

            # Logits(no activation) Layer: final FC == 625 inputs -> 10 outputs
            self.hypothesis = tf.layers.dense(inputs=dropout4, units=10,
                                              kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                              bias_initializer=tf.random_normal_initializer())

        # define cost/loss & optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.hypothesis, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)

        # Accuracy computation
        correct_prediction = tf.equal(tf.argmax(self.hypothesis, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def predict(self, x_test, training=False):
        return self.sess.run(self.hypothesis, feed_dict={self.X: x_test,
                                                         self.training: training})

    def get_accuracy(self, x_test, y_test, training=False):
        return self.sess.run(self.accuracy, feed_dict={self.X: x_test,
                                                       self.Y: y_test,
                                                       self.training: training})

    def train(self, x_data, y_data, training=True):
        return self.sess.run([self.cost, self.optimizer], feed_dict={self.X: x_data,
                                                                 self.Y: y_data,
                                                                 self.training: training})

# Launch graph
with tf.Session() as sess:
    # create several models
    models = []
    num_models = 3
    for m in range(num_models):
        models.append(Model(sess, "model" + str(m)))

    # Initialize Tensorflow variables
    sess.run(tf.global_variables_initializer())
    # Training cycle
    print("Start Learning.....")
    for epoch in range(training_epochs):
        avg_cost_list = np.zeros(len(models))
        total_batch = int(mnist.train.num_examples / batch_size)
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            for m_idx, m in enumerate(models):
                c, _ = m.train(batch_xs, batch_ys)
                avg_cost_list[m_idx] += c / total_batch
        print('Epoch: ', '%04d' % (epoch + 1), 'cost =', avg_cost_list)

    print("Learning Finished!")

    # Test the model and check accuracy
    test_size = len(mnist.test.labels)
    predictions = np.zeros(test_size * 10).reshape(test_size, 10)

    for m_idx, m in enumerate(models):
        print(m_idx, 'Accuracy:', m.get_accuracy(mnist.test.images, mnist.test.labels))
        p = m.predict(mnist.test.images)
        predictions += p

    ensemble_correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(mnist.test.labels, 1))
    ensemble_accuracy = tf.reduce_mean(tf.cast(ensemble_correct_prediction, tf.float32))
    print('Ensemble accuracy:', sess.run(ensemble_accuracy))

    # Get one and predict
    r = random.randint(0, mnist.test.num_examples - 1)
    ensemble_prediction = np.zeros(1 * 10).reshape(1, 10)
    print("Label: ", sess.run(
        tf.argmax(mnist.test.labels[r:r + 1], 1)))
    for m_idx, m in enumerate(models):
        p = m.predict(mnist.test.images[r:r + 1])
        ensemble_prediction += p

    print("Prediction: ", sess.run(tf.argmax(ensemble_prediction, 1)))
