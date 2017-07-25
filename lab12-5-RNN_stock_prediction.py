###
# Stock daily: RNN
###

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def minMaxScaler(data):
    """Min Max Normalization

    Parameters
    ----------
    data: numpy.ndarray
        input data to be normalized
        shape: [Batch size, dimension]

    Return
    ----------
    data: num.ndarray
        normalized data
        shape: [Batch size, dimension]

    References
    ----------
    [1] http://sebastianraschka.com/Articles/2014_about_feature_scaling.html
    """
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)

# train parameters
timesteps = seq_length = 7
data_dim = 5
hidden_dim = 10
output_dim = 1
learning_rate = 0.01
iterations = 1001

# open, high, low, volume, close
xy = np.loadtxt('data-04-stock_daily.csv', delimiter=',')
xy = xy[::-1]  # reverse order (chronically ordered)
xy = minMaxScaler(xy)
x = xy
y = xy[:, [-1]]  # close as label

# build a dataset
dataX = []
dataY = []
for i in range(0, len(y) - seq_length):
    _x = x[i:i + seq_length]
    _y = y[i + seq_length]  # Next close price
    print(_x, "->", _y)
    dataX.append(_x)
    dataY.append(_y)

# train/test split
train_size = int(len(dataY) * 0.7)
test_size = len(dataY) - train_size
trainX, testX = np.array(dataX[0:train_size]), np.array(dataX[train_size:len(dataX)])
trainY, testY = np.array(dataY[0:train_size]), np.array(dataY[train_size:len(dataY)])

# input/output placeholders
X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
Y = tf.placeholder(tf.float32, [None, 1])

# RNN model
cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh)
outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
# Fully_connected layer
Y_pred = tf.contrib.layers.fully_connected(
    outputs[:, -1], output_dim, activation_fn=None)  # just use the last cell's output

# cost/loss
loss = tf.reduce_sum(tf.square(Y_pred - Y))  # sum of the squares
# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(loss)

# Launch graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(iterations):
        _, step_loss = sess.run([train, loss], feed_dict={X: trainX, Y: trainY})
        print(step, step_loss)

    # test
    testPredict = sess.run(Y_pred, feed_dict={X: testX})

    # Plot predictions
    plt.plot(testY)
    plt.plot(testPredict)
    plt.xlabel("Time Period")
    plt.ylabel("Stock Price")
    plt.show()
