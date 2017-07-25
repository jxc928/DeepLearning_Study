###
# RNN
# Teach hello: hihell -> ihello
###

import tensorflow as tf
import numpy as np

idx2char = ['h', 'i', 'e', 'l', 'o']  # h=0, i=1, e=2, l=3, o=4

# input/output data
x_data = [[0, 1, 0, 2, 3, 3]]         # hihell
x_one_hot = [[[1, 0, 0, 0, 0],        # h 0
              [0, 1, 0, 0, 0],        # i 1
              [1, 0, 0, 0, 0],        # h 0
              [0, 0, 1, 0, 0],        # e 2
              [0, 0, 0, 1, 0],        # l 3
              [0, 0, 0, 1, 0],]]      # l 3

y_data = [[1, 0, 2, 3, 3, 4]]         # ihello

# hyper parameters
num_classes = 5
input_dim = 5  # one-hot size
hidden_size = 5  # output from the LSTM. 5 to directly predict one-hot
batch_size = 1   # one sentence
sequence_length = 6  # |ihello| == 6
learning_rate = 0.1

# input/output placeholders
X = tf.placeholder(tf.float32, [None, sequence_length, input_dim])  # X one-hot
Y = tf.placeholder(tf.int32, [None, sequence_length])  # Y label

# RNN model
cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
initial_state = cell.zero_state(batch_size, tf.float32)
outputs, _states = tf.nn.dynamic_rnn(cell, X, initial_state=initial_state, dtype=tf.float32)

# initialize weights
weights = tf.ones([batch_size, sequence_length])

# define cost/loss & optimizer
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

#
prediction = tf.argmax(outputs, axis=2)

# Launch graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(50):
        l, _ = sess.run([loss, train], feed_dict={X: x_one_hot, Y: y_data})
        result = sess.run(prediction, feed_dict={X: x_one_hot})
        print(i, "loss:", l, "prediction: ", result, "true Y: ", y_data)

        # print char using dic
        result_str = [idx2char[c] for c in np.squeeze(result)]
        print("\tPrediction str: ", ''.join(result_str))
