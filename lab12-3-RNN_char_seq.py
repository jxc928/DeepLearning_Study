###
# RNN: char -> sequence
#
###
# 'if you want you'(sample) --> 'f you want you'(predict result)

import tensorflow as tf
import numpy as np

sample = "if you want you"
idx2char = list(set(sample))  # index -> char
print('idx2char: ', idx2char)
char2idx = {c: i for i, c in enumerate(idx2char)}  # char -> idx
print('char2idx: ', char2idx)

# hyper parameters
dic_size = len(char2idx)  # RNN input size (one-hot size)
hidden_size = len(char2idx)  # RNN output size
num_class = len(char2idx)  # final output size (RNN or softmax, etc.)
batch_size = 1  # one sample data, one batch
sequence_length = len(sample) - 1  # number of lstm rollings (unit #)
learning_rate = 0.1

# dic & input/output data
sample_idx = [char2idx[c] for c in sample]  # char to index
x_data = [sample_idx[:-1]]  # X data sample (0 ~ n-1) hello: hell
y_data = [sample_idx[1:]]  # Y data sample (1 ~ n) hello: ello
print('x_data: ', x_data)
print('y_data: ', y_data)

# input/output placeholder
X = tf.placeholder(tf.int32, [None, sequence_length])  # X data
Y = tf.placeholder(tf.int32, [None, sequence_length])  # Y label
# One-hot encoding
X_one_hot = tf.one_hot(X, num_class)  # one-hot: 1 -> 0 1 0 0 0 0 0 0 0 0
print('X_one_hot.shape: ', X_one_hot.shape)  # check out the shape

# RNN model
cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
initial_state = cell.zero_state(batch_size, tf.float32)
outputs, _states = tf.nn.dynamic_rnn(cell, X_one_hot, initial_state=initial_state, dtype=tf.float32)

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
        l, _ = sess.run([loss, train], feed_dict={X: x_data, Y: y_data})
        result = sess.run(prediction, feed_dict={X: x_data})

        # print char using dic
        result_str = [idx2char[c] for c in np.squeeze(result)]
        print(i, "loss:", l, "prediction: ", ''.join(result_str))
