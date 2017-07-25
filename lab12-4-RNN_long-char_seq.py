###
# RNN: sentence -> sequence
# (optional) Fully-connected layer: softmax, etc.
#    - it is more useful
###
# "if you want to build a ship, don't drum up people together to "
# "collect wood and don't assign them tasks and work, but rather "
# "teach them to long for the endless immensity of the sea."

import tensorflow as tf
import numpy as np

sentence = ("if you want to build a ship, don't drum up people together to "
            "collect wood and don't assign them tasks and work, but rather "
            "teach them to long for the endless immensity of the sea.")

char_set = list(set(sentence))  # sentence set to list
char_dic = {w: i for i, w in enumerate(char_set)}

# hyper parameters
data_dim = len(char_set)  # RNN input size (one-hot size)
hidden_size = len(char_set)  # RNN output size
num_class = len(char_set)  # final output size (RNN or softmax, etc.)
seq_length = 10  # sequence length: any arbitrary number
learning_rate = 0.1

# make dataset
dataX = []
dataY = []
for i in range(0, len(sentence)-seq_length):
    x_str = sentence[i:i + seq_length]
    y_str = sentence[i + 1:i + seq_length + 1]
    # print(i, x_str, '-->', y_str)

    x = [char_dic[c] for c in x_str]  # x_str to index
    y = [char_dic[c] for c in y_str]  # y_str to index

    dataX.append(x)
    dataY.append(y)

batch_size = len(dataX)  # batch size

# input/output placeholder
X = tf.placeholder(tf.int32, [None, seq_length])  # X data
Y = tf.placeholder(tf.int32, [None, seq_length])  # Y label
# One-hot encoding
X_one_hot = tf.one_hot(X, num_class)  # one-hot: 1 -> 0 1 0 0 0 0 0 0 0 0
print('X_one_hot.shape: ', X_one_hot.shape)  # check out the shape

# RNN model
cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
cells = tf.contrib.rnn.MultiRNNCell([cell] * 2, state_is_tuple=True)
initial_state = cells.zero_state(batch_size, tf.float32)
outputs, _states = tf.nn.dynamic_rnn(cells, X_one_hot, initial_state=initial_state, dtype=tf.float32)

# (optional) softmax layer
X_for_softmax = tf.reshape(outputs, [-1, hidden_size])

softmax_w = tf.get_variable("softmax_w", [hidden_size, num_class])
softmax_b = tf.get_variable("softmax_b", [num_class])
outputs = tf.matmul(X_for_softmax, softmax_w) + softmax_b
# reshape out for sequence_loss
outputs = tf.reshape(outputs, [batch_size, seq_length, num_class])

# initialize weights (All weights are 1 : equal weights)
weights = tf.ones([batch_size, seq_length])

# define cost/loss & optimizer
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

#
prediction = tf.argmax(outputs, axis=2)

# Launch graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(500):
        _, l, results = sess.run([train, loss, outputs], feed_dict={X: dataX, Y: dataY})

        for j, result in enumerate(results):
            index = np.argmax(result, axis=1)
            print(i, j, ''.join([char_set[t] for t in index]), l)

    # print the last char of each result to check it works
    results = sess.run(outputs, feed_dict={X: dataX})
    for j, result in enumerate(results):
        index = np.argmax(result, axis=1)
        if j is 0:  # print all for the first result to make a sentence
            print(''.join([char_set[t] for t in index]), end='')
        else:
            print(char_set[index[-1]], end='')
