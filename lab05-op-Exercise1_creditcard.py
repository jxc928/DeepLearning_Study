###
# Exercise - Credit Card Fraud Detection
# Using - tf.TextLineReader, tf.decode_csv
# DataSet: https://www.kaggle.com/dalpozz/creditcardfraud?
###
## not working well: cost and weight always gets nan

import tensorflow as tf
import numpy as np

filename_queue = tf.train.string_input_producer(
    ['data-creditcard_binaClass.csv'], shuffle=False, name='filename_queue')

reader = tf.TextLineReader(skip_header_lines=1)  # skip header line
key, value = reader.read(filename_queue)
print(key, value)

# Default values, in case of empty columns. Also specifies the type of the decoded result.
# record_defaults = [[0.], [0.], [0.], [0.]]
record_defaults = [[0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.],
                   [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.],
                   [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.],
                   [0.]]
print(record_defaults)
xy = tf.decode_csv(value, record_defaults=record_defaults)

# collect batches of csv in
# time, V1 ~ V28, Amount, class
train_x_batch, train_y_batch = tf.train.batch([xy[1:6], xy[-1:]], batch_size=1000)  # V1 ~ V5, class
print(train_x_batch)
print(train_y_batch)

# placeholders for a tensor that will be always fed
X = tf.placeholder(tf.float32, shape=[None, 5])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([5, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis using sigmoid: tf.div(1., 1. + tf.exp(tf.matmul(X, W)))
hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

# cost/loss function
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=1e-7).minimize(cost)

# Accuracy computation
# True if hypothesis>0.5 else False
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

# Launch graph
with tf.Session() as sess:
    # Initialize Tensorflow variables
    sess.run(tf.global_variables_initializer())

    # Start populating the filename queue
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for step in range(1001):
        x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
        cost_val, _ = sess.run([cost, train], feed_dict={X: x_batch, Y: y_batch})
        if step % 20 == 0:
            print(step, cost_val)

    coord.request_stop()
    coord.join(threads)

    # Accuracy report
    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: x_batch, Y: y_batch})
    print("\nHypothesis: ", h, "\nCorrect(Y): ", c, "\nAccuracy: ", a)
