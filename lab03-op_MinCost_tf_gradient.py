###
# can manually get gradient value within GradientDescentOptimizer method
###

import tensorflow as tf

x_train = [1, 2, 3]
y_train = [1, 2, 3]

W = tf.Variable(5.0)
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# hypothesis for linear model XW
hypothesis = X * W

# Manual gradient
gradient = tf.reduce_mean((W * X - Y) * X) * 2
# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

# Get gradients: gvs - grads and vars
gvs = optimizer.compute_gradients(cost)
# Apply gradients
apply_gradients = optimizer.apply_gradients(gvs)

# Launch the graph in a session
sess = tf.Session()
# Initializes global variables in the graph
sess.run(tf.global_variables_initializer())

for step in range(101):
    print(step, sess.run(gradient, feed_dict={X:x_train, Y: y_train}),
          sess.run(W), sess.run(gvs, feed_dict={X:x_train, Y: y_train}))
    sess.run(apply_gradients, feed_dict={X: x_train, Y: y_train})
