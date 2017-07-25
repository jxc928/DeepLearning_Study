import tensorflow as tf

x_train = [1, 2, 3]
y_train = [1, 2, 3]

W = tf.Variable(-3.0)
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# hypothesis for linear model XW
hypothesis = X * W

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize: Gradient Descent Magic
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

# Launch the graph in a session
sess = tf.Session()
# Initializes global variables in the graph
sess.run(tf.global_variables_initializer())

for step in range(101):
    print(step, sess.run(cost, feed_dict={X:x_train, Y: y_train}), sess.run(W))
    sess.run(train, feed_dict={X: x_train, Y: y_train})
