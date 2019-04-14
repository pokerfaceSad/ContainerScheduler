import tensorflow as tf
import numpy as np

var1 = tf.Variable(np.array([[1, 2, 3], [1, 2, 3]]), dtype=tf.float32)
# var1 = tf.Variable(np.array([1, 2, 3]), dtype=tf.float32)

sf = tf.nn.softmax(var1)
am = tf.argmax(var1, 1)
prob = tf.contrib.distributions.Categorical(probs=sf)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    # result = sess.run(prob.sample())
    print(sess.run(tf.shape(var1)))
    print(sess.run(am))
    # print(result)
