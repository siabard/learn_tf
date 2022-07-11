# operator.py

import tensorflow as tf

x = tf.constant([[1, 2]])
negMatrix = tf.negative(x)
print(negMatrix)