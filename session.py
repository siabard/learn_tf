# session.py

import tensorflow as tf

x = tf.constant([[1., 2.]])

@tf.function
def forward(tfc):
    return tf.negative(tfc)

print(forward(x))