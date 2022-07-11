import tensorflow as tf

string = tf.Variable("this is a string", tf.string)
number = tf.Variable(324, tf.int16)

print(string)
print(number)