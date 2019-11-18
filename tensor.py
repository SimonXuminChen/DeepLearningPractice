import tensorflow as tf

mammal = tf.Variable("Elephant",tf.string)
ignition = tf.Variable(451,tf.int16)
floating = tf.Variable(3.1415926,tf.float64)
its_complicated = tf.Variable(12.3-4.85j,tf.complex64)

a = [mammal,ignition,floating,its_complicated]

mystr = tf.Variable(["hello","world"],tf.string)



print(mystr)