import tensorflow as tf


# model of session
x1 = tf.constant(5)
x2 = tf.constant(6)
result = tf.multiply(x1,x2)


# actually runnign a model that we just defined
with tf.Session() as sess:
    output = sess.run(result)
    print(output)
    sess.close()

