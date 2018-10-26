import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnsit = input_data.read_data_sets('MNIST_data', one_hot=True)


width = 28 # pixels
height = 28 # pixels
flat = width * height  # one full row of pixels
class_output = 10  # number of labels


x = tf.placeholder(tf.float32, shape=[None, flat])          #   Inputs
_y = tf.placeholder(tf.float32, shape=[None, class_output]) #   Outputs


# turn image in 2d tensor (1 row and 784 columns)

                      # -1 ( the size allocated for the id number of the picture which is column 1. set to negitive -1 so it doesn't matter)
                      # 28 28  the size of the picture
                      # 1  the number of lists inside each array place  (called channels)
x_image = tf.reshape(x, [-1,28,28,1])


# Can create a weights variable when passed the shape of the weight
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# can create a baises variable if passed the biases shape (tensor size)
def biases_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# can create a layer is passed the inputs and the weights  (biases can just always stay the same)
def conv2d(x, W):
    return tf.nn.conv2d(x,W, strides=[1,1,1,1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')



# First Layer of Conv net

# Weights and biases
W_conv1 = weight_variable([5,5,1,32])
b_conv1 = biases_variable([32])

# Layer
convole1 = conv2d(x_image, W_conv1) + b_conv1

#Activation function
h_conv1 = tf.nn.relu(convole1)





