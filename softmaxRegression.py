# Implement Softmax Regression
# from https://www.tensorflow.org/versions/master/tutorials/mnist/beginners/index.html

# To use Tensorflow, we first must import it
import tensorflow as tf

# x isn't a specific value. It's a placeholder, a value that we'll input when
# we ask TensorFlow to run a computation. We want to be able to input any
# number of MNIST images, each flattened into a 784-dimensional vector. We
# represent this as a 2-D tensor of floating-point numbers, with a shape
# [None, 784]. (Here None means that a dimension can be of any length.)
x = tf.placeholder(tf.float32, [None, 784])

# We create these Variables by giving tf.Variable the initial value of the
# Variable: in this case, we initialize both W and b as tensors full of zeros.
# Since we are going to learn W and b, it doesn't matter very much what they
# initially are.
#
# Notice that W has a shape of [784, 10] because we want to multiply the
# 784-dimensional image vectors by it to produce 10-dimensional vectors of
# evidence for the difference classes. b has a shape of [10] so we can add it
# to the output.
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# First, we multiply x by W with the expression tf.matmul(x, W). This is
# flipped from when we multiplied them in our equation, where we had Wx, as a
# small trick to deal with x being a 2D tensor with multiple inputs. We then
# add b, and finally apply tf.nn.softmax.
y = tf.nn.softmax(tf.matmul(x, W) + b)
