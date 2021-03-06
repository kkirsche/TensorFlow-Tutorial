# Implement Softmax Regression
# from https://www.tensorflow.org/versions/master/tutorials/mnist/beginners/index.html

# To use Tensorflow, we first must import it
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# SECTION: IMPLEMENTING ALGO

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

# SECTION: TRAINING

# One very common, very nice cost function is "cross-entropy." Surprisingly,
# cross-entropy arises from thinking about information compressing codes in
# information theory but it winds up being an important idea in lots of areas,
# from gambling to machine learning. It's defined:
#
# Hy′(y)=−∑iyi′log⁡(yi)
#
# Where y is our predicted probability distribution, and y′ is the true
# distribution (the one-hot vector we'll input). In some rough sense, the
# cross-entropy is measuring how inefficient our predictions are for describing
# the truth. Going into more detail about cross-entropy is beyond the scope of
# this tutorial, but it's well worth
# [understanding](http://colah.github.io/posts/2015-09-Visual-Information/).
#
# Add placeholder for correct answers
y_ = tf.placeholder(tf.float32, [None, 10])

# First, tf.log computes the logarithm of each element of y. Next, we multiply
# each element of y_ with the corresponding element of tf.log(y). Then
# tf.reduce_sum adds the elements in the second dimension of y, due to the
# reduction_indices=[1] parameter. Finally, tf.reduce_mean computes the mean
# over all the examples in the batch.
cross_entropy = tf.reduce_mean(
    -tf.reduce_sum(
        y_ * tf.log(y),
        reduction_indices=[1]
    )
)

# Now that we know what we want our model to do, it's very easy to have
# TensorFlow train it to do so. Because TensorFlow knows the entire graph of
# your computations, it can automatically use the backpropagation algorithm to
# efficiently determine how your variables affect the cost you ask it to
# minimize. Then it can apply your choice of optimization algorithm to modify
# the variables and reduce the cost.
# http://colah.github.io/posts/2015-08-Backprop/ — More Info on Algo

# In this case, we ask TensorFlow to minimize cross_entropy using the gradient
# descent algorithm with a learning rate of 0.5. Gradient descent is a simple
# procedure, where TensorFlow simply shifts each variable a little bit in the
# direction that reduces the cost. But TensorFlow also provides many other
# optimization algorithms: using one is as simple as tweaking one line.
#
# What TensorFlow actually does here, behind the scenes, is it adds new
# operations to your graph which implement backpropagation and gradient
# descent. Then it gives you back a single operation which, when run, will do a
# step of gradient descent training, slightly tweaking your variables to reduce
# the cost.
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# Initialize the variables that we've set up so far.
init = tf.initialize_all_variables()

# We can now launch the model in a Session, and run the operation that
# initializes the variables:
sess = tf.Session()
sess.run(init)

# Each step of the loop, we get a "batch" of one hundred random data points
# from our training set. We run train_step feeding in the batches data to
# replace the placeholders.
#
# Using small batches of random data is called stochastic training -- in this
# case, stochastic gradient descent. Ideally, we'd like to use all our data for
# every step of training because that would give us a better sense of what we
# should be doing, but that's expensive. So, instead, we use a different subset
# every time. Doing this is cheap and has much of the same benefit.
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# SECTION: EVALUATING OUR MODEL
# tf.argmax is an extremely useful function which gives you the index of the
# highest entry in a tensor along some axis. For example, tf.argmax(y,1) is the
# label our model thinks is most likely for each input, while tf.argmax(y_,1)
# is the correct label. We can use tf.equal to check if our prediction matches
# the truth.
correct_prediction = tf.equal(
    tf.argmax(y, 1),
    tf.argmax(y_, 1)
)

# That gives us a list of booleans. To determine what fraction are correct, we
# cast to floating point numbers and then take the mean. For example, [True,
# False, True, True] would become [1,0,1,1] which would become 0.75.
accuracy = tf.reduce_mean(
    tf.cast(correct_prediction, tf.float32)
)

# Finally, we ask for our accuracy on our test data. This should be about 92%.
print(
    'Accuracy of training data: ' + repr(
        sess.run(
            accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}
        ) * 100
    ) + '%'
)
