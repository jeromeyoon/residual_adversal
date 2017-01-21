import tensorflow as tf
from resnet import softmax_layer, conv_layer, residual_block, weight_variable,deconv_layer
import pdb
# ResNet architectures used for CIFAR-10
def resnet(inpt, n,num_filter):
    if n < 20 or (n - 20) % 12 != 0:
        print "ResNet depth invalid."
        return

    dims = inpt.get_shape().as_list()
    num_conv = (n - 20) / 12 + 1
    layers = []

    with tf.variable_scope('conv1'):
        conv1 = conv_layer(inpt, [9, 9, 1, num_filter], 1)
        layers.append(conv1)

    with tf.variable_scope('conv2'):
        conv2 = conv_layer(layers[-1], [3, 3, num_filter, num_filter*2], 1)
        layers.append(conv2)
    
    with tf.variable_scope('conv3'):
        conv3 = conv_layer(layers[-1], [3, 3, num_filter*2, num_filter*4], 1)
        layers.append(conv3)

    for i in range (5):
        with tf.variable_scope('conv4_%d' % (i+1)):
            conv4 = residual_block(layers[-1], num_filter*4, False)
	    layers.append(conv4)
    """
    with tf.variable_scope('conv_t1'):
	conv_t1 = deconv_layer(layers[-1],64,3,2)
	layers.append(conv_t1)

    with tf.variable_scope('conv_t2'):
	conv_t2 = deconv_layer(layers[-1],32,3,2)
	layers.append(conv_t2)
    """
    with tf.variable_scope('conv5'):
        conv_t3 = tf.nn.tanh(conv_layer(layers[-1], [9, 9, 128,3], 1,relu=False))
        layers.append(conv_t3)
    return layers[-1]
