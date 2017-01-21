import tensorflow as tf
import pdb
from resnet import softmax_layer, conv_layer, residual_block, weight_variable

# ResNet architectures used for CIFAR-10
def resnet(inpt, n,num_filter):
    if n < 20 or (n - 20) % 12 != 0:
        print "ResNet depth invalid."
        return

    dims = inpt.get_shape().as_list()
    num_conv = (n - 20) / 12 + 1
    layers = []

    with tf.variable_scope('conv_1'):
        conv1 = conv_layer(inpt, [3, 3, 1, num_filter], 1)
	layers.append(conv1)

    with tf.variable_scope('conv_2'):
        conv1 = conv_layer(layers[-1], [3, 3, num_filter, num_filter*2], 1)
        layers.append(conv1)

    for i in range (3):
        with tf.variable_scope('conv_3_%d' % (i+1)):
            conv2_x = residual_block(layers[-1], num_filter*4, False)
            layers.append(conv2_x)
    
    with tf.variable_scope('conv_4'):
        conv4 = conv_layer(layers[-1], [3, 3, num_filter*4, num_filter], 1)
        layers.append(conv4)
    
    with tf.variable_scope('conv_5'):
	filter_shape = [3,3,layers[-1].get_shape().as_list()[3],3]
	filter_ = weight_variable(filter_shape)
	conv_final = tf.nn.tanh(tf.nn.conv2d(layers[-1],filter=filter_,strides=[1,1,1,1],padding='SAME'))
	layers.append((conv_final))
    
    return layers[-1]
