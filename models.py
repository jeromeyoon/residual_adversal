import tensorflow as tf
from resnet import softmax_layer, conv_layer, residual_block, weight_variable

# ResNet architectures used for CIFAR-10
def resnet(inpt, n,num_filter):
    if n < 20 or (n - 20) % 12 != 0:
        print "ResNet depth invalid."
        return

    dims = inpt.get_shape().as_list()
    num_conv = (n - 20) / 12 + 1
    layers = []

    with tf.variable_scope('conv1'):
        conv1 = conv_layer(inpt, [5, 5, 1, num_filter], 2)
        layers.append(conv1)

    with tf.variable_scope('conv2'):
        conv2 = conv_layer(layers[-1], [3, 3, num_filter, num_filter*2],2)
        layers.append(conv2)

    for i in range (2):
        with tf.variable_scope('conv3_%d' % (i+1)):
            conv3 = residual_block(layers[-1], num_filter*4, False)
            layers.append(conv3)

    with tf.variable_scope('conv_t1'):
	conv_t1 = deconv_layer(layers[-1],num_filter*2,3,2)
	layers.append(conv_t1)

    with tf.variable_scope('conv_t2'):
	conv_t2 = deconv_layer(layers[-1],num_filter,3,2)
	layers.append(conv_t2)
    """
    with tf.variable_scope('conv4'):
        conv4 = conv_layer(layers[-1], [3, 3, num_filter*4, num_filter], 1)
        layers.append(conv4)
    """
    with tf.variable_scope('conv_final'):
	filter_shape = [3,3,layers[-1].get_shape().as_list()[3],3]
	filter_ = weight_variable(filter_shape)
	conv_final = tf.nn.tanh(tf.nn.conv2d(layers[-1],filter=filter_,strides=[1,1,1,1],padding='SAME'))
	layers.append((conv_final))
    return layers[-1]
