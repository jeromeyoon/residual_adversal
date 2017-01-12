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
        conv1 = conv_layer(inpt, [5, 5, 1, num_filter], 1)
        layers.append(conv1)

    with tf.variable_scope('conv_2'):
        conv1 = conv_layer(layers[-1], [1, 1, num_filter, 12], 1)
        layers.append(conv1)

    for i in range (num_conv):
        with tf.variable_scope('conv_3_%d' % (i+1)):
            conv2_x = residual_block(layers[-1], 12, False)
            conv2 = residual_block(conv2_x, 12, False)
            layers.append(conv2_x)
            layers.append(conv2)

        #assert conv2.get_shape().as_list()[1:-1] == [dims[1], dims[2], num_filter]
    for i in range (num_conv):
        with tf.variable_scope('conv_4_%d' % (i+1)):
            conv3_x = residual_block(layers[-1], 12, False)
            conv3 = residual_block(conv3_x, 12, False)
            layers.append(conv3_x)
            layers.append(conv3)

    with tf.variable_scope('conv_5'):
        conv4 = conv_layer(layers[-1], [1, 1, 12, num_filter], 1)
        layers.append(conv4)
    """
        #assert conv4.get_shape().as_list()[1:] == [dims[1], dims[2], num_filter]
    with tf.variable_scope('conv6'):
        conv4 = conv_layer(layers[-1], [9, 9, num_filter, 3], 1,batch=False)
        layers.append(conv4)
    
    with tf.variable_scope('deconv'):
        deconv = deconv_layer(layers[-1],[dims[0],dims[1],dims[2],dims[3]],[5,5,num_filter,1],2)  
    for i in range (num_conv):
        with tf.variable_scope('conv4_%d' % (i+1)):
            conv4_x = residual_block(layers[-1], 64, False)
            conv4 = residual_block(conv4_x, 64, False)
            layers.append(conv4_x)
            layers.append(conv4)

        assert conv4.get_shape().as_list()[1:] == [dims[1],dims[2],num_filter]

    for i in range (num_conv):
        with tf.variable_scope('conv5_%d' % (i+1)):
            conv5_x = residual_block(layers[-1], 64, False)
            conv5 = residual_block(conv5_x, 64, False)
            layers.append(conv4_x)
            layers.append(conv4)

        assert conv4.get_shape().as_list()[1:] == [dims[1],dims[2],num_filter]
    """
    with tf.variable_scope('conv_6'):
	filter_shape = [3,3,layers[-1].get_shape().as_list()[3],3]
	filter_ = weight_variable(filter_shape)
	conv_final = tf.nn.tanh(tf.nn.conv2d(layers[-1],filter=filter_,strides=[1,1,1,1],padding='SAME'))
	layers.append((conv_final))
    
    return layers[-1]
