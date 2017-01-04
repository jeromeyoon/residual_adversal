import tensorflow as tf
from resnet import softmax_layer, conv_layer, residual_block, weight_variable

# ResNet architectures used for CIFAR-10
def resnet(inpt,mask, n,num_filter):
    if n < 20 or (n - 20) % 12 != 0:
        print "ResNet depth invalid."
        return

    dims = inpt.get_shape().as_list()
    num_conv = (n - 20) / 12 + 1
    layers = []

    with tf.variable_scope('conv1'):
        conv1 = conv_layer(inpt, [3, 3, 1, 64], 1)
        layers.append(conv1)

    for i in range (num_conv):
        with tf.variable_scope('conv2_%d' % (i+1)):
            conv2_x = residual_block(layers[-1], num_filter, False)
            conv2 = residual_block(conv2_x, num_filter, False)
            layers.append(conv2_x)
            layers.append(conv2)

        assert conv2.get_shape().as_list()[1:] == [dims[1], dims[2], num_filter]

    for i in range (num_conv):
        with tf.variable_scope('conv3_%d' % (i+1)):
            conv3_x = residual_block(layers[-1], num_filter, False)
            conv3 = residual_block(conv3_x, num_filter, False)
            layers.append(conv3_x)
            layers.append(conv3)

        assert conv3.get_shape().as_list()[1:] == [dims[1], dims[2], num_filter]
    
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

    with tf.variable_scope('conv_final'):
	filter_shape = [3,3,layers[-1].get_shape().as_list()[3],3]
	filter_ = weight_variable(filter_shape)
	conv_final = tf.nn.tanh(tf.nn.conv2d(layers[-1],filter=filter_,strides=[1,1,1,1],padding='SAME'))
        tmp1 = tf.mul(conv_final,mask)		
        tmp2 = [tmp1 == 0][0] *-1.
	conv_final = tmp1 + tmp2			
	layers.append((conv_final))
    return layers[-1]
