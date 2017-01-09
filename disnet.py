import tensorflow as tf 
from ops import lrelu ,linear
from resnet import conv_layer
import pdb
def disnet(ipt,keep_prob,df_dims,reuse=False):
    num_block =2
    layers=[]
    if reuse:
         tf.get_variable_scope().reuse_variables()
    with tf.variable_scope('dis_conv0'):
    	h0 = lrelu(conv_layer(ipt,[3,3,3,df_dims],2)) #output size: 96x96
	layers.append(h0)

    with tf.variable_scope('dis_conv1'):
    	input_depth = layers[-1].get_shape().as_list()[3]
    	h1 = lrelu(conv_layer(layers[-1],[3,3,input_depth,df_dims],2)) #output size: 96x96
	layers.append(h1)

    for i in range(num_block):
        with tf.variable_scope('dis_conv3_%d' %(i+1)):     
            h3 = disblock(layers[-1],df_dims*2)#output 24 x 24 
	    layers.append(h3)
    """    
    for i in range(num_block):
        with tf.variable_scope('dis_conv4_%d' %(i+1)):     
            h4 = disblock(layers[-1],df_dims*4) #output 6 x 6
	    layers.append(h4)
    """
    with tf.variable_scope('dis_fc1'):
	batch_size = layers[-1].get_shape().as_list()[0]
	h5 = tf.reshape(layers[-1],[batch_size,-1])
    	h5 = lrelu(linear(h5,1024)) #output size: 1x1
	layers.append(h5)

    with tf.variable_scope('dis_fc2'):
    	h6 = tf.nn.dropout(lrelu(linear(layers[-1],1024)),keep_prob) #output size: 1x1
	layers.append(h6)

    with tf.variable_scope('dis_fc3'):
    	h7 = linear(layers[-1],1) #output size: 1x1
	#layers.append(h7)
    return tf.nn.sigmoid(h7),h7
 

def disblock(ipt,output_depth):
    input_depth = ipt.get_shape().as_list()[3]
    conv1 = conv_layer(ipt, [3, 3, input_depth, output_depth], 2)
    conv2 = conv_layer(conv1, [3, 3, output_depth, output_depth], 1)
    return conv2
