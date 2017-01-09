import pdb
import numpy as np 
from numpy import inf
import tensorflow as tf

def ang_error(samples,gt_samples):

    [b,h,w,c] = samples.get_shape().as_list()
    #mask_samples = (mask_samples +1.0)/2.0
    valid_pixel = b*h*w
    #samples = (samples +1.0)/2.0
    #gt_samples = (gt_samples +1.0)/2.0
    output = l2_normalize(samples)
    gt_output = l2_normalize(gt_samples)
    tmp = tf.reduce_sum(tf.mul(output,gt_output),3)
    output = tf.div(tf.reduce_sum(tf.sub(tf.ones_like(tmp,dtype=tf.float32),tmp)),valid_pixel)
    return tmp,output

def l2_normalize(input_):
    tmp1 = tf.square(input_)
    tmp2 = tf.sqrt(tf.reduce_sum(tmp1,3))
    tmp3 = tf.expand_dims(tf.maximum(tmp2,1e-12),-1)
    tmp4 = tf.div(input_,tmp3)
    return tmp4
