import pdb
import numpy as np 
from numpy import inf
import tensorflow as tf

def ang_error(samples,gt_samples):

    [batch,h,w,c] = samples.get_shape().as_list()
    #mask_samples = (mask_samples +1.0)/2.0
    valid_pixel = b*h*w
    output = l2_normalize(samples)
    gt_output = l2_normalize(gt_samples)
    tmp = tf.reduce_sum(tf.mul(output,gt_output),3)
    output = tf.div(tf.reduce_sum(tf.sub(tf.ones_like(tmp,dtype=tf.float32),tmp),[1,2,3]),h*w)
    return output/batch

def l2_normalize(input_):
    tmp1 = tf.square(input_)
    tmp2 = tf.expand_dims(tf.sqrt(tf.reduce_sum(tmp1,3)),-1)
    #tmp3 = tf.expand_dims(tf.maximum(tmp2,1e-12),-1)
    tmp4 = tf.div(input_,tmp2)
    return tmp4
