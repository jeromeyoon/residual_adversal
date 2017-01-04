import pdb
import numpy as np 
from numpy import inf
import tensorflow as tf

def ang_error(samples,gt_samples,mask_samples):

    [b,h,w,c] = samples.get_shape().as_list()
    valid_pixel = tf.reduce_sum(mask_samples)
    samples = (samples +1.0)/2.0
    gt_samples = (gt_samples +1.0)/2.0
    output = tf.nn.l2_normalize(samples,3)
    gt_output = tf.nn.l2_normalize(gt_samples,3)
    tmp = tf.mul(output,gt_output)
    tmp = tf.div(tf.reduce_sum(tmp),valid_pixel)
    return tmp


