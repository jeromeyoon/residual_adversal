import pdb
import numpy as np 
from numpy import inf
import tensorflow as tf

def ang_error(samples,gt_samples):

    [b,h,w,c] = samples.get_shape().as_list()
    samples = (samples +1.0)/2.0
    gt_samples = (gt_samples +1.0)/2.0
    output = tf.nn.l2_normalize(samples,3)
    gt_output = tf.nn.l2_normalize(gt_samples,3)
    tmp = tf.mul(output,gt_output)
    tmp = tf.div(tf.reduce_sum(tf.abs(tmp)),b*h*w)
    """
    samples = tf.div(tf.add(samples,1.0),2) # 0~1
    #samples = tf.mul(samples,mask)
    samples = tf.clip_by_value(samples,1e-10,1.0)
    gt_sample = tf.div(tf.add(gt_sample,1.0),2) #0~1
    gt_sample = tf.clip_by_value(gt_sample,1e-10,1.0)

    output = tf.sqrt(tf.reduce_sum(tf.pow(samples,2.),3))
    output = tf.expand_dims(output,-1)
    output = tf.div(samples,output)
    gt_output = tf.sqrt(tf.reduce_sum(tf.pow(gt_sample,2.),3))
    gt_output = tf.expand_dims(gt_output,-1)
    gt_output = tf.div(gt_sample,gt_output)
    tmp = tf.mul(output,gt_output)
    tmp = tf.reduce_mean(tmp)
    #tmp = tf.div(tf.reduce_sum(tf.sub(1.0,tf.reduce_sum(tmp,3))),tf.to_float(samples.get_shape()[1]*samples.get_shape()[2]*3))
    """
    return tmp
    """ 
    output = np.zeros((samples.shape[0],samples.shape[1],samples.shape[2],samples.shape[-1])).astype(np.float32).astype(float)

    for idx,sample in enumerate (samples):
        output[idx,:,:,0] = sample[:,:,0]/(np.sqrt(np.power(sample[:,:,0],2) + np.power(sample[:,:,1],2) + np.power(sample[:,:,2],2)))
        output[idx,:,:,1] = sample[:,:,1]/(np.sqrt(np.power(sample[:,:,0],2) + np.power(sample[:,:,1],2) + np.power(sample[:,:,2],2)))
        output[idx,:,:,2] = sample[:,:,2]/(np.sqrt(np.power(sample[:,:,0],2) + np.power(sample[:,:,1],2) + np.power(sample[:,:,2],2)))

    gt_output = np.zeros((samples.shape[0],samples.shape[1],samples.shape[2],samples.shape[-1])).astype(np.float32).astype(float)
    for idx,sample in enumerate (gt_sample):
        gt_output[idx,:,:,0] = sample[:,:,0]/(np.sqrt(np.power(sample[:,:,0],2) + np.power(sample[:,:,1],2) + np.power(sample[:,:,2],2)))
        gt_output[idx,:,:,1] = sample[:,:,1]/(np.sqrt(np.power(sample[:,:,0],2) + np.power(sample[:,:,1],2) + np.power(sample[:,:,2],2)))
        gt_output[idx,:,:,2] = sample[:,:,2]/(np.sqrt(np.power(sample[:,:,0],2) + np.power(sample[:,:,1],2) + np.power(sample[:,:,2],2)))
    """

    #output = output[np.logical_not(np.isnan(x))]
    #gt_output = gt_output[np.logical_not(np.isnan(x))]
    #output[output == inf] = 0
    #gt_output[gt_output == inf] = 0

    #tmp = np.multiply(output,gt_output)
    #tmp = 1.0 - np.sum(tmp,axis=3,dtype=np.float32)
    
    #tmp = np.arccos(tmp)
    #return  np.sum(tmp).astype(float)/(samples.shape[0]*samples.shape[1]*samples.shape[2])
    #return np.asarray(output)


