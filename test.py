import models, threading, scipy.misc,os,ang_loss,random,time
import numpy as np
import disnet
from ops import *
import scipy.ndimage
import tensorflow as tf


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.002, 'Learning rate')
flags.DEFINE_float('dropout', 0.7, 'Drop out')
flags.DEFINE_integer('batch_size', 1, 'Batch size')
flags.DEFINE_integer('num_threads', 1, 'number of threads')
flags.DEFINE_string('dataset','0105', 'checkpoint name')
flags.DEFINE_integer('epochs', 1000, 'epochs size')

def create_mask(images):
    tmp1 = [images >-1.][0]*1.
    #tmp2 = [tmp1 ==0][0]*-1.
    #mask = tmp1 + tmp2
    return tmp1


def create_mask2(images):
    tmp1 = [images ==-1.][0]*-1.
    return tmp1
if __name__ =='__main__':

	IR_shape=[1,600,800,1]
	Normal_shape=[1,600,800,3]

	IR_images = tf.placeholder(tf.float32,shape= IR_shape)
	Mask_images = tf.placeholder(tf.float32,shape= IR_shape)
	Normal_images = tf.placeholder(tf.float32,shape=Normal_shape)
	keep_prob = tf.placeholder(tf.float32)

	# Buidl networks
	pred_Normal = models.resnet(IR_images, 20,32)
	sess = tf.Session()
	sess.run(tf.initialize_all_variables())

	saver = tf.train.Saver(max_to_keep=10)
	ckpt = tf.train.get_checkpoint_state(os.path.join('checkpoint',FLAGS.dataset))
	pdb.set_trace()
	if ckpt and ckpt.model_checkpoint_path :
	    	print "Restoring from checkpoint"
        	ckpt_name = os.path.basename(ckpt.all_model_checkpoint_paths[-1])
		saver.restore(sess, os.path.join('checkpoint',FLAGS.dataset,ckpt_name))
	else:
	    	print "Couldn't find checkpoint to restore from. Starting over."
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
	ckpt_name = ckpt_name.encode("UTF-8")

	####### Load Validation dataset #########

	list_val = [11,16,21,22,33,36,38,53,59,92]
	savepath ='./resi_ad_result'
	for idx in range(len(list_val)):
		if not os.path.exists(os.path.join(savepath,FLAGS.dataset,ckpt_name,'%03d' %list_val[idx])):
			os.makedirs(os.path.join(savepath,FLAGS.dataset,ckpt_name,'%03d' %list_val[idx]))
		for idx2 in range(5,7): #tilt angles 1~9 
			for idx3 in range(5,7): # light source 
				print("Selected material %03d/%d" % (list_val[idx],idx2))
			        img = '/research2/IR_normal_small/save%03d/%d' % (list_val[idx],idx2)
			        input_ = scipy.misc.imread(img+'/%d.bmp' %idx3).astype(float) #input NIR image
			        input_ = scipy.misc.imresize(input_,[600,800])
			        input_  = input_/127.5 -1.0 # normalize -1 ~1
			        input_ = np.reshape(input_,(1,600,800,1)) 
			        input_ = np.array(input_).astype(np.float32)
				mask = create_mask(input_)
				mask = np.squeeze(create_mask(input_))
				mask = np.expand_dims(mask,axis=-1)
				mask2 = np.squeeze(create_mask2(input_))
				mask2 = np.expand_dims(mask2,axis=-1)
				mask2 = np.tile(mask2,(1,1,3))
			        #input_ = input_ - mean_mask	
			        start_time = time.time() 
			        sample  = sess.run(pred_Normal, feed_dict={IR_images: input_})
			        print('time: %.8f' %(time.time()-start_time))     
			        # normalization #
			        sample = np.squeeze(sample).astype(np.float32)
				sample = sample * mask +mask2
				t = np.square(sample)
				t = np.sqrt(t.sum(axis=2))
				t = np.expand_dims(t,axis=-1)
				sample=sample/t	
			        sample = (sample+1.)/2.

			        if not os.path.exists(os.path.join(savepath,'%s/%s/%03d/%d' %(FLAGS.dataset,ckpt_name,list_val[idx],idx2))):
			            os.makedirs(os.path.join(savepath,'%s/%s/%03d/%d' %(FLAGS.dataset,ckpt_name,list_val[idx],idx2)))
			        savename = os.path.join(savepath,'%s/%s/%03d/%d/single_normal_%03d.bmp' % (FLAGS.dataset,ckpt_name,list_val[idx],idx2,idx3))
			        scipy.misc.imsave(savename, sample)


	sess.close()



