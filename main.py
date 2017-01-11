import models, threading, scipy.misc,os,ang_loss,random,time
import numpy as np
import disnet
from compute_ei import *
from ops import *
import scipy.ndimage
import tensorflow as tf


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.0002, 'Learning rate')
flags.DEFINE_float('dropout', 0.7, 'Drop out')
flags.DEFINE_integer('batch_size', 20, 'Batch size')
flags.DEFINE_integer('num_threads', 8, 'number of threads')
flags.DEFINE_string('dataset','0110', 'checkpoint name')
flags.DEFINE_float('gpu_ratio','1.0', 'gpu fraction')
flags.DEFINE_integer('epochs', 1000, 'epochs size')

def create_mask(images):
    tmp1 = [images >-1.][0]*1.
    #tmp2 = [tmp1 ==0][0]*-1.
    #mask = tmp1 + tmp2
    return tmp1


def indices(a,func):
    return [i for (i,val) in enumerate(a) if func(val)]

def load_and_enqueue(sess,coord,IR_shape,file_list,label_list,S,idx=0,num_thread=1):
	count =0;
    	length = len(file_list)
    	#rot=[0,90,180,270]
    	while not coord.should_stop():
		i = (count*num_thread + idx) % length;
		r = random.randint(0,2)
		input_img = scipy.misc.imread(file_list[S[i]]).reshape([224,224,1]).astype(np.float32)
		gt_img = scipy.misc.imread(label_list[S[i]]).reshape([224,224,3]).astype(np.float32)
		input_img = input_img/127.5 -1.
		gt_img = gt_img/127.5 -1.
		rand_x = np.random.randint(64,224-64)
		rand_y = np.random.randint(64,224-64)
		ipt =  input_img[rand_y:rand_y+64,rand_x:rand_x+64,:]
		label = gt_img[rand_y:rand_y+64,rand_x:rand_x+64,:]
		mask = create_mask(ipt)
		#gamma = np.random.uniform(0.7,2.5)
		#gamma = 2.2	
		#ipt = np.power((ipt/255.0),1/gamma)*255.0
		#ipt = ipt /127.5 -1.
		#input_img = scipy.ndimage.rotate(input_img,rot[r])
		#gt_img = scipy.ndimage.rotate(gt_img,rot[r])
		sess.run(enqueue_op,feed_dict={IR_single:ipt,Mask_single:mask,Normal_single:label})
		count +=1


if __name__ =='__main__':
	if not os.path.exists(os.path.join('checkpoint',FLAGS.dataset)):
	    os.makedirs(os.path.join('checkpoint',FLAGS.dataset))

	IR_shape=[64,64,1]
	Normal_shape=[64,64,3]

	# Threading setting 
	print 'Queue loading'
	IR_single = tf.placeholder(tf.float32,shape= IR_shape)
	Mask_single = tf.placeholder(tf.float32,shape= IR_shape)
	Normal_single = tf.placeholder(tf.float32,shape=Normal_shape)
	keep_prob = tf.placeholder(tf.float32)
	q = tf.FIFOQueue(8000,[tf.float32,tf.float32,tf.float32],[[IR_shape[0],IR_shape[1],1],[IR_shape[0],IR_shape[1],1],[Normal_shape[0],Normal_shape[1],3]])
	enqueue_op = q.enqueue([IR_single,Mask_single,Normal_single])
	IR_images,Mask_images,Normal_images = q.dequeue_many(FLAGS.batch_size)

	# Buidl networks
	pred_Normal = models.resnet(IR_images, 20,64)
	D_real,D_real_logits = disnet.disnet(Normal_images,keep_prob,64)
	D_fake,D_fake_logits = disnet.disnet(pred_Normal,keep_prob,64,reuse=True)
	# Discriminator loss
	D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(D_real_logits, tf.random_uniform(D_real.get_shape(),minval=0.7,maxval=1.2,dtype=tf.float32)))
	D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(D_fake_logits, tf.random_uniform(D_fake.get_shape(),minval=0.0,maxval=0.3,dtype=tf.float32)))
	D_loss = D_loss_real + D_loss_fake

	# Generator loss
	#G_loss= tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(D_fake_logits, tf.ones_like(D_fake)))
	G_loss= tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(D_fake_logits, tf.random_uniform(D_fake.get_shape(),minval=0.7,maxval=1.2,dtype=tf.float32)))
	pdb.set_trace()
	L2_loss = tf.sqrt(tf.reduce_mean(tf.square(Normal_images[:,10:-10,10:-10,:] - pred_Normal[:,10:-10,10:-10,:])))
	ang_tmp,ang_loss = ang_loss.ang_error(pred_Normal[:,10:-10,10:-10,:],Normal_images[:,10:-10,10:-10,:]) # ang_loss is normalized 0~1
	#ei_loss = tf.py_func(compute_ei,[pred_Normal],[tf.float64])
	#ei_loss = tf.pack(ei_loss[0])
	#ei_loss = tf.to_float(ei_loss[0])
	Gen_loss = G_loss*0.1 + L2_loss + ang_loss

	# Optimizer
	t_vars = tf.trainable_variables()
	d_vars =[var for var in t_vars if 'dis_' in var.name]
	g_vars =[var for var in t_vars if 'conv' in var.name]
	global_step = tf.Variable(0,name='global_step',trainable=False)
	global_step1 = tf.Variable(0,name='global_step1',trainable=False)
	g_lr = tf.train.exponential_decay(FLAGS.learning_rate,global_step,20000,0.6,staircase=True)
	G_opt = tf.train.AdamOptimizer(g_lr).minimize(Gen_loss,global_step=global_step,var_list=g_vars)
	D_opt = tf.train.AdamOptimizer(g_lr).minimize(D_loss,global_step=global_step1,var_list=d_vars)


	config = tf.ConfigProto()
	config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_ratio
	sess = tf.Session(config=config)
	sess.run(tf.initialize_all_variables())


	saver = tf.train.Saver(max_to_keep=10)
	ckpt = tf.train.latest_checkpoint(os.path.join('checkpoint',FLAGS.dataset))
	if ckpt and ckpt.model_checkpoint_path:
	    	print "Restoring from checkpoint", checkpoint
		saver.restore(sess, os.path.join('checkpoint',FLAGS.dataset,ckpt))
	else:
	    	print "Couldn't find checkpoint to restore from. Starting over."

	####### Load training dataset #########
        data = json.load(open("/research2/ECCV_journal/with_light/json/traininput.json"))
        data_label = json.load(open("/research2/ECCV_journal/with_light/json/traingt.json"))
	#data = json.load(open("/research2/ECCV_journal/deconv/Adversal_vgg/patch_224/traininput.json"))
	#data_label = json.load(open("/research2/ECCV_journal/deconv/Adversal_vgg/patch_224/traingt.json"))
	train_input =[data[idx] for idx in xrange(0,len(data))]
	train_gt =[data_label[idx] for idx in xrange(0,len(data))]
	#train_input =[''.join(train_input[idx]) for idx in xrange(0,len(train_input))]
	#train_gt =[''.join(train_gt[idx]) for idx in xrange(0,len(train_gt))]
	S = range(len(train_input))
	random.shuffle(S)
	coord = tf.train.Coordinator()
	num_thread =FLAGS.num_threads

	

	for i in range(num_thread):
		t = threading.Thread(target=load_and_enqueue,args=(sess,coord,IR_shape,train_input,train_gt,S,i,num_thread))
	    	t.start()

	for epoch in xrange(FLAGS.epochs):
	    	batch_idxs = len(train_input)/FLAGS.batch_size
	    	sum_L = 0.0
	    	sum_g =0.0
	    	sum_ang =0.0
	    	sum_ei =0.0
	    	if epoch ==0:
			train_log = open(os.path.join("logs",'train_%s.log' %FLAGS.dataset),'w')
			val_log = open(os.path.join("logs",'val_%s.log' %FLAGS.dataset),'w')
	    	else:
			train_log = open(os.path.join("logs",'train_%s.log' %FLAGS.dataset),'aw')
			val_log = open(os.path.join("logs",'val_%s.log' %FLAGS.dataset),'w')
	    	for idx in xrange(0,batch_idxs):
			start_time = time.time()
			_,d_loss_real,d_loss_fake = sess.run([D_opt,D_loss_real,D_loss_fake],feed_dict={keep_prob:FLAGS.dropout})
			_,g_loss,ang_err,ang_err2,L_loss = sess.run([G_opt,G_loss,ang_loss,ang_tmp,L2_loss],feed_dict={keep_prob:FLAGS.dropout})
			print("Epoch: [%2d] [%4d/%4d] time: %4.4f g_loss: %.6f d_real: %.6f d_fake: %.6f L_loss:%.4f ang_loss: %.6f" \
			% (epoch, idx, batch_idxs,time.time() - start_time,g_loss,d_loss_real,d_loss_fake,L_loss,ang_err))
			sum_L += L_loss 	
			sum_g += g_loss
			sum_ang += ang_err
			#sum_ei += ei_err
	    		if np.mod(global_step.eval(session=sess),6000) ==0:
			    saver.save(sess,os.path.join('checkpoint',FLAGS.dataset,'Res_DCGAN'),global_step=global_step)

	    	train_log.write('epoch %06d mean_g %.6f  mean_L %.6f mean_ang %.6f \n' %(epoch,sum_g/(batch_idxs),sum_L/(batch_idxs),sum_ang/batch_idxs))
	    	train_log.close()
	    	saver.save(sess,os.path.join('checkpoint',FLAGS.dataset,'Res_DCGAN'),global_step=global_step)

	sess.close()



