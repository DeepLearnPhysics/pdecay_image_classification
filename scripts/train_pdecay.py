import subprocess
from larcv import larcv
from larcv.dataloader2 import larcv_threadio
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#%matplotlib inline
import os,sys,time
import sys

# tensorflow/gpu start-up configuration
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf

TRAIN_IO_CONFIG  = './config/pdecay_train.cfg'
TEST_IO_CONFIG   = './config/pdecay_test.cfg' 
TRAIN_BATCH_SIZE = 50
TEST_BATCH_SIZE  = 50
LOGDIR           = 'log_pdecay6s'
WEIGHTS          = './weights/weight'
ITERATIONS       = 25000
SAVE_SUMMARY     = 200
SAVE_WEIGHTS     = 1000

for argv in sys.argv:
  if 'train=' in argv:
    TRAIN_IO_CONFIG = argv.replace('train=','')
  elif 'test=' in argv:
    TEST_IO_CONFIG = argv.replace('test=','')
  elif 'log=' in argv:
    LOGDIR = argv.replace('log=','')
  elif 'weight=' in argv:
    WEIGHTS = argv.replace('weights=','')

# Check log directory is empty
train_logdir = os.path.join(LOGDIR,'train')
test_logdir  = os.path.join(LOGDIR,'test')
if not os.path.isdir(train_logdir): os.makedirs(train_logdir)
if not os.path.isdir(test_logdir):  os.makedirs(test_logdir)
if len(os.listdir(train_logdir)) or len(os.listdir(test_logdir)):
  sys.stderr.write('Error: train or test log dir not empty...\n')
  raise OSError
    
#
# Step 0: IO
#
# for "train" data set
train_io = larcv_threadio()  # create io interface
train_io_cfg = {'filler_name' : 'TrainIO',
                'verbosity'   : 0,
                'filler_cfg'  : TRAIN_IO_CONFIG}
train_io.configure(train_io_cfg)   # configure
train_io.start_manager(TRAIN_BATCH_SIZE) # start read thread
time.sleep(2)
train_io.next()

# for "test" data set
test_io = larcv_threadio()   # create io interface
test_io_cfg = {'filler_name' : 'TestIO',
               'verbosity'   : 0,
               'filler_cfg'  : TEST_IO_CONFIG}
test_io.configure(test_io_cfg)   # configure
test_io.start_manager(TEST_BATCH_SIZE) # start read thread
time.sleep(2)
test_io.next()

#
# Step 1: Define network
#
import tensorflow.contrib.slim as slim
import tensorflow.python.platform
from build_resnet import build

#
# Step 2: Build network + define loss & solver
#
# retrieve dimensions of data for network construction
dim_data  = train_io.fetch_data('train_image0').dim()
# define place holders
data_tensors    = []
data_tensors_2d = []
data_tensor0, data_tensor1 = (None,None)

train_data0  = train_io.fetch_data('train_image0')
if train_data0 is not None:
  dim_data0    = train_data0.dim()
  data_tensors.append(tf.placeholder(tf.float32, [None, dim_data0[1] * dim_data0[2] * dim_data0[3]], name='image0'))
  data_tensors_2d.append(tf.reshape(data_tensors[-1], [-1,dim_data0[1],dim_data0[2],dim_data0[3]], name='reshape0'))

train_data1  = train_io.fetch_data('train_image1')
if train_data1 is not None:
  dim_data1    = train_data1.dim()
  data_tensors.append(tf.placeholder(tf.float32, [None, dim_data1[1] * dim_data1[2] * dim_data1[3]], name='image1'))
  data_tensors_2d.append(tf.reshape(data_tensors[-1], [-1,dim_data1[1],dim_data1[2],dim_data1[3]], name='reshape1'))

# retrieve dimensions of label for network construction
dim_label    = train_io.fetch_data('train_label').dim()
label_tensor = tf.placeholder(tf.float32, [None, dim_label[1]], name='label')

# Let's keep 10 random set of images in the log
#tf.summary.image('input',data_tensor_2d0,10)
# build net
net = build(input_tensors=data_tensors_2d, num_class=int(dim_label[1]), trainable=True, debug=True)

# Define accuracy
with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(net,1), tf.argmax(label_tensor,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
# Define loss + backprop as training step
with tf.name_scope('train'):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label_tensor, logits=net))
    tf.summary.scalar('cross_entropy',cross_entropy)
    train_step = tf.train.AdamOptimizer(0.00005).minimize(cross_entropy)
    
#                                                                                                                                      
# Step 3: weight saver & summary writer                                                                                                
#                                                                                                                                      
# Create a bandle of summary                                                                                                           
merged_summary=tf.summary.merge_all()
# Create a session                                                                                                                     
sess = tf.InteractiveSession()
# Initialize variables                                                                                                                 
sess.run(tf.global_variables_initializer())
# Create a summary writer handle                                                                                                       
writer_train=tf.summary.FileWriter(train_logdir)
writer_train.add_graph(sess.graph)
writer_test=tf.summary.FileWriter(test_logdir)
writer_test.add_graph(sess.graph)
# Create weights saver                                                                                                                 
saver = tf.train.Saver(max_to_keep=10, keep_checkpoint_every_n_hours=0.5)

#
# Step 4: Run training loop
#
for i in range(ITERATIONS):

  feed_dict = {}
  if train_data0 is not None: feed_dict[data_tensors[0]] = train_io.fetch_data('train_image0').data()
  if train_data1 is not None: feed_dict[data_tensors[1]] = train_io.fetch_data('train_image1').data()
  feed_dict[label_tensor] = train_io.fetch_data('train_label').data()
  
  loss, acc, _ = sess.run([cross_entropy, accuracy, train_step], feed_dict=feed_dict)
  print('step',i,loss,acc)
  if (i+1)%SAVE_SUMMARY == 0:
    # Save train log
    sys.stdout.write('Training in progress @ step %d loss %g accuracy %g          \n' % (i,loss,acc))
    sys.stdout.flush()
    s = sess.run(merged_summary, feed_dict=feed_dict)
    writer_train.add_summary(s,i)
    
    # Calculate & save test log
    test_data0 = test_io.fetch_data('test_image0')
    test_data1 = test_io.fetch_data('test_image1')
    test_label = test_io.fetch_data('test_label')
    feed_dict  = {}
    if test_data0 is not None: feed_dict[data_tensors[0]] = test_data0.data()
    if test_data1 is not None: feed_dict[data_tensors[1]] = test_data1.data()
    feed_dict[label_tensor] = test_label.data()
    loss, acc = sess.run([cross_entropy, accuracy], feed_dict=feed_dict)
    sys.stdout.write('Testing in progress @ step %d loss %g accuracy %g          \n' % (i,loss,acc))
    sys.stdout.flush()
    s = sess.run(merged_summary, feed_dict=feed_dict)
    writer_test.add_summary(s,i)
    
    test_io.next()
    
  train_io.next()
    
  if (i+1)%SAVE_WEIGHTS == 0:
    ssf_path = saver.save(sess,WEIGHTS,global_step=i)
    print('saved @',ssf_path)

# inform log directory
print()
print('Run `tensorboard --logdir=%s` in terminal to see the results.' % LOGDIR)
train_io.reset()
test_io.reset()
