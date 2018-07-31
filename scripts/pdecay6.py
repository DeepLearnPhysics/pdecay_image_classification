import subprocess
from larcv import larcv
from larcv.dataloader2 import larcv_threadio
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os,sys,time

# tensorflow/gpu start-up configuration
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf

TUTORIAL_DIR     = '..'
TRAIN_IO_CONFIG  = os.path.join(TUTORIAL_DIR, 'tf/pdecay_train.cfg')
TEST_IO_CONFIG   = os.path.join(TUTORIAL_DIR, 'tf/pdecay_test.cfg' )
TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE  = 100
LOGDIR           = 'log_pdecay6'
ITERATIONS       = 25000
SAVE_SUMMARY     = 100
SAVE_WEIGHTS     = 1000

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

def build(input_tensor, num_class=4, trainable=True, debug=True):

    net = input_tensor
    if debug: print('input tensor:', input_tensor.shape)

    filters = 32
    num_modules = 5
    with tf.variable_scope('conv'):
        for step in xrange(5):
            stride = 2
            if step: stride = 1
            net = slim.conv2d(inputs        = net,        # input tensor
                              num_outputs   = filters,    # number of filters (neurons) = # of output feature maps
                              kernel_size   = [3,3],      # kernel size
                              stride        = stride,     # stride size
                              trainable     = trainable,  # train or inference
                              activation_fn = tf.nn.relu, # relu
                              normalizer_fn = slim.batch_norm,
			      scope         = 'conv%da_conv' % step)

            net = slim.conv2d(inputs        = net,        # input tensor
                              num_outputs   = filters,    # number of filters (neurons) = # of output feature maps
                              kernel_size   = [3,3],      # kernel size
                              stride        = 1,          # stride size
                              trainable     = trainable,  # train or inference
                              activation_fn = tf.nn.relu, # relu
			      normalizer_fn = slim.batch_norm,
                              scope         = 'conv%db_conv' % step)
            if (step+1) < num_modules:
                net = slim.max_pool2d(inputs      = net,    # input tensor
                                      kernel_size = [2,2],  # kernel size
                                      stride      = 2,      # stride size
                                      scope       = 'conv%d_pool' % step)

            else:
                net = tf.layers.average_pooling2d(inputs = net,
                                                  pool_size = [net.get_shape()[-2].value,net.get_shape()[-3].value],
                                                  strides = 1,
                                                  padding = 'valid',
                                                  name = 'conv%d_pool' % step)
            filters *= 2

            if debug: print('After step',step,'shape',net.shape)

    with tf.variable_scope('final'):
        net = slim.flatten(net, scope='flatten')

        if debug: print('After flattening', net.shape)

        net = slim.fully_connected(net, int(num_class), scope='final_fc')

        if debug: print('After final_fc', net.shape)

    return net

#
# Step 2: Build network + define loss & solver
#
# retrieve dimensions of data for network construction
dim_data  = train_io.fetch_data('train_image').dim()
dim_label = train_io.fetch_data('train_label').dim()
# define place holders
data_tensor    = tf.placeholder(tf.float32, [None, dim_data[1] * dim_data[2] * dim_data[3]], name='image')
label_tensor   = tf.placeholder(tf.float32, [None, dim_label[1]], name='label')
data_tensor_2d = tf.reshape(data_tensor, [-1,dim_data[1],dim_data[2],dim_data[3]],name='image_reshape')

# Let's keep 10 random set of images in the log
tf.summary.image('input',data_tensor_2d,10)
# build net
net = build(input_tensor=data_tensor_2d, num_class=dim_label[1], trainable=True, debug=False)
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
saver = tf.train.Saver()

#
# Step 4: Run training loop
#
for i in range(ITERATIONS):

    train_data  = train_io.fetch_data('train_image').data()
    train_label = train_io.fetch_data('train_label').data()

    feed_dict = { data_tensor  : train_data,
                  label_tensor : train_label }

    loss, acc, _ = sess.run([cross_entropy, accuracy, train_step], feed_dict=feed_dict)
    print('step',i,loss, acc)
    if (i+1)%SAVE_SUMMARY == 0:
        # Save train log
        sys.stdout.write('Training in progress @ step %d loss %g accuracy %g          \n' % (i,loss,acc))
        sys.stdout.flush()
        s = sess.run(merged_summary, feed_dict=feed_dict)
        writer_train.add_summary(s,i)
    
        # Calculate & save test log
        test_data  = test_io.fetch_data('test_image').data()
        test_label = test_io.fetch_data('test_label').data()
        feed_dict  = { data_tensor  : test_data,
                       label_tensor : test_label }
        loss, acc = sess.run([cross_entropy, accuracy], feed_dict=feed_dict)
        sys.stdout.write('Testing in progress @ step %d loss %g accuracy %g          \n' % (i,loss,acc))
        sys.stdout.flush()
        s = sess.run(merged_summary, feed_dict=feed_dict)
        writer_test.add_summary(s,i)
        
        test_io.next()

    train_io.next()

    if (i+1)%SAVE_WEIGHTS == 0:
        ssf_path = saver.save(sess,'weights_pdecay6/toynet',global_step=i)
        print('saved @',ssf_path)

# inform log directory
print()
print('Run `tensorboard --logdir=%s` in terminal to see the results.' % LOGDIR)
train_io.reset()
test_io.reset()
