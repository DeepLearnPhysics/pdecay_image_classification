import subprocess
from larcv import larcv
from larcv.dataloader2 import larcv_threadio
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#%matplotlib inline
import os,sys,time
import sys

TEST_IO_CONFIG   = './config/pdecay_test.cfg'
TEST_BATCH_SIZE  = 50
WEIGHTS          = './weights/weight'
ITERATIONS       = 50000
GPU              = 0

for argv in sys.argv:
  if 'test=' in argv:
    TEST_IO_CONFIG = argv.replace('test=','')
  elif 'weights=' in argv:
    WEIGHTS = argv.replace('weights=','')
  elif 'gpu=' in argv:
    GPU = argv.replace('gpu=','')

# tensorflow/gpu start-up configuration
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU)
import tensorflow as tf

#
# Step 0: IO
#
# for "test" data set
test_io = larcv_threadio()   # create io interface
test_io_cfg = {'filler_name' : 'TestIO',
               'verbosity'   : 0,
               'filler_cfg'  : TEST_IO_CONFIG}
test_io.configure(test_io_cfg)   # configure
test_io.start_manager(TEST_BATCH_SIZE) # start read thread
time.sleep(2)
test_io.next(store_entries=True,store_event_ids=True)

# Step 0.1: additional input for analysis
from ROOT import TChain
ana_chain = TChain("particle_mctruth_tree")
input_filelist = test_io._proc.pd().io().file_list()
for f in input_filelist:
    ana_chain.AddFile(f)
    print 'Will analyze', ana_chain.GetEntries(), 'entries...'

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
dim_data  = test_io.fetch_data('test_image0').dim()
# define place holders
data_tensors    = []
data_tensors_2d = []
data_tensor0, data_tensor1 = (None,None)

test_data0  = test_io.fetch_data('test_image0')
if test_data0 is not None:
  dim_data0    = test_data0.dim()
  data_tensors.append(tf.placeholder(tf.float32, [None, dim_data0[1] * dim_data0[2] * dim_data0[3]], name='image0'))
  data_tensors_2d.append(tf.reshape(data_tensors[-1], [-1,dim_data0[1],dim_data0[2],dim_data0[3]], name='reshape0'))

test_data1  = test_io.fetch_data('test_image1')
if test_data1 is not None:
  dim_data1    = test_data1.dim()
  data_tensors.append(tf.placeholder(tf.float32, [None, dim_data1[1] * dim_data1[2] * dim_data1[3]], name='image1'))
  data_tensors_2d.append(tf.reshape(data_tensors[-1], [-1,dim_data1[1],dim_data1[2],dim_data1[3]], name='reshape1'))

# retrieve dimensions of label for network construction
dim_label    = test_io.fetch_data('test_label').dim()
label_tensor = tf.placeholder(tf.float32, [None, dim_label[1]], name='label')

# Let's keep 10 random set of images in the log
#tf.summary.image('input',data_tensor_2d0,10)
# build net
net = build(input_tensors=data_tensors_2d, num_class=int(dim_label[1]), trainable=False, debug=False)
softmax_op = tf.nn.softmax(logits=net)
# Create a session                                                                                                                     
sess = tf.InteractiveSession()
# Initialize variables                                                                                                                 
sess.run(tf.global_variables_initializer())
# Create weights loader
loader = tf.train.Saver()
loader.restore(sess, WEIGHTS)

csv_filename='1plane_inference.csv'

fout=open(csv_filename,'w')
# Basic information in csv file
fout.write('entry,run,subrun,event,label,prediction,probability')
# More information in csv file
fout.write(',kaon_ke,proton_ke,muon_ke')
fout.write('\n')
ctr = 0
num_events = test_io.fetch_n_entries()
while ctr < num_events:
  if ctr%100 == 0: print ctr
  feed_dict = {}
  if test_data0 is not None: feed_dict[data_tensors[0]] = test_io.fetch_data('test_image0').data()
  if test_data1 is not None: feed_dict[data_tensors[1]] = test_io.fetch_data('test_image1').data()
  test_label = test_io.fetch_data('test_label').data()
  feed_dict[label_tensor] = test_label
  
  softmax_batch     = sess.run(softmax_op, feed_dict=feed_dict)
  processed_events  = test_io.fetch_event_ids()
  processed_entries = test_io.fetch_entries()
  
  for j in xrange(len(softmax_batch)):
    softmax_array = softmax_batch[j]
    entry         = processed_entries[j]
    event_id      = processed_events[j]
    label = np.argmax(test_label[j])
    prediction      = np.argmax(softmax_array)
    prediction_prob = softmax_array[prediction]
    
    # Basic information in csv file
    data_string = '%d,%d,%d,%d,%d,%d,%g' % (entry,event_id.run(),event_id.subrun(),event_id.event(), label, prediction, prediction_prob)
    # More information in csv file
    ana_chain.GetEntry(entry)
    event_particle = ana_chain.particle_mctruth_branch #used to be mctruth
    
    (kaon_ke,proton_ke,muon_ke) = (-1.,-1.,-1.)
    for p in event_particle.as_vector():
      if p.pdg_code() == 321:
        kaon_ke = p.energy_init()
      elif p.pdg_code() == -13:
        muon_ke = p.energy_init()
      elif p.pdg_code() == 2212:
        proton_ke = p.energy_init()
    data_string += ',%g,%g,%g\n' % (kaon_ke,proton_ke,muon_ke)
    fout.write(data_string)

    ctr += 1
    if ctr == num_events:
      break
    
  if ctr == num_events:
    break

  test_io.next(store_entries=True,store_event_ids=True)

test_io.reset()
fout.close()
