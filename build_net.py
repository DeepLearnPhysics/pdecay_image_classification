from resnet_module import double_resnet
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.python.platform
import numpy as np
import sys, time

def build(input_tensors,filters=32,trainable=True,reuse=False,debug=False):

    nets = []
    # Build an individual tower
    for index,tensor in enumerate(input_tensors):
        prefix = ''
        if reuse:
            prefix = 'tower%d' % index
        net = build_tower(input_tensor=tensor,filters=filters,trainable=trainable,prefix=prefix,debug=debug)
        nets.append(net)

    # Now combine
    net = tf.concat(nets,axis=3,name='siamese_concat')
    if debug: print('after concat:', net.shape)
    net = build_stage(input_tensor = net,
                      filters      = net.get_shape()[-1].value / 2,
                      trainable    = trainable,
                      reuse        = tf.AUTO_REUSE,
                      stage_prefix = 'final',
                      stride       = 2,
                      kernel       = 2)
    return net

def build_stage(input_tensor,filters,trainable,reuse,stage_prefix,stride=2,kernel=3):

    net =  slim.conv2d(inputs        = input_tensor,
                       num_outputs   = filters,
                       kernel_size   = kernel,
                       stride        = stride,
                       trainable     = trainable,
                       activation_fn = tf.nn.relu,
                       normalizer_fn = slim.batch_norm,
                       reuse         = reuse,
                       scope         = '%s_0' % stage_prefix)
    
    net =  slim.conv2d(inputs        = net,
                       num_outputs   = filters,
                       kernel_size   = kernel,
                       stride        = 1,
                       trainable     = trainable,
                       activation_fn = tf.nn.relu,
                       normalizer_fn = slim.batch_norm,
                       reuse         = reuse,
                       scope         = '%s_1' % stage_prefix)

    return net

def build_tower(input_tensor,filters=32,trainable=True,prefix='',debug=False):

    net = input_tensor
    if debug: print('input tensor:', input_tensor.shape)
    reuse = tf.AUTO_REUSE
    if prefix: reuse=False
    #
    # More stuff!
    #
    for i in np.arange(5):
        stage_prefix = 'conv%d' % i
        if prefix: stage_prefix = prefix + '_' + stage_prefix

        net = build_stage(input_tensor = net,
                          filters      = filters,
                          trainable    = trainable,
                          reuse        = reuse,
                          stage_prefix = stage_prefix,
                          stride       = 2,
                          kernel       = 3)

        filters *= 2
        
        if debug: print('tensor shape after stage %d:' % i, net.shape)

    return net

if __name__ == '__main__':
    image0 = tf.placeholder(tf.float32, [10,64,64,1], name='image0')
    image1 = tf.placeholder(tf.float32, [10,64,64,1], name='image1')
    images = [image0,image1]

    reuse=False
    if 'reuse' in sys.argv:
        reuse=True    
    net    = build(images,filters=32,trainable=True,reuse=reuse,debug=True)
    print('Final net shape:', net.shape)

    if 'save' in sys.argv:
        # Create a session
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        # Create a summary writer handle + save graph
        writer=tf.summary.FileWriter('siamese_graph')
        writer.add_graph(sess.graph)
        writer.flush()
        writer.close()

