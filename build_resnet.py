import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.python.platform
import numpy as np
import sys, time

def build(input_tensors,num_class,filters=32,trainable=True,reuse=False,debug=False):

    nets = []
    # Build an individual tower
    for index,tensor in enumerate(input_tensors):
        with tf.variable_scope('tower%d' % index):
            net = build_tower(input_tensor=tensor,filters=filters,trainable=trainable,debug=debug)
        nets.append(net)

    # If multiple towers, combine
    if len(nets) > 1:
        net = tf.concat(nets,axis=3,name='siamese_concat')
        if debug: print('after concat:', net.shape)
        net = resnet_layer(input_tensor = net,
                           num_outputs  = net.get_shape()[-1].value/2,
                           num_layers   = 3,
                           trainable    = trainable,
                           kernel       = 3,
                           stride       = 1,
                           scope        = 'siamese_conv',
                           debug        = debug)

    # average pooling
    net = slim.avg_pool2d(inputs      = net,
                          kernel_size = [net.get_shape()[-3].value,net.get_shape()[-2].value],
                          stride      = 1,
                          scope       = 'avg_pool2d')
    if debug: print('after avg_pool2d tensor shape', net.shape)
    
    # flatten
    net = slim.flatten(net,scope='flatten')
    if debug: print('after flatten tensor shape', net.shape)

    keep_prob = 1.0
    if trainable: keep_prob = 0.5
    net = slim.dropout(inputs    = net,
                       keep_prob = keep_prob,
                       scope     = 'dropout0')

    net = slim.fully_connected(inputs      = net,
                               num_outputs = 1024,
                               trainable   = trainable,
                               scope       = 'fc0')
    if debug: print('after fc0 tensor shape', net.shape)
    
    net = slim.dropout(inputs    = net,
                       keep_prob = keep_prob,
                       scope     = 'dropout1')
    
    net = slim.fully_connected(inputs      = net,
                               num_outputs = 2048,
                               trainable   = trainable,
                               scope       = 'fc1')
    if debug: print('after fc1 tensor shape', net.shape)

    net = slim.fully_connected(inputs      = net,
                               num_outputs = num_class,
                               trainable   = trainable,
                               scope       = 'classification')
    
    return net

def resnet_module(input_tensor, num_outputs, trainable=True, kernel=3, stride=1, scope='noscope', debug=False):

    fn_conv = slim.conv2d
    if len(input_tensor.shape) == 5:
        fn_conv = slim.conv3d

    num_inputs  = input_tensor.get_shape()[-1].value
    with tf.variable_scope(scope):
        #
        # shortcut path
        #
        shortcut = None
        if num_outputs == num_inputs and stride == 1 :
            shortcut = input_tensor
        else:
            shortcut = fn_conv(inputs      = input_tensor,
                               num_outputs = num_outputs,
                               kernel_size = 1,
                               stride      = stride,
                               trainable   = trainable,
                               padding     = 'same',
                               normalizer_fn = slim.batch_norm, 
                               activation_fn = None,
                               #reuse       = tf.AUTO_REUSE,
                               scope       = 'shortcut')

        if debug: print('%s shortcut tensor shape' % scope, shortcut.shape)
        
        #
        # residual path
        #
        residual = input_tensor
        residual = fn_conv(inputs      = residual,
                           num_outputs = num_outputs,
                           kernel_size = kernel,
                           stride      = stride,
                           trainable   = trainable,
                           padding     = 'same',
                           normalizer_fn = slim.batch_norm,
                           activation_fn = None,
                           #reuse       = tf.AUTO_REUSE,
                           scope       = 'resnet_conv1')
        residual = fn_conv(inputs      = residual,
                           num_outputs = num_outputs,
                           kernel_size = kernel,
                           stride      = 1,
                           trainable   = trainable,
                           padding     = 'same',
                           normalizer_fn = slim.batch_norm,
                           activation_fn = None,
                           #reuse       = tf.AUTO_REUSE,
                           scope       = 'resnet_conv2')

        if debug: print('%s residual tensor shape' % scope, residual.shape)
        
        return tf.nn.relu(shortcut + residual)

def resnet_layer(input_tensor, num_layers, num_outputs, trainable=True, kernel=3, stride=1, scope='noscope', debug=False):

    net = input_tensor
    with tf.variable_scope(scope):

        for i in np.arange(num_layers):

            if i>0: stride = 1
            net = resnet_module(input_tensor=net,
                                trainable=trainable,
                                kernel=kernel,
                                stride=stride,
                                num_outputs=num_outputs,
                                scope='module%d' % i,
                                debug=debug)
        
    return net

def build_tower(input_tensor,filters=32,trainable=True,debug=False):

    net = input_tensor
    if debug: print('input tensor:', input_tensor.shape)
    #
    # More stuff!
    #
    NUM_STAGES=5
    for i in np.arange(NUM_STAGES):
        stage_name = 'stage%d' % i
        net =  resnet_layer(input_tensor  = net,
                            num_outputs   = filters,
                            num_layers    = 3,
                            kernel        = 3,
                            stride        = 2,
                            trainable     = trainable,
                            scope         = stage_name,
                            debug         = debug)

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

