import tensorflow as tf

FLOW_PATH = ['conv3_64', 'maxpool', 'conv3_128', 'maxpool', 'conv3_256', 'maxpool', 'conv3_512', 'maxpool',
             'conv3_512', 'maxpool', 'FC_4096', 'FC_4096', 'FC_1000', 'softmax']
LAYER_NUMS = {
    'conv3_64': 2,
    'conv3_128': 2,
    'conv3_256': 3,
    'conv3_512': 3,
    'maxpool': 1,
    'FC_4096': 1,
    'FC_1000': 1,
    'softmax': 1
}
GLOBAL_DICT = {}


def wei_mat(shape):
    ini = tf.random.truncated_normal(shape, stddev=0.1)
    return tf.Variable(ini)


def bias_mat(shape):
    ini = tf.constant(0.1, dtype=tf.float32, shape=shape)
    return tf.Variable(ini)


def conv_2d(x_, W):
    return tf.nn.conv2d(x_, W, [1, 1, 1, 1], padding='SAME')


def max_pool(x_):
    return tf.nn.max_pool2d(x_, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def mean_pool(x_):
    return tf.nn.avg_pool2d(x_, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def multi(x_, W):
    return tf.matmul(x_, W)


def layer(name, _x, drop_rate):
    global GLOBAL_DICT
    if name[:4] == 'conv':
        conv_kernel_size = int(name[4])
        output_channel_num = int(name[6:])
        for num in range(LAYER_NUMS[name]):
            if num:
                GLOBAL_DICT['SHAPE'] = (GLOBAL_DICT['SHAPE'][0], GLOBAL_DICT['SHAPE'][1], output_channel_num)
            with tf.compat.v1.variable_scope(name + str(num)):
                W = wei_mat([conv_kernel_size, conv_kernel_size, GLOBAL_DICT['SHAPE'][2], output_channel_num])
                B = bias_mat([output_channel_num])
                _x = tf.nn.relu(conv_2d(_x, W) + B)
        return _x
    elif name == 'maxpool':
        a, b = int(float(GLOBAL_DICT['SHAPE'][0]) / 2 + 0.5), int(float(GLOBAL_DICT['SHAPE'][1]) / 2 + 0.5)
        GLOBAL_DICT['SHAPE'] = (a, b, GLOBAL_DICT['SHAPE'][2])
        return max_pool(_x)
    elif name[:2] == 'FC':
        if not GLOBAL_DICT['IS_FLATTENED']:
            GLOBAL_DICT['FLATTENED_SHAPE'] = GLOBAL_DICT['SHAPE'][0] * GLOBAL_DICT['SHAPE'][1] * GLOBAL_DICT['SHAPE'][2]
            _x = tf.reshape(_x, [-1, GLOBAL_DICT['FLATTENED_SHAPE']])
            GLOBAL_DICT['IS_FLATTENED'] = True
        with tf.compat.v1.variable_scope(name):
            bias_num = int(name[3:])
            W = wei_mat([GLOBAL_DICT['FLATTENED_SHAPE'], bias_num])
            B = bias_mat([bias_num])
            Out = tf.nn.relu(multi(_x, W) + B)
            Out_dropout = tf.nn.dropout(Out, rate=drop_rate)
            GLOBAL_DICT['FLATTENED_SHAPE'] = bias_num
            return Out_dropout
    elif name == 'softmax':
        with tf.compat.v1.variable_scope(name):
            W = wei_mat([GLOBAL_DICT['FLATTENED_SHAPE'], GLOBAL_DICT['CLASSES']])
            B = bias_mat([GLOBAL_DICT['CLASSES']])
            y = tf.nn.softmax(multi(_x, W) + B)
            return y
    return None


def model(ynum, name='VGG16', show_flowshape=False):
    global FLOW_PATH, GLOBAL_DICT, LAYER_NUMS
    GLOBAL_DICT = {
        'IS_FLATTENED': False,
        'FLATTENED_SHAPE': 0,
        'SHAPE': (224, 224, 3),
        'CLASSES': 17,
        'BATCH_SIZE': 64
    }
    if name == 'VGG19':
        LAYER_NUMS['conv3_256'], LAYER_NUMS['conv3_512'] = 4, 4
    x = tf.compat.v1.placeholder(tf.float32,
                                 [None, GLOBAL_DICT['SHAPE'][0] * GLOBAL_DICT['SHAPE'][1] * GLOBAL_DICT['SHAPE'][2]])
    label = tf.compat.v1.placeholder(tf.float32, [None, ynum])
    var = tf.reshape(x, [-1, GLOBAL_DICT['SHAPE'][0], GLOBAL_DICT['SHAPE'][1], GLOBAL_DICT['SHAPE'][2]])
    drop_rate = tf.compat.v1.placeholder(tf.float32)
    if show_flowshape:
        print('Input tensor shape : ' + str(var.shape))

    for FLOW in FLOW_PATH:
        var = layer(FLOW, var, drop_rate)
        print('Operation is %s ,repeat %d times, tensor shape : ' % (FLOW, LAYER_NUMS[FLOW]) + str(var.shape))

    return x, label, var, drop_rate
