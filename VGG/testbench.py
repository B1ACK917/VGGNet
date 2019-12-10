import tensorflow as tf
import numpy as np
import time
from neural_network.VGG import VGG
import os


def run_benchmark(testname, name='VGG16', batch_size=16, iteration=100):
    with tf.Graph().as_default():
        image_size = 224
        images = np.random.randn(batch_size, image_size * image_size * 3)
        labels = np.random.randn(batch_size, 17)
        x, label, y, drop_rate = VGG.model(ynum=17, name=name, show_flowshape=True)
        cross_entropy = -tf.reduce_sum(label * tf.math.log(y))
        train_step = tf.compat.v1.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        init = tf.compat.v1.global_variables_initializer()
        timeperf_rec = []
        with tf.compat.v1.Session() as sess:
            sess.run(init)
            for i in range(iteration + 10):
                if i < 10:
                    _ = sess.run(train_step, feed_dict={x: images, label: labels, drop_rate: 0.5})
                else:
                    start = time.perf_counter()
                    if testname == 'forward':
                        _ = cross_entropy.eval(feed_dict={x: images, label: labels, drop_rate: 0})
                    elif testname == 'forward_backward':
                        _ = sess.run(train_step, feed_dict={x: images, label: labels, drop_rate: 0.5})
                    else:
                        raise RuntimeError('Unknown testbench mode : %s' % testname)
                    timeperf_rec.append(time.perf_counter() - start)
        print('Testbench mode is %s, with batchsize = %d, iteration = %d' % (testname, batch_size, iteration))
        print('Max iteration duration is %gs\nMin iteration duration is %gs\nMean iteration duration is %gs' % (
            max(timeperf_rec), min(timeperf_rec), float(np.mean(timeperf_rec))))
        print('Testbench running completed\n\n')


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    with tf.device("/cpu:0"):
        run_benchmark(testname='forward_backward', name='VGG16', batch_size=6, iteration=20)
        run_benchmark(testname='forward', name='VGG16', batch_size=6, iteration=20)
        run_benchmark(testname='forward_backward', name='VGG19', batch_size=6, iteration=20)
        run_benchmark(testname='forward', name='VGG19', batch_size=6, iteration=20)
