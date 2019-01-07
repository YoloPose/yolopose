import tensorflow as tf
from tensorflow.contrib.layers import variance_scaling_initializer as VarianceScaling
from tensorflow.contrib.layers import l2_regularizer as l2

import numpy as np
import cv2

class Inception():
    def __init__(self):
        self.channel_axis = -1

        self.build_graph()
        return

    def conv2d_bn(self, x, nb_filter, num_row, num_col, padding='SAME', strides=(1,1), use_bias=True):
        x = tf.layers.Conv2D(nb_filter, (num_row, num_col),
                             strides=strides,
                             padding=padding,
                             use_bias=use_bias,
                             kernel_regularizer=l2(0.00004),
                             kernel_initializer=VarianceScaling(factor=2.0, mode='FAN_IN',
                                                                             uniform='normal',
                                                                             seed=None))(x)

        x = tf.layers.batch_normalization(inputs=x, axis=self.channel_axis, momentum=0.9997, scale=False)
        x = tf.nn.relu(x)
        return x

    def block_inception_a(self, input):
        branch_0 = self.conv2d_bn(input, 96, 1, 1)

        branch_1 = self.conv2d_bn(input, 64, 1, 1)
        branch_1 = self.conv2d_bn(branch_1, 96, 3, 3)

        branch_2 = self.conv2d_bn(input, 64, 1, 1)
        branch_2 = self.conv2d_bn(branch_2, 96, 3, 3)
        branch_2 = self.conv2d_bn(branch_2, 96, 3, 3)

        branch_3 = tf.layers.average_pooling2d(input, (3, 3), strides=(1, 1), padding='SAME')
        branch_3 = self.conv2d_bn(branch_3, 96, 1, 1)

        x =  tf.concat([branch_0, branch_1, branch_2, branch_3], axis=self.channel_axis)
        return x

    def block_reduction_a(self, input):
        branch_0 = self.conv2d_bn(input, 384, 3, 3, strides=(2, 2), padding='VALID')

        branch_1 = self.conv2d_bn(input, 192, 1, 1)
        branch_1 = self.conv2d_bn(branch_1, 224, 3, 3)
        branch_1 = self.conv2d_bn(branch_1, 256, 3, 3, strides=(2, 2), padding='VALID')

        branch_2 = tf.layers.max_pooling2d(input, (3, 3), strides=(2, 2), padding='VALID')

        x = tf.concat([branch_0, branch_1, branch_2], axis=self.channel_axis)
        return x

    def block_inception_b(self, input):
        branch_0 = self.conv2d_bn(input, 384, 1, 1)

        branch_1 = self.conv2d_bn(input, 192, 1, 1)
        branch_1 = self.conv2d_bn(branch_1, 224, 1, 7)
        branch_1 = self.conv2d_bn(branch_1, 256, 7, 1)

        branch_2 = self.conv2d_bn(input, 192, 1, 1)
        branch_2 = self.conv2d_bn(branch_2, 192, 7, 1)
        branch_2 = self.conv2d_bn(branch_2, 224, 1, 7)
        branch_2 = self.conv2d_bn(branch_2, 224, 7, 1)
        branch_2 = self.conv2d_bn(branch_2, 256, 1, 7)

        branch_3 = tf.layers.average_pooling2d(input, (3, 3), strides=(1, 1), padding='SAME')
        branch_3 = self.conv2d_bn(branch_3, 128, 1, 1)

        x = tf.concat([branch_0, branch_1, branch_2, branch_3], axis=self.channel_axis)
        return x

    def block_reduction_b(self, input):
        branch_0 = self.conv2d_bn(input, 192, 1, 1)
        branch_0 = self.conv2d_bn(branch_0, 192, 3, 3, strides=(2, 2), padding='VALID')

        branch_1 = self.conv2d_bn(input, 256, 1, 1)
        branch_1 = self.conv2d_bn(branch_1, 256, 1, 7)
        branch_1 = self.conv2d_bn(branch_1, 320, 7, 1)
        branch_1 = self.conv2d_bn(branch_1, 320, 3, 3, strides=(2, 2), padding='VALID')

        branch_2 = tf.layers.average_pooling2d(input, (3, 3), strides=(2, 2), padding='VALID')

        x = tf.concat([branch_0, branch_1, branch_2], axis=self.channel_axis)
        return x

    def block_inception_c(self, input):
        branch_0 = self.conv2d_bn(input, 256, 1, 1)

        branch_1 = self.conv2d_bn(input, 384, 1, 1)
        branch_10 = self.conv2d_bn(branch_1, 256, 1, 3)
        branch_11 = self.conv2d_bn(branch_1, 256, 3, 1)
        branch_1 = tf.concat([branch_10, branch_11], axis=self.channel_axis)

        branch_2 = self.conv2d_bn(input, 384, 1, 1)
        branch_2 = self.conv2d_bn(branch_2, 448, 3, 1)
        branch_2 = self.conv2d_bn(branch_2, 512, 1, 3)
        branch_20 = self.conv2d_bn(branch_2, 256, 1, 3)
        branch_21 = self.conv2d_bn(branch_2, 256, 3, 1)
        branch_2 = tf.concat([branch_20, branch_21], axis=self.channel_axis)

        branch_3 = tf.layers.average_pooling2d(input, (3, 3), strides=(1, 1), padding='SAME')
        branch_3 = self.conv2d_bn(branch_3, 256, 1, 1)

        x = tf.concat([branch_0, branch_1, branch_2, branch_3], axis=self.channel_axis)
        return x

    def inception_base(self, input):

        # Input Shape is 299 x 299 x 3 (th) or 3 x 299 x 299 (th)
        net = self.conv2d_bn(input, 32, 3, 3, strides=(2,2), padding='VALID')
        net = self.conv2d_bn(net, 32, 3, 3, padding='VALID')

        net = self.conv2d_bn(net, 64, 3, 3)

        branch_0 = tf.layers.max_pooling2d(net, (3, 3), strides=(2, 2), padding='VALID')

        branch_1 = self.conv2d_bn(net, 96, 3, 3, strides=(2, 2), padding='VALID')

        net = tf.concat([branch_0, branch_1], axis=self.channel_axis)

        branch_0 = self.conv2d_bn(net, 64, 1, 1)
        branch_0 = self.conv2d_bn(branch_0, 96, 3, 3, padding='valid')

        branch_1 = self.conv2d_bn(net, 64, 1, 1)
        branch_1 = self.conv2d_bn(branch_1, 64, 1, 7)
        branch_1 = self.conv2d_bn(branch_1, 64, 7, 1)
        branch_1 = self.conv2d_bn(branch_1, 96, 3, 3, padding='VALID')

        net = tf.concat([branch_0, branch_1], axis=self.channel_axis)

        branch_0 = self.conv2d_bn(net, 192, 3, 3, strides=(2, 2), padding='VALID')
        branch_1 = tf.layers.max_pooling2d(net, (3, 3), strides=(2, 2), padding='VALID')

        net = tf.concat([branch_0, branch_1], axis=self.channel_axis)

        # 35 x 35 x 384
        # 4 x Inception-A blocks
        for idx in range(4):
            net = self.block_inception_a(net)

        # 35 x 35 x 384
        # Reduction-A block
        net = self.block_reduction_a(net)

        # 17 x 17 x 1024
        # 7 x Inception-B blocks
        for idx in range(7):
            net = self.block_inception_b(net)

        # 17 x 17 x 1024
        # Reduction-B block
        net = self.block_reduction_b(net)

        # 8 x 8 x 1536
        # 3 x Inception-C blocks    def
        for idx in range(3):
            net = self.block_inception_c(net)

        return net

    def build_graph(self):
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        with self.sess.as_default():
            with self.graph.as_default():
                self.input_x = tf.placeholder(tf.float32, (None, 299, 299, 3), 'input_x')
                self.label_y = tf.placeholder(tf.float32, (None, 32), 'label_y')
                self.correcter = tf.placeholder(tf.float32, (None, 1))

                x = self.inception_base(self.input_x)

                avg_x = tf.layers.average_pooling2d(x, (2, 2), strides=(2, 2), padding='same')
                drop_x = tf.nn.dropout(avg_x, 0.9)

                flatten = tf.layers.flatten(drop_x)

                fc_1 = tf.layers.dense(flatten, 4096, activation=tf.nn.tanh, use_bias=True,
                                    kernel_initializer=None)
                dp_x = tf.nn.dropout(fc_1, 0.9)

                fc_2 = tf.layers.dense(dp_x, 4096, activation=tf.nn.tanh, use_bias=True,
                                    kernel_initializer=None)
                dp_2 = tf.nn.dropout(fc_2, 0.9)

                self.y_hat = tf.layers.dense(dp_2, 32, activation=None, use_bias=True,
                                    kernel_initializer=None)
                self.loss = tf.reduce_mean( tf.sqrt( tf.square(self.y_hat - self.label_y) ) / self.correcter )
                self.optimizer = tf.train.GradientDescentOptimizer(0.0001)
                self.train_step = self.optimizer.minimize(self.loss)
                self.saver = tf.train.Saver()
                self.saver.restore(self.sess, './weights/10.ckpt')
                print("model loaded")
        return

    
    def detect(self, image):
        print("image shape: ",image.shape)
        image = cv2.resize(image, (299,299) )
        input = np.resize(image, (1, 299, 299, 3)) / 255 - 0.5
        with self.sess.as_default():
            with self.graph.as_default():
                feed_dict = {self.input_x: input}
                pred = self.sess.run(self.y_hat, feed_dict=feed_dict)
        return pred


