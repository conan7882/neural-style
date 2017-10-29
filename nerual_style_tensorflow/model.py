# File: VGG.py
# Author: Qian Ge <geqian1001@gmail.com>

import numpy as np
import scipy.misc
import tensorflow as tf

from tensorcv.models.layers import *
from tensorcv.models.base import BaseModel
from tensorcv.dataflow.common import load_image

from tensorcv.algorithms.pretrained.VGG import VGG19_FCN



class NerualStyle(BaseModel):
    """ base of VGG class """
    def __init__(self,
                 im_height=224, im_width=224,
                 num_channels=3,
                 learning_rate=1,
                 pre_train_path=None,
                 is_rescale=False,
                 init_im=None):
        """ 
        Args:
            num_class (int): number of image classes
            num_channels (int): number of input channels
            im_height, im_width (int): size of input image
                               Can be unknown when testing.
            learning_rate (float): learning rate of training
        """

        self._lr = learning_rate
        self._nchannel = num_channels
        self._h = im_height
        self._w = im_width
        self._is_rescale = is_rescale

        self._init_im = init_im

        self.layer = {}

        if pre_train_path is None:
            raise ValueError('pre_train_path can not be None!')
        self._pre_train_path = pre_train_path

        self.set_is_training(True)

    def _create_input(self):
        # self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        # content image
        self.c_im = tf.placeholder(tf.float32, name='content_image',
                            shape=[1, self._h, self._w, self._nchannel])
        # style image
        self.s_im = tf.placeholder(tf.float32, name='style_image',
                            shape=[1, None, None, self._nchannel])


        self.set_model_input([self.c_im, self.s_im])
        # self.set_dropout(self.keep_prob, keep_prob=0.5)
        self.set_train_placeholder([self.c_im, self.s_im])
        self.set_prediction_placeholder([self.c_im, self.s_im])

    def _create_model(self):

        self.content_layers = ['conv4_2']
        self.style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']

        c_im = self.model_input[0]
        s_im = self.model_input[1]

        vgg_model = VGG19_FCN(is_load=True, is_rescale=self._is_rescale, pre_train_path=self._pre_train_path, trainable=False)

        vgg_model.create_model([c_im, 1])
        # self.c_im_layer = vgg_model.layer
        self.c_im_layer = {}
        for layer_name in self.content_layers:
            self.c_im_layer[layer_name] = vgg_model.layer[layer_name]
        
        with tf.variable_scope(tf.get_variable_scope()) as scope:
            scope.reuse_variables()
            vgg_model.create_model([s_im, 1])
        self.s_im_layer = {}
        for layer_name in self.style_layers:
            self.s_im_layer[layer_name] = vgg_model.layer[layer_name]

        with tf.variable_scope('random_input'):
            shape=[1, self._h, self._w, self._nchannel]
            # noise = np.random.normal(size=shape, scale=np.std(c_im) * 0.1)
            # initializer = tf.constant_initializer(c_im)
            initializer = tf.random_normal_initializer(stddev = 0.2)
            self.random_im = tf.get_variable('random_im', 
                                    # shape=shape, 
                                  initializer=self._init_im.astype('float32'),
                                  trainable=True) 
            # weight_decay = tf.multiply(tf.nn.l2_loss(self.random_im), 0.002, name='weight_loss')
            # tf.add_to_collection('losses', weight_decay)


        # g = tf.get_default_graph()
        # with g.gradient_override_map({'Relu': 'GuidedRelu'}):
        with tf.variable_scope(tf.get_variable_scope()) as scope:
            scope.reuse_variables()
            vgg_model.create_model([self.random_im, 1])

        self.random_im_layer = {}
        for layer_name in self.content_layers + self.style_layers:
            self.random_im_layer[layer_name] = vgg_model.layer[layer_name]

        # out_r_im = tf.identity(random_im_layer['conv1_2'], name='r_im_out')

    def _get_loss(self):
        with tf.name_scope('loss'):
            # for idx, layer_name in enumerate(self.content_layers):
            #     side_cost = self._layer_content_loss(layer_name)*tf.constant(1e-6, tf.float32)
            #     tf.add_to_collection('losses', side_cost)
            #     tf.summary.scalar("content_loss_{}".format(idx), side_cost, collections=['train'])

            for idx, layer_name in enumerate(self.style_layers):
                side_cost = self._layer_style_loss(layer_name)*tf.constant(0.2, tf.float32)
                tf.add_to_collection('losses', side_cost)
                tf.summary.scalar("style_loss_{}".format(idx), side_cost, collections=['train'])

            self.check_loss = tf.add_n(tf.get_collection('losses'), name='result') 

            return self.check_loss 

    def _get_optimizer(self):

        BETA1 = 0.9
        BETA2 = 0.999
        EPSILON = 1e-08
        return tf.train.AdamOptimizer(learning_rate=self._lr, beta1=0.9, beta2=0.999, epsilon=1e-08)


    def _layer_content_loss(self, layer_name):
        """ 
        Args:
            layer_name (str): name of the layer to be used to compute content loss
        """
        return tf.losses.mean_squared_error(self.c_im_layer[layer_name], 
                                           self.random_im_layer[layer_name], 
                                           scope='content_loss_{}'.format(layer_name))

    def _layer_style_loss(self, layer_name):
        h = tf.cast(tf.shape(self.s_im_layer[layer_name])[1], tf.float32)
        w = tf.cast(tf.shape(self.s_im_layer[layer_name])[2], tf.float32)

        n_filter = self.s_im_layer[layer_name].shape.as_list()[-1]
        s_flatten_layer = tf.reshape(self.s_im_layer[layer_name], shape=[-1, n_filter])
        s_G = tf.matmul(tf.transpose(s_flatten_layer), s_flatten_layer)/(h*w)
        print(s_flatten_layer.shape)

        h = tf.cast(tf.shape(self.random_im_layer[layer_name])[1], tf.float32)
        w = tf.cast(tf.shape(self.random_im_layer[layer_name])[2], tf.float32)
        random_flatten_layer = tf.reshape(self.random_im_layer[layer_name], shape=[-1, n_filter])
        random_G = tf.matmul(tf.transpose(random_flatten_layer), random_flatten_layer)/(h*w)
        print(random_G.shape)

        return tf.losses.mean_squared_error(s_G, random_G, 
                                           scope='style_loss_{}'.format(layer_name))


    def get_train_op(self):
        opt = self.get_optimizer()
        loss = self.get_loss()
        grads = opt.compute_gradients(loss)
        return opt.apply_gradients(grads, name='train')



if __name__ == '__main__':


    VGG_PATH = 'E:\\GITHUB\\workspace\\CNN\\pretrained\\vgg19.npy'
    SAVE_DIR = 'E:\\GITHUB\\workspace\\CNN\\result\\'
    STYLE_PATH = 'E:\\GITHUB\\workspace\\CNN\\dataset\\la_muse.jpg'
    CONTENT_PATH = 'E:\\GITHUB\\workspace\\CNN\\dataset\\test_2.png'
    # IM_PATH = 'E:\\GITHUB\\workspace\\CNN\\dataset\\test3\\134_0095.jpg'

    # VGG_PATH = 'D:\\Qian\\GitHub\\workspace\\VGG\\vgg19.npy'
    # STYLE_PATH = 'D:\\Qian\\GitHub\\workspace\\t\\mo.jpg'
    # CONTENT_PATH = 'D:\\Qian\\GitHub\\workspace\\VGG\\test_2.png'
    # SAVE_DIR = 'D:\\Qian\\GitHub\\workspace\\t\\'

    s_im = load_image(STYLE_PATH, read_channel=3)
    c_im = load_image(CONTENT_PATH, read_channel=3)

    c_h = c_im.shape[1]
    c_w = c_im.shape[2]

    style_trans_model = NerualStyle(pre_train_path=VGG_PATH, im_height=c_h, im_width=c_w, init_im=c_im)
    
    style_trans_model.create_graph()
    train_op = style_trans_model.get_train_op()
    loss_op = style_trans_model.check_loss

    gen_im = style_trans_model.random_im
    

    writer = tf.summary.FileWriter(SAVE_DIR)
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        writer.add_graph(sess.graph)

        cnt = 0
        while cnt < 10000:
            _, content_im, loss = sess.run([train_op, gen_im, loss_op], feed_dict = {style_trans_model.c_im: c_im, style_trans_model.s_im: s_im})
            print(loss)
            if cnt % 100 == 0:

                scipy.misc.imsave('{}test_{}.png'.format(SAVE_DIR, cnt), np.squeeze(content_im))
            print(cnt)
            cnt += 1


    writer.close()



 