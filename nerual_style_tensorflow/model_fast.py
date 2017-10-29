# File: VGG.py
# Author: Qian Ge <geqian1001@gmail.com>

import numpy as np
from scipy.optimize import fmin_l_bfgs_b
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
                 learning_rate=0.1,
                 pre_train_path=None,
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

        # self._init_im = init_im

        self.layer = {}

        if pre_train_path is None:
            raise ValueError('pre_train_path can not be None!')
        self._pre_train_path = pre_train_path

        self.set_is_training(True)

    def _create_input(self):
        # content image
        self.c_im = tf.placeholder(tf.float32, name='content_image',
                            shape=[1, self._h, self._w, self._nchannel])
        # style image
        self.s_im = tf.placeholder(tf.float32, name='style_image',
                            shape=[1, None, None, self._nchannel])

        self.set_model_input([self.c_im, self.s_im])
        self.set_train_placeholder([self.c_im, self.s_im])
        self.set_prediction_placeholder([self.c_im, self.s_im])

    def _create_model(self):
        self.content_layers = ['conv4_2']
        self.style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']

        c_im = self.model_input[0]
        s_im = self.model_input[1]

        vgg_model = VGG19_FCN(is_load=True, trainable=False,
                     pre_train_path=self._pre_train_path)

        vgg_model.create_conv([c_im, 1])
        # self.c_im_layer = vgg_model.layer
        self.c_im_layer = {}
        for layer_name in self.content_layers:
            print(layer_name)
            self.c_im_layer[layer_name] = vgg_model.layer[layer_name]
        
        with tf.variable_scope(tf.get_variable_scope()) as scope:
            scope.reuse_variables()
            vgg_model.create_conv([s_im, 1])
        self.s_im_layer = {}
        for layer_name in self.style_layers:
            self.s_im_layer[layer_name] = vgg_model.layer[layer_name]

        with tf.variable_scope('random_input'):
            shape=[1, self._h, self._w, self._nchannel]
            mean, std = tf.nn.moments(self.c_im, axes=[0,1,2,3])
            # initializer = tf.random_normal_initializer(stddev=std, mean=mean)
            init_im = np.random.uniform(0, 255, (1, self._h, self._w, 3)) - 128.
            initializer = tf.random_normal_initializer(stddev=0.02, mean=128)
            self.random_im = tf.get_variable('random_im', 
                                  initializer=self.c_im,
                                  trainable=True)
            # self.random_im = tf.get_variable('random_im', 
            #                       shape=shape,
            #                       initializer=initializer,
            #                       trainable=True) 
 

        with tf.variable_scope(tf.get_variable_scope()) as scope:
            scope.reuse_variables()
            vgg_model.create_conv([self.random_im, 1])

        self.random_im_layer = {}
        for layer_name in self.content_layers + self.style_layers:
            self.random_im_layer[layer_name] = vgg_model.layer[layer_name]

    def comp_context_style_feats(self):
        self.c_feats = {}
        self.s_feats = {}
        for idx, layer in enumerate(self.content_layers):
            print(layer)
            self.c_feats[layer] = tf.get_variable('content_{}'.format(idx), 
                                  initializer=self.c_im_layer[layer],
                                  trainable=False) 

        for idx, layer in enumerate(self.style_layers):
            h = tf.cast(tf.shape(self.s_im_layer[layer])[1], tf.float32)
            w = tf.cast(tf.shape(self.s_im_layer[layer])[2], tf.float32)

            n_filter = self.s_im_layer[layer].shape.as_list()[-1]
            s_flatten_layer = tf.reshape(self.s_im_layer[layer], shape=[-1, n_filter])
            s_G = tf.matmul(tf.transpose(s_flatten_layer), s_flatten_layer) / (h*w)
            # s_G = tf.matmul(tf.transpose(s_flatten_layer), s_flatten_layer)
            self.s_feats[layer] = tf.get_variable('style_{}'.format(idx), 
                                  initializer=s_G, trainable=False)

    def _get_loss(self):
        with tf.name_scope('loss'):
            alpha, beta = 5e-4, 1
            self.contant_loss = 0
            for idx, layer_name in enumerate(self.content_layers):
                print(layer_name)
                side_cost = alpha * self._layer_content_loss(layer_name)
                self.contant_loss += side_cost
                tf.add_to_collection('losses_new', side_cost)
                tf.summary.scalar("content_loss_{}".format(idx), side_cost, collections=['train'])

            self.style_loss = 0
            for idx, layer_name in enumerate(self.style_layers):
                side_cost = beta * self._layer_style_loss(layer_name)*tf.constant(0.2, tf.float32)
                self.style_loss += side_cost
                tf.add_to_collection('losses_new', side_cost)
                tf.summary.scalar("style_loss_{}".format(idx), side_cost, collections=['train'])

            self.check_loss = tf.add_n(tf.get_collection('losses_new'), name='result') 
            # check = tf.identity(contant_loss, name='contant_loss')
            # check = tf.identity(style_loss, name='style_loss')

            return self.check_loss 

    def _get_optimizer(self):
        # return tf.train.AdamOptimizer(learning_rate=self._lr, beta1=0.9, beta2=0.999, epsilon=1e-08)
        return tf.contrib.opt.ScipyOptimizerInterface(self.total_loss, 
            method='L-BFGS-B', options={'maxiter': 200})

    def _layer_content_loss(self, layer_name):
        """ 
        Args:
            layer_name (str): name of the layer to be used to compute content loss
        # """
        # mse = tf.losses.mean_squared_error(self.c_feats[layer_name], 
        #                                    self.random_im_layer[layer_name], 
        #                                    scope='content_loss_{}'.format(layer_name))
        # print(layer_name)
        N = self.c_feats[layer_name].shape.as_list()[-1]
        h = self.c_feats[layer_name].shape.as_list()[1]
        w = self.c_feats[layer_name].shape.as_list()[2]
        M = h * w
        mse = tf.reduce_sum(tf.pow((self.c_feats[layer_name] - self.random_im_layer[layer_name]), 2))
        return mse

    def _layer_style_loss(self, layer_name):
        s_G = self.s_feats[layer_name] 

        N = self.random_im_layer[layer_name].shape.as_list()[-1]
        h = tf.cast(tf.shape(self.random_im_layer[layer_name])[1], tf.float32)
        w = tf.cast(tf.shape(self.random_im_layer[layer_name])[2], tf.float32)
        M = h * w
        random_flatten_layer = tf.reshape(self.random_im_layer[layer_name], shape=[-1, N])
        random_G = tf.matmul(tf.transpose(random_flatten_layer), random_flatten_layer) / (M)
        # random_G = tf.matmul(tf.transpose(random_flatten_layer), random_flatten_layer)
        mse = tf.reduce_sum(tf.pow((s_G - random_G), 2))

        # mse = tf.losses.mean_squared_error(s_G, random_G, 
        #                                    scope='style_loss_{}'.format(layer_name))

        return mse * (1. / (4 * N ** 2))

    def train_step(self, sess, save_dir):
        global _step
        _step = 0

        def loss_callback(tl, cl, sl, g_im):
            global _step 
            print('[{}] total loss: {}, c loss: {}, s loss: {}'.
                format(_step, tl, cl, sl))
            scipy.misc.imsave('{}test_{}.png'.format(save_dir, _step), np.squeeze(g_im))
            _step += 1

        self.total_loss = self.get_loss()
        opt = self.get_optimizer()
        opt.minimize(sess, fetches=[self.check_loss, self.contant_loss, self.style_loss, self.random_im],
            loss_callback=loss_callback)


    def get_train_op(self):
        
        self.total_loss = self.get_loss()
        opt = self.get_optimizer()
        # return opt
        grads = opt.compute_gradients(self.total_loss)
        return opt.apply_gradients(grads, name='train')

if __name__ == '__main__':


    VGG_PATH = 'E:\\GITHUB\\workspace\\CNN\\pretrained\\vgg19.npy'
    SAVE_DIR = 'E:\\GITHUB\\workspace\\CNN\\result\\'
    STYLE_PATH = 'E:\\GITHUB\\workspace\\CNN\\dataset\\chong.jpg'
    CONTENT_PATH = 'E:\\GITHUB\\workspace\\CNN\\dataset\\test.png'

    VGG_PATH = 'D:\\Qian\\GitHub\\workspace\\VGG\\vgg19.npy'
    STYLE_PATH = 'D:\\Qian\\GitHub\\workspace\\t\\vangohg1.jpg'
    CONTENT_PATH = 'D:\\Qian\\GitHub\\workspace\\VGG\\test_2.png'
    SAVE_DIR = 'D:\\Qian\\GitHub\\workspace\\t\\'

    s_im = load_image(STYLE_PATH, read_channel=3)
    c_im = load_image(CONTENT_PATH, read_channel=3)

    c_h = c_im.shape[1]
    c_w = c_im.shape[2]

    style_trans_model = NerualStyle(pre_train_path=VGG_PATH, im_height=c_h, im_width=c_w, init_im=c_im)
    
    style_trans_model.create_graph()
    style_trans_model.comp_context_style_feats()
    # train_op = style_trans_model.get_train_op()
    # loss_op = style_trans_model.check_loss
    # content_loss_op = style_trans_model.contant_loss
    # style_loss_op = style_trans_model.style_loss

    gen_im = style_trans_model.random_im
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    # session = tf.Session(config=config, ...)

    writer = tf.summary.FileWriter(SAVE_DIR)
    with tf.Session(config=config) as sess:

        initializer = tf.global_variables_initializer()
        sess.run(initializer,
            feed_dict = {style_trans_model.c_im: c_im, style_trans_model.s_im: s_im})

        writer.add_graph(sess.graph)
        style_trans_model.train_step(sess, SAVE_DIR)


        # cnt = 0
        # while cnt < 10000:
        #     _, content_im, loss, cl, sl = sess.run([train_op, gen_im, loss_op, content_loss_op, style_loss_op])
        #     print('total loss: {}, content_loss: {}, style_loss: {}'.format(loss, cl, sl))
        #     if cnt % 100 == 0:

        #         scipy.misc.imsave('{}test_{}.png'.format(SAVE_DIR, cnt), np.squeeze(content_im))
        #     print(cnt)
        #     cnt += 1

    writer.close()



 