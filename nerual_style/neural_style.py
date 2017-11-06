# File: neural_style.py
# Author: Qian Ge <geqian1001@gmail.com>

import os
import numpy as np
import scipy.misc
import tensorflow as tf

from VGG import VGG19_FCN


class NerualStyle(object):
    """ base of VGG class """
    def __init__(self,
                 im_height=224, im_width=224,
                 num_channels=3,
                 pre_train_path=None,
                 content_weight=5e-4,
                 style_weight=0.2,
                 variation_weight=0.05,
                 max_iter=500
                 ):
        """
        Args:
            num_channels (int): number of content image channels
            im_height, im_width (int): size of output content image
            pre_train_path (str): path of pre-trained vgg19 parameters
        """
        self._nchannel = num_channels
        self._h = im_height
        self._w = im_width

        self._c_w = content_weight
        self._s_w = style_weight
        self._v_w = variation_weight

        self._max_iter = max_iter

        self.layer = {}

        # if pre_train_path is None:
        #     raise ValueError('pre_train_path can not be None!')
        if not os.path.isdir(pre_train_path):
            pre_train_path = None
        self._pre_train_path = pre_train_path

        self.content_layers = ['conv4_2']
        self.style_layers = ['conv1_1', 'conv2_1', 'conv3_1',
                             'conv4_1', 'conv5_1']

    def create_graph(self):
        self._create_input()
        self._create_model()
        self._comp_context_style_feats()

    def _create_input(self):
        # content image
        self.c_im = tf.placeholder(
            tf.float32, name='content_image',
            shape=[1, self._h, self._w, self._nchannel])
        # style image
        self.s_im = tf.placeholder(tf.float32, name='style_image',
                                   shape=[1, None, None, self._nchannel])

    def _create_model(self):

        c_im = self.c_im
        s_im = self.s_im

        # init vgg19 model
        if self._pre_train_path is None:
            is_load = False
        else:
            is_load = True
        vgg_model = VGG19_FCN(is_load=is_load, trainable=False,
                              pre_train_path=self._pre_train_path)

        # vgg for content image
        vgg_model.create_conv([c_im, 1])
        self.c_layer = {}
        for layer in self.content_layers:
            self.c_layer[layer] = vgg_model.layer[layer]

        # vgg for style image
        with tf.variable_scope(tf.get_variable_scope()) as scope:
            scope.reuse_variables()
            vgg_model.create_conv([s_im, 1])
        self.s_layer = {}
        for layer in self.style_layers:
            self.s_layer[layer] = vgg_model.layer[layer]

        # vgg for mix image
        with tf.variable_scope('mix_im'):
            self.mix_im = tf.get_variable('mix_im',
                                          initializer=self.c_im,
                                          trainable=True)
        with tf.variable_scope(tf.get_variable_scope()) as scope:
            scope.reuse_variables()
            vgg_model.create_conv([self.mix_im, 1])

        self.mix_layer = {}
        for layer in self.content_layers + self.style_layers:
            self.mix_layer[layer] = vgg_model.layer[layer]

    def _comp_context_style_feats(self):
        self.c_feats = {}
        self.s_feats = {}
        # content feature
        for idx, layer in enumerate(self.content_layers):
            self.c_feats[layer] = tf.get_variable(
                'content_{}'.format(idx),
                initializer=self.c_layer[layer], trainable=False)
        # style feature
        for idx, layer in enumerate(self.style_layers):
            with tf.variable_scope('style_{}'.format(idx)):
                gram_matrix = self._gram_matrix(self.s_layer[layer])
                self.s_feats[layer] = tf.get_variable(
                    'style_feats_{}'.format(idx),
                    initializer=gram_matrix, trainable=False)

    def _get_optimizer(self):
        return tf.contrib.opt.ScipyOptimizerInterface(
            self.total_loss, method='L-BFGS-B', options={'maxiter': self._max_iter})

    def _get_loss(self):
        with tf.name_scope('loss'):
            # content loss
            self.contant_loss = 0
            for idx, layer in enumerate(self.content_layers):
                side_cost = self._c_w * self._layer_content_loss(
                    layer, 'c_loss_{}'.format(idx))
                self.contant_loss += side_cost
                tf.add_to_collection('losses_new', side_cost)
            # style loss
            self.style_loss = 0
            for idx, layer in enumerate(self.style_layers):
                side_cost = self._s_w * self._layer_style_loss(
                    layer, 's_loss_{}'.format(idx))
                self.style_loss += side_cost
                tf.add_to_collection('losses_new', side_cost)
            # total variation loss
            self.tv_loss = self._v_w * self._total_variation(self.mix_im)
            tf.add_to_collection('losses_new', self.tv_loss)
            # total loss
            self.total_loss = tf.add_n(tf.get_collection('losses_new'),
                                       name='result')

            return self.total_loss

    def _total_variation(self, image):
        # h = tf.cast(tf.shape(image)[1], tf.float32)
        # w = tf.cast(tf.shape(image)[2], tf.float32)
        var_x = tf.pow(image[:, 1:, :-1, :] - image[:, :-1, :-1, :], 2)
        var_y = tf.pow(image[:, :-1, 1:, :] - image[:, :-1, :-1, :], 2)
        return tf.reduce_sum(var_x + var_y)

    def _layer_content_loss(self, layer, name='content_loss'):
        """
        Args:
            layer (str): name of the layer to be used to compute content loss
        """
        with tf.name_scope(name):
            # N = self.c_feats[layer].shape.as_list()[-1]
            h = self.c_feats[layer].shape.as_list()[1]
            w = self.c_feats[layer].shape.as_list()[2]
            M = h * w
            pow_mat = tf.pow((self.c_feats[layer] - self.mix_layer[layer]), 2)
            mse = tf.reduce_sum(pow_mat)
            return mse / M

    def _layer_style_loss(self, layer, name='style_loss'):
        with tf.name_scope(name):
            N = self.mix_layer[layer].shape.as_list()[-1]
            s_G = self.s_feats[layer]
            random_G = self._gram_matrix(self.mix_layer[layer])
            mse = tf.reduce_sum(tf.pow((s_G - random_G), 2))
            return mse * (1. / (4 * N ** 2))

    def _gram_matrix(self, layer, name='gram_matrix'):
        with tf.name_scope(name):
            h = tf.cast(tf.shape(layer)[1], tf.float32)
            w = tf.cast(tf.shape(layer)[2], tf.float32)
            n_filter = layer.shape.as_list()[-1]

            flatten_layer = tf.reshape(layer, shape=[-1, n_filter])
            g_mat = tf.matmul(tf.transpose(flatten_layer), flatten_layer)
            return g_mat / (h * w)

    def train_step(self, sess, save_dir):
        global _step
        _step = 0

        def loss_callback(tl, cl, sl, tvl, g_im):
            global _step
            if _step % 20 == 0:
                print('[{}] total loss: {}, content loss: {},\
                      style loss: {}, total variation loss: {}'.
                      format(_step, tl, cl, sl, tvl))
                g_im = np.clip(g_im, 0, 255).astype(np.uint8)
                scipy.misc.imsave('{}test_{}.png'.format(save_dir, _step),
                                  np.squeeze(g_im))
            _step += 1

        self._get_loss()
        opt = self._get_optimizer()
        opt.minimize(sess, fetches=[self.total_loss, self.contant_loss,
                                    self.style_loss, self.tv_loss,
                                    self.mix_im],
                     loss_callback=loss_callback)
