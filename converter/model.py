""" Dlib Resnet

This model replicates the structure of Davis King's (Dlib) Face Recognition network.

Reference -
    https://github.com/davisking/dlib/blob/master/examples/dnn_face_recognition_ex.cpp

Notes -:
    * Number of layers seen after analyzing the model:
      {'relu': 29, 'affine_con': 29, 'con': 29, 'add_prev': 14, 'avg_pool': 5, 'loss_metric': 1, 'fc_no_bias': 1, 'max_pool': 1, 'input_rgb_image': 1})      

    * Affine_con is essentially a replacement of BatchNormalization layer for inference mode. 
"""

import tensorflow as tf

from tensorflow.keras import layers as KL
from tensorflow.keras import models as KM
from tensorflow.keras import backend as K

class ScaleLayer(KL.Layer):
    def __init__(self, **kwargs):
        super(ScaleLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2

        if K.image_data_format() == 'channels_last':
            ndim = int(input_shape[-1])
        else:
            ndim = int(input_shape[1])

        self.gamma = self.add_weight(name='gamma', shape=(ndim, ))
        self.beta = self.add_weight(name='beta', shape=(ndim, ))

        super(ScaleLayer, self).build(input_shape)

    def call(self, x):
        input_shape = K.int_shape(x)

        bn_axis = 3 if K.image_data_format() == 'channels_last' else 1

        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[bn_axis] = input_shape[bn_axis]

        broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
        broadcast_beta = K.reshape(self.beta, broadcast_shape)

        output = tf.math.multiply(x, broadcast_gamma)
        output = tf.math.add(output, broadcast_beta)
        return output

    def compute_output_shape(self, input_shape):
        return input_shape


class ReshapeLayer(KL.Layer):
    def __init__(self, **kwargs):
        super(ReshapeLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        super(ReshapeLayer, self).build(input_shape)

    def call(self, x):
        s = K.shape(x)
        zeros_w = tf.zeros((s[0], 1, s[2], s[3]), tf.float32)
        r = K.concatenate([x, zeros_w], 1)

        s = K.shape(r)
        zeros_h = tf.zeros((s[0], s[1], 1, s[3]), tf.float32)
        r = K.concatenate([r, zeros_h], 2)
        return r    

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        if K.image_data_format() == 'channels_last':
            shape[1] = shape[1] + 1
            shape[2] = shape[2] + 1
        else:
            shape[2] = shape[2] + 1
            shape[3] = shape[3] + 1
        return tf.TensorShape(shape)


def pad_depth(x, desired_channels):
    y = K.zeros_like(x)
    new_channels = desired_channels - x.shape.as_list()[-1]
    y = y[:, :, :new_channels]
    return K.concatenate([x, y])

def _convLayer(x,
               num_filters,
               filters,
               strides,
               conv_layer_counter,
               use_bn,
               with_relu,
               padding='same'):
    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
    conv_layer_counter = conv_layer_counter + 1
    x = KL.Conv2D(
        num_filters,
        filters,
        strides,
        padding,
        name='conv_' + str(conv_layer_counter))(x)

    if use_bn:
        x = KL.BatchNormalization(
            axis=bn_axis, name='bn_' + str(conv_layer_counter))(x)
    else:
        x = ScaleLayer(name='sc_' + str(conv_layer_counter))(x)

    if with_relu is True:
        x = KL.Activation('relu')(x)
    return x, conv_layer_counter


def _conv(x, num_filters, filters, conv_layer_counter, use_bn):
    return _convLayer(x, num_filters, filters, (1, 1), conv_layer_counter,
                      use_bn, True)


def _convNoRelu(x, num_filters, filters, conv_layer_counter, use_bn):
    return _convLayer(x, num_filters, filters, (1, 1), conv_layer_counter,
                      use_bn, False)


def _convDown(x, num_filters, filters, conv_layer_counter, use_bn):
    return _convLayer(
        x,
        num_filters,
        filters, (2, 2),
        conv_layer_counter,
        use_bn,
        True,
        padding='valid')


def _residual(x, num_filters, filters, conv_layer_counter, use_bn):
    c1, conv_layer_counter = _conv(x, num_filters, filters, conv_layer_counter,
                                   use_bn)
    c1, conv_layer_counter = _convNoRelu(c1, num_filters, filters,
                                         conv_layer_counter, use_bn)
    x = KL.Add()([c1, x])
    x = KL.Activation('relu')(x)
    return x, conv_layer_counter


def _residualDown(x, num_filters, filters, stage_num, conv_layer_counter,
                  use_bn):

    c1, conv_layer_counter = _convDown(x, num_filters, filters,
                                       conv_layer_counter, use_bn)
    c1, conv_layer_counter = _convNoRelu(c1, num_filters, filters,
                                         conv_layer_counter, use_bn)

    pooled = KL.AveragePooling2D(
        pool_size=(2, 2), strides=(2, 2), padding='valid')(x)

    if K.image_data_format() == 'channels_last':
        shouldPad = not pooled.shape[3] == c1.shape[3]
        shouldAdjustShape = pooled.shape[1] != c1.shape[1] or pooled.shape[
            2] != c1.shape[2]
    else:
        shouldPad = not pooled.shape[1] == c1.shape[1]
        shouldAdjustShape = pooled.shape[2] != c1.shape[2] or pooled.shape[
            3] != c1.shape[3]

    if shouldAdjustShape:
        c1 = ReshapeLayer(name='reshape_' + str(stage_num))(c1)

    if shouldPad:
        if K.image_data_format() == 'channels_last':
            desired_channels = c1.shape.as_list()[-1]
        else:
            desired_channels = c1.shape.as_list()[1]
        arguments = {'desired_channels': desired_channels}
        pooled = KL.Lambda(
            pad_depth, arguments=arguments,
            name='pad_' + str(stage_num))(pooled)

    c1 = KL.Add()([pooled, c1])
    c1 = KL.Activation('relu')(c1)
    return c1, conv_layer_counter


def build_dlib_model(image_h=150, image_w=150, use_bn=False):

    if K.image_data_format() == 'channels_last':
        batch_input_shape = (1, image_h, image_w, 3)
        input_shape = (image_h, image_w, 3)
    else:
        batch_input_shape = (1, 3, image_h, image_w)
        input_shape = (3, image_h, image_w)

    img_input = KL.Input(shape=input_shape, name='input_image')

    conv_layer_counter = 0

    # Head of the network
    x, conv_layer_counter = _convDown(img_input, 32, (7, 7),
                                      conv_layer_counter, use_bn)
    x = KL.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x)

    # Body of the network
    x, conv_layer_counter = _residual(x, 32, (3, 3), conv_layer_counter,
                                      use_bn)
    x, conv_layer_counter = _residual(x, 32, (3, 3), conv_layer_counter,
                                      use_bn)
    x, conv_layer_counter = _residual(x, 32, (3, 3), conv_layer_counter,
                                      use_bn)

    x, conv_layer_counter = _residualDown(x, 64, (3, 3), 1, conv_layer_counter,
                                          use_bn)
    x, conv_layer_counter = _residual(x, 64, (3, 3), conv_layer_counter,
                                      use_bn)
    x, conv_layer_counter = _residual(x, 64, (3, 3), conv_layer_counter,
                                      use_bn)
    x, conv_layer_counter = _residual(x, 64, (3, 3), conv_layer_counter,
                                      use_bn)

    x, conv_layer_counter = _residualDown(x, 128, (3, 3), 2,
                                          conv_layer_counter, use_bn)
    x, conv_layer_counter = _residual(x, 128, (3, 3), conv_layer_counter,
                                      use_bn)
    x, conv_layer_counter = _residual(x, 128, (3, 3), conv_layer_counter,
                                      use_bn)

    x, conv_layer_counter = _residualDown(x, 256, (3, 3), 3,
                                          conv_layer_counter, use_bn)
    x, conv_layer_counter = _residual(x, 256, (3, 3), conv_layer_counter,
                                      use_bn)
    x, conv_layer_counter = _residual(x, 256, (3, 3), conv_layer_counter,
                                      use_bn)
    x, conv_layer_counter = _residualDown(x, 256, (3, 3), 4,
                                          conv_layer_counter, use_bn)

    x = KL.GlobalAveragePooling2D()(x)

    embedding = KL.Dense(128, name="embedding_layer", use_bias=False)(x)    

    return KM.Model(inputs=img_input, outputs=embedding)
