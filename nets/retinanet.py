#! /usr/bin/python3
# -*- coding:utf-8 -*-

'''
@Author : Gabriel
@About : RetinaNet网络结构
'''
import math
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np

from tensorflow.keras import layers
from nets.resnet import ResNet50


class UpSample(keras.layers.Layer):
    def call(self, inputs, **kwargs):
        source, target = inputs
        target_shape = keras.backend.shape(target)
        return tf.image.resize(source, (target_shape[1], target_shape[2]),
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, antialias=False)

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0],) + input_shape[1][1:3] + (input_shape[0][-1],)


class PriorProbability(keras.initializers.Initializer):
    def __init__(self, probability=0.01):
        self.probability = probability

    def get_config(self):
        return {'probability': self.probability}

    def __call__(self, shape, dtype=None):
        res = np.ones(shape, dtype=np.float32) * -math.log((1 - self.probability) / self.probability)
        return res


def make_last_layer_loc(num_anchors, pyramid_feature_size=256):
    '''
    获得回归预测结果
    所有特征层共用一个retinaHead
    '''
    inputs = layers.Input(shape=(None, None, pyramid_feature_size))
    options = {
        'kernel_size': 3,
        'strides': 1,
        'padding': 'same',
        'kernel_initializer': keras.initializers.random_normal(mean=0.0, stddev=0.001, seed=None),
        'bias_initializer': 'zeros'
    }
    outputs = inputs
    # 进行4次卷积，通道数为256
    for i in range(4):
        outputs = layers.Conv2D(filters=256, activation='relu', name=f'pyramid_regression_{i}', **options)(outputs)

    # 获得回归预测结果，并reshape
    outputs = layers.Conv2D(num_anchors * 4, name='pyramid_regression', **options)(outputs)
    regression = layers.Reshape((-1, 4), name='pyramid_regression_reshape')(outputs)

    # 构建一个模型
    regression_model = keras.models.Model(inputs=inputs, outputs=regression, name='regression_submodel')

    return regression_model


def make_last_layer_cla(num_classes, num_anchors, pyramid_feature_size=256):
    '''
    获得分类预测结果
    所有特征层共用一个retinaHead
    '''
    inputs = layers.Input(shape=(None, None, pyramid_feature_size))
    options = {
        'kernel_size': 3,
        'strides': 1,
        'padding': 'same',
    }
    outputs = inputs
    # 进行4次卷积，通道数为256
    for i in range(4):
        outputs = layers.Conv2D(filters=256, activation='relu', name=f'pyramid_classification_{i}',
                                kernel_initializer=keras.initializers.random_normal(mean=0., stddev=.01, seed=None),
                                bias_initializer='zeros', **options)(outputs)

    # 获得回归预测结果，并reshape
    outputs = layers.Conv2D(num_anchors * num_classes,
                            kernel_initializer=keras.initializers.random_normal(mean=0., stddev=.01, seed=None),
                            bias_initializer=PriorProbability(probability=0.01), name='pyramid_classification',
                            **options)(outputs)
    outputs = layers.Reshape((-1, num_classes), name='pyramid_classification_reshape')(outputs)

    # 使用sigmoid转换成概率
    classification = layers.Activation('sigmoid', name='pyramid_classification_sigmoid')(outputs)
    # 构建一个模型
    classification_model = keras.models.Model(inputs=inputs, outputs=classification, name='classification_submodel')

    return classification_model


def resnet_retinanet(input_shape, num_classes, num_anchors=9, name='retinanet'):
    inputs = layers.Input(shape=input_shape)
    '''
    取出三个有效特征层，分别是C3、C4、C5
    C3     [75,75,512]
    C4     [38,38,1024]
    C5     [19,19,2048]
    '''
    C3, C4, C5 = ResNet50(inputs)

    # [75,75,512]->[75,75,256]
    P3 = layers.Conv2D(256, kernel_size=1, strides=1, padding='same', name='C3_reduced')(C3)
    # [38,38,1024]->[38,38,1256]
    P4 = layers.Conv2D(256, kernel_size=1, strides=1, padding='same', name='C4_reduced')(C4)
    # [19,19,2048]->[19,19,256]
    P5 = layers.Conv2D(256, kernel_size=1, strides=1, padding='same', name='C5_reduced')(C5)

    # [19,19,256]->[38,38,256]
    P5_upsampled = UpSample(name='P5_upsampled')([P5, P4])
    # [38,38,256]+[38,38,256]->[38,38,256]
    P4 = layers.Add(name='P4_merged')([P5_upsampled, P4])

    # [38,38,256]->[75,75,256]
    P4_upsampled = UpSample(name='P4_upsampled')([P4, P3])
    # [75,75,256]+[75,75,256]->[75,75,256]
    P3 = layers.Add(name='P3_merged')([P4_upsampled, P3])

    # [75,75,256]->[75,75,256]
    P3 = layers.Conv2D(256, kernel_size=3, strides=1, padding='same', name='P3')(P3)
    # [38,38,256]->[38,38,256]
    P4 = layers.Conv2D(256, kernel_size=3, strides=1, padding='same', name='P4')(P4)
    # [19,19,256]->[19,19,256]
    P5 = layers.Conv2D(256, kernel_size=3, strides=1, padding='same', name='P5')(P5)

    # [19,19,2048]->[10,10,246]
    P6 = layers.Conv2D(256, kernel_size=3, strides=2, padding='same', name='P6')(C5)
    P7 = layers.Activation('relu', name='C6_relu')(P6)
    # [10,10,256]->[5,5,256]
    P7 = layers.Conv2D(256, kernel_size=3, strides=2, padding='same', name='P7')(P7)

    features = [P3,P4,P5,P6,P7]

    regression_model = make_last_layer_loc(num_anchors)
    classification_model = make_last_layer_cla(num_classes,num_anchors)

    regressions = []
    classifications = []

    '''
    将获取到的P3~P7传入到RetinaHead里面进行预测，
    获得回归预测结果和分类预测结果，
    将所有特征层的预测结果进行堆叠
    '''
    for feature in features:
        regression = regression_model(feature)
        classification = classification_model(feature)

        regressions.append(regression)
        classifications.append(classification)

    regressions = layers.Concatenate(axis=1,name='regression')(regressions)
    classifications = layers.Concatenate(axis=1,name='classification')(classifications)

    model = keras.models.Model(inputs,[regressions,classifications],name=name)

    return model


if __name__ == '__main__':
    model = resnet_retinanet(input_shape=(600,600,3),num_classes=20)
    model.summary()
