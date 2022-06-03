#! /usr/bin/python3
# -*- coding:utf-8 -*-

'''
@Author : Gabriel
@About : 生成anchor
'''
import numpy as np
from tensorflow import keras


class AnchorBox:
    def __init__(self, ratios, scales):
        self.ratios = ratios
        self.scales = scales
        self.num_anchors = len(ratios) * len(scales)

    def generate_anchors(self, base_size=16):
        anchors = np.zeros((self.num_anchors, 4))
        anchors[:, 2:] = base_size * np.tile(self.scales, (2, len(self.scales))).T

        # 计算先验框的面积
        areas = anchors[:, 2] * anchors[:, 3]

        anchors[:, 2] = np.sqrt(areas / np.repeat(self.ratios, len(self.scales)))
        anchors[:, 3] = np.sqrt(areas * np.repeat(self.ratios, len(self.scales)))

        anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
        anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

        return anchors

    def shift(self, shape, stride, anchors):
        # 生成特征层的网格中心
        shift_x = (np.arange(0, shape[0], dtype=keras.backend.floatx()) + 0.5) * stride
        shift_y = (np.arange(0, shape[1], dtype=keras.backend.floatx()) + 0.5) * stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)

        shift_x = np.reshape(shift_x, [-1])
        shift_y = np.reshape(shift_y, [-1])

        # 将网格中心进行堆叠
        shifts = np.stack([
            shift_x,
            shift_y,
            shift_x,
            shift_y
        ], axis=0)

        shifts = np.transpose(shifts)
        num_anchors = np.shape(anchors)[0]

        k = np.shape(shifts)[0]
        shift_anchors = np.reshape(anchors, [1, num_anchors, 4]) + np.array(np.reshape(shifts, [k, 1, 4]))
        shift_anchors = np.reshape(shift_anchors, [k * num_anchors, 4])

        return shift_anchors


def get_img_output_length(height, width):
    '''
    计算共享特征层的大小
    '''
    filter_sizes = [7, 3, 3, 3, 3, 3, 3]
    padding = [3, 1, 1, 1, 1, 1, 1]
    stride = [2, 2, 2, 2, 2, 2, 2]
    feature_heights = []
    feature_widths = []
    for i in range(len(filter_sizes)):
        height = (height + 2 * padding[i] - filter_sizes[i]) // stride[i] + 1
        width = (width + 2 * padding[i] - filter_sizes[i]) // stride[i] + 1
        feature_heights.append(height)
        feature_widths.append(width)
    return np.array(feature_heights)[-5:], np.array(feature_widths)[-5:]


def get_anchors(input_shape, anchors_size=[32, 64, 128, 256, 512], strides=[8, 16, 32, 64, 128], \
                ratios=[0.5, 1, 2], scales=[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]):
    feature_h, feature_w = get_img_output_length(input_shape[0], input_shape[1])

    all_anchors = []
    anchors_box = AnchorBox(ratios, scales)
    for i in range(len(anchors_size)):
        # 生成每个点的9个先验框
        anchors = anchors_box.generate_anchors(anchors_size[i])
        shift_anchors = anchors_box.shift([feature_h[i], feature_w[i]], strides[i], anchors)
        all_anchors.append(shift_anchors)

    # 将每个特征层的先验框进行堆叠
    all_anchors = np.concatenate(all_anchors, axis=0)
    all_anchors = all_anchors / np.array([input_shape[1], input_shape[0], input_shape[1], input_shape[0]])
    all_anchors = all_anchors.clip(0, 1)

    return all_anchors

get_anchors(input_shape=(600,600))
