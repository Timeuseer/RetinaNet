#! /usr/bin/python3
# -*- coding:utf-8 -*-

'''
@Author : Gabriel
@About : 数据预处理
'''

import math
import cv2
import numpy as np
from tensorflow import keras
from random import shuffle
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from PIL import Image
from utils.utils import cvtColor


class RetinanetDatasets(keras.utils.Sequence):
    def __init__(self, annotation_lines, input_shape, anchors, batch_size, num_classes, train, ignore_threshold=0.4,
                 overlap_threshold=0.5):
        self.annotation_lines = annotation_lines
        self.length = len(self.annotation_lines)
        self.input_shape = input_shape
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.train = train
        self.ignore_threshold = ignore_threshold
        self.overlap_threshold = overlap_threshold

    def __len__(self):
        return math.ceil(len(self.annotation_lines) / float(self.batch_size))

    def __getitem__(self, index):
        img_data = []
        regressions = []
        classifications = []
        for i in range(index * self.batch_size, (index + 1) * self.batch_size):
            i = i % self.length
            '''
            训练时进行数据增强，验证时不进行
            '''
            img, box = self.get_random_data(self.annotation_lines[i], self.input_shape, random=self.train)
            if len(box) != 0:
                boxes = np.array(box[:, :4], dtype=np.float32)
                boxes[:, [0, 2]] = boxes[:, [0, 2]] / self.input_shape[1]
                boxes[:, [1, 3]] = boxes[:, [1, 3]] / self.input_shape[0]
                one_hot_label = np.eye(self.num_classes)[np.array(box[:, 4], np.int32)]
                box = np.concatenate([boxes, one_hot_label], axis=-1)
            assignment = self.assign_boxes(box)
            regression = assignment[:, :5]
            classification = assignment[:, 5:]

            img_data.append(preprocess_input(np.array(img, np.float32)))
            regressions.append(regression)
            classifications.append(classification)

        return np.array(img_data), [np.array(regressions, dtype=np.float32),
                                    np.array(classifications, dtype=np.float32)]

    def on_epoch_end(self):
        shuffle(self.annotation_lines)

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, annotation_line, input_shape, jitter=0.3, hue=0.1, sat=0.7, val=0.4, random=True):
        line = annotation_line.split()
        '''
        转换图像为RGB
        '''
        img = Image.open(line[0])
        img = cvtColor(img)

        iw, ih = img.size
        h, w = input_shape

        '''
        获得预测框
        '''
        box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

        if not random:
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)
            dx = (w - nw) // 2
            dy = (h - nh) // 2

            '''
            将图像多于的部分加上灰条
            '''
            img = img.resize((nw, nh), Image.BICUBIC)
            new_img = Image.new('RGB', (w, h), (128, 128, 128))
            new_img.paste(img, (dx, dy))
            img_data = np.array(new_img, np.float32)

            '''
            对真实框进行调整
            '''
            if len(box) > 0:
                np.random.shuffle(box)
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w > 1, box_h > 1)]

            return img_data, box

        '''
        对图像进行缩放，并进行长和宽的扭曲
        '''
        new_ar = iw / ih * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(0.25, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw * new_ar)
        img = img.resize((nw, nh), Image.BICUBIC)

        '''
        将图像多余部分加上灰条
        '''
        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_img = Image.new('RGB', (w, h), (128, 128, 128))
        new_img.paste(img, (dx, dy))
        img = new_img

        '''
        翻转图像
        '''
        flip = self.rand() < .5
        if flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        img_data = np.array(img, np.uint8)

        '''
        对图像进行色域变换
        '''
        # 计算色域变换的参数
        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        # 将图像转换到HSV
        hue, sat, val = cv2.split(cv2.cvtColor(img_data, cv2.COLOR_RGB2HSV))
        dtype = img_data.dtype
        # 应用变换
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        img_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        img_data = cv2.cvtColor(img_data, cv2.COLOR_HSV2RGB)

        '''
        对真实框进行调整
        '''
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            if flip: box[:, [0, 2]] = w - box[:, [2, 0]]
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]

        return img_data, box

    def iou(self, box):
        '''
        计算出每个真实框与所有的先验框的IOU
        判断真实框与先验框的重合程度
        '''
        inter_upleft = np.maximum(self.anchors[:, :2], box[:2])
        inter_bottomleft = np.minimum(self.anchors[:, 2:4], box[2:])

        inter_wh = inter_bottomleft - inter_upleft
        inter_wh = np.maximum(inter_wh, 0)
        inter = inter_wh[:, 0] * inter_wh[:, 1]

        '''
        真实框的面积
        '''
        area_true = (box[2] - box[0]) * (box[3] - box[1])

        '''
        先验框的面积
        '''
        area_gt = (self.anchors[:, 2] - self.anchors[:, 0]) * (self.anchors[:, 3] - self.anchors[:, 1])

        '''
        计算IOU
        '''
        union = area_true + area_gt - inter

        iou = inter / union

        return iou

    def encode_box(self, box, return_iou=True, variances=0.2):
        '''
        计算当前真实框和先验框的重合情况
        iou         [self.num_anchors]
        encode_box  [self.num_anchors,5]
        '''
        iou = self.iou(box)
        ignored_box = np.zeros((self.num_anchors, 1))

        '''
        找到处于忽略门限值范围内的先验框
        '''
        assign_mask_ignore = (iou > self.ignore_threshold) & (iou < self.overlap_threshold)
        ignored_box[:, 0][assign_mask_ignore] = iou[assign_mask_ignore]

        encoded_box = np.zeros((self.num_anchors, 4 + return_iou))
        '''
        找到每一个真实框，重合程度较高的先验框
        '''
        assign_mask = iou > self.overlap_threshold

        '''
        如果没有满足上述要求的，则选择重合度最大的为正样本
        '''
        if not assign_mask.any():
            assign_mask[iou.argmax()] = True

        '''
        利用iou进行赋值
        '''
        if return_iou:
            encoded_box[:, -1][assign_mask] = iou[assign_mask]

        '''
        找到对应的先验框
        '''
        assign_anchors = self.anchors[assign_mask]

        '''
        逆向编码，将真实框转化为retinanet预测结果的格式
        '''
        # 先计算真实框的中心和长宽
        assign_anchors_w = (assign_anchors[:, 2] - assign_anchors[:, 0])
        assign_anchors_h = (assign_anchors[:, 3] - assign_anchors[:, 1])

        '''
        逆向求取retinanet该有的预测结果
        '''
        # 先求取中心的预测结果，再求取宽高的预测结果
        encoded_box[:, 0][assign_mask] = (box[0] - assign_anchors[:, 0]) / assign_anchors_w / variances
        encoded_box[:, 1][assign_mask] = (box[1] - assign_anchors[:, 1]) / assign_anchors_h / variances
        encoded_box[:, 2][assign_mask] = (box[2] - assign_anchors[:, 2]) / assign_anchors_w / variances
        encoded_box[:, 3][assign_mask] = (box[3] - assign_anchors[:, 3]) / assign_anchors_h / variances

        return encoded_box.ravel(), ignored_box.ravel()

    def assign_boxes(self, boxes):
        '''
        assignment分为三个部分:
        :4      的内容为网络应该有的回归预测结果
        4:-1    的内容为先验框所对应的种类，默认为背景
        -1      的内容为先验框中是否包含物体
        '''
        assignment = np.zeros((self.num_anchors, 4 + 1 + self.num_classes + 1))
        assignment[:, 4] = 1.0
        assignment[:, -1] = 0.0
        if len(boxes) == 0:
            return assignment

        '''
        对每一个真实框都计算IOU
        '''
        apply_along_axis_boxes = np.apply_along_axis(self.encode_box, 1, boxes[:, :4])
        encoded_boxes = np.array([apply_along_axis_boxes[i, 0] for i in range(len(apply_along_axis_boxes))])
        ignored_boxes = np.array([apply_along_axis_boxes[i, 1] for i in range(len(apply_along_axis_boxes))])

        '''
        在reshape之后，活动ignored_boxes的shape:
            [num_true_box,num_anchors,1] 其中1位Iou
        '''
        ignored_boxes = ignored_boxes.reshape(-1, self.num_anchors, 1)
        ignored_iou = ignored_boxes[:, :, 0].max(axis=0)
        ignored_iou_mask = ignored_iou > 0

        assignment[:, 4][ignored_iou_mask] = -1
        assignment[:, -1][ignored_iou_mask] = -1

        '''
        在reshape之后，活动encoded_boxes的shape:
            [num_true_box,num_anchors,4+1] 其中1位Iou
        '''
        encoded_boxes = encoded_boxes.reshape(-1, self.num_anchors, 5)

        '''
        求取每一个先验框重合度最大的真实框
        '''
        best_iou = encoded_boxes[:, :, -1].max(axis=0)
        best_iou_idx = encoded_boxes[:, :, -1].argmax(axis=0)
        best_iou_mask = best_iou > 0
        best_iou_idx = best_iou_idx[best_iou_mask]

        '''
        计算总共有多少个先验框满足要求
        '''
        assign_num = len(best_iou_idx)

        # 取出编码后的真实框
        encoded_boxes = encoded_boxes[:, best_iou_mask, :]
        assignment[:, :4][best_iou_mask] = encoded_boxes[best_iou_idx, np.arange(assign_num), :4]

        '''
        4代表为背景的概率，设定为0，表示先验框含有物体        
        '''
        assignment[:, 4][best_iou_mask] = 1
        assignment[:, 5:-1][best_iou_mask] = boxes[best_iou_idx, 4:]

        '''
        -1表示先验框是否有对应物体
        '''
        assignment[:, -1][best_iou_mask] = 1

        return assignment
