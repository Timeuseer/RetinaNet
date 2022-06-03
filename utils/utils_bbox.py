#! /usr/bin/python3
# -*- coding:utf-8 -*-

'''
@Author : Gabriel
@About :
'''
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K


class BBoxUtility:
    def __init__(self, num_classes, nms_thresh=0.45, top_k=300):
        self.num_classes = num_classes
        self._nems_thresh = nms_thresh
        self._top_k = top_k
        self.boxes = K.placeholder(dtype='float32', shape=(None, 4))
        self.scores = K.placeholder(dtype='float32', shape=(None,))

    def bbox_iou(self, b1, b2):
        b1_x1, b1_y1, b1_x2, b1_y2 = b1[0], b1[1], b1[2], b1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = b2[:, 0], b2[:, 1], b2[:, 2], b2[:3]

        inter_rect_x1 = np.maximum(b1_x1, b2_x1)
        inter_rect_y1 = np.maximum(b1_y1, b2_y1)
        inter_rect_x2 = np.minimum(b1_x2, b2_x2)
        inter_rect_y2 = np.minimum(b1_y2, b2_y2)

        inter_area = np.maximum(inter_rect_x2 - inter_rect_x1, 0) * np.maximum(inter_rect_y2 - inter_rect_y1, 0)

        area_b1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        area_b2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

        iou = inter_area / np.maximum((area_b1 + area_b2 - inter_area), 1e-6)

        return iou

    def efficientdet_correct_boxes(self, box_xy, box_wh, input_shape, img_shape, letterbox_img):
        '''
        将y轴放前面方便预测框和图像的宽高进行相乘
        '''
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        input_shape = np.array(input_shape)
        img_shape = np.array(img_shape)

        if letterbox_img:
            '''
            这里求出来的offset是图像有效区域相对于图像左上角的偏移情况
            new_shape指的是宽高缩放情况
            '''
            new_shape = np.round(img_shape * np.min(input_shape / img_shape))
            offset = (input_shape - new_shape) / 2. / input_shape
            scale = input_shape / new_shape

            box_yx = (box_yx - offset) * scale
            box_hw *= scale

        box_mines = box_yx - (box_hw / 2.)
        box_maxes = box_yx + (box_hw / 2.)
        boxes = np.concatenate([box_mines[..., 0:1], box_mines[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]],
                               axis=-1)
        boxes *= np.concatenate([img_shape, img_shape], axis=-1)

        return boxes

    def decode_boxes(self, mbox_loc, anchors, variance=0.2):
        anchor_w = anchors[:, 2] - anchors[:, 0]
        anchor_h = anchors[:, 3] - anchors[:, 1]

        # 获取真实框的左上角和右下角
        decode_bbox_xmin = mbox_loc[:, 0] * anchor_w * variance + anchors[:, 0]
        decode_bbox_ymin = mbox_loc[:, 1] * anchor_h * variance + anchors[:, 1]
        decode_bbox_xmax = mbox_loc[:, 2] * anchor_w * variance + anchors[:, 2]
        decode_bbox_ymax = mbox_loc[:, 3] * anchor_h * variance + anchors[:, 3]

        # 进行堆叠
        decode_bbox = np.concatenate((decode_bbox_xmin[:, None],
                                      decode_bbox_ymin[:, None],
                                      decode_bbox_xmax[:, None],
                                      decode_bbox_ymax[:, None]
                                      ), axis=-1)

        # 防止超出[0,1]
        decode_bbox = np.minimum(np.maximum(decode_bbox, 0.), 1.)

        return decode_bbox

    def decode_box(self, predictions, anchors, img_shape, input_shape, letterbox_img, confidence=0.5):
        '''
        获取回归预测结果
        '''
        mbox_loc = predictions[0]

        '''
        获取种类的置信度
        '''
        mbox_conf = predictions[1]

        results = [None for _ in range(len((mbox_loc)))]

        '''
        对每一张图片进行处理
        '''
        for i in range(len(mbox_loc)):
            results.append([])
            '''
            利用回归结果对先验框进行解码
            '''
            decode_bbox = self.decode_boxes(mbox_loc[i], anchors)

            class_conf = np.expand_dims(np.max(mbox_conf[i], 1), -1)
            class_pred = np.expand_dims(np.argmax(mbox_conf[i], 1), -1)

            '''
            判断置信度是否大于阈值
            '''
            conf_mask = (class_conf >= confidence)[:, 0]

            '''
            将预测结果进行堆叠
            '''
            detections = np.concatenate((decode_bbox[conf_mask], class_conf[conf_mask], class_pred[conf_mask]), 1)
            unique_labels = np.unique(detections[:, -1])

            '''
            对种类进行循环
            '''
            for c in unique_labels:
                '''
                获取该类得分筛选后全部的预测结果
                '''
                detections_class = detections[detections[:, -1] == c]
                '''
                进行非极大抑制
                '''
                idx = tf.image.non_max_suppression(detections_class[:, :4], detections_class[:, 4], self._top_k,
                                                   iou_threshold=self._nms_thresh)

                max_detections = detections_class[idx]

                results[i] = max_detections if results[i] is None else np.concatenate((results[i], max_detections),
                                                                                      axis=0)

            '''
            从有灰条的结果转化为没有灰条
            '''
            if results[i] is not None:
                results[i] = np.array(results[i])
                box_xy, box_wh = (results[i][:, 0:2] + results[i][:, 2:4]) / 2, (
                        results[i][:, 2:4] - results[i][:, 0:2])
                results[i][:, :4] = self.efficientdet_correct_boxes(box_xy, box_wh, input_shape, img_shape,
                                                                    letterbox_img)

        return results
