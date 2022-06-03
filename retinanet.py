#! /usr/bin/python3
# -*- coding:utf-8 -*-

'''
@Author : Gabriel
@About : 
'''
import colorsys
import os
import time
import numpy as np
import yaml

from tensorflow.keras.applications.imagenet_utils import preprocess_input
from PIL import ImageDraw, ImageFont
from nets.retinanet import resnet_retinanet
from utils.utils_bbox import BBoxUtility
from utils.utils import get_classes, resize_img, cvtColor
from utils.anchors import get_anchors


class RetinaNet:
    _defaluts = yaml.load(open('config/retinanet.yaml', 'r', encoding='utf-8'))

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaluts:
            return cls._defaluts[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaluts)
        for name, value in kwargs.items():
            setattr(self, name, value)

        '''
        计算总的类别数量
        '''
        self.class_names, self.num_classes = get_classes(self.classes_path)
        self.anchors = get_anchors(self.input_shape, self.anchors_size)
        self.num_classes = self.num_classes

        '''
        设置框的颜色
        '''
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

        self.bbox_util = BBoxUtility(self.num_classes, nms_thresh=self.nms_iou)
        self.generate()

    def generate(self):
        '''
        载入模型
        '''
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        self.retinanet = resnet_retinanet([self.input_shape[0], self.input_shape[1], 3], self.num_classes)
        self.retinanet.load_weights(self.model_path, by_name=True)
        print(f'{model_path} model, anchors, and classes loaded.')

    def detect_img(self, img, crop=False):
        '''
        检测图片
        '''
        img_shape = np.array(np.shape(img))[0:2]

        '''
        将图片转化为RGB
        '''
        img = cvtColor(img)

        '''
        对图片进行resize
        '''
        img_data = resize_img(img, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)

        '''
        图片预处理
        '''
        img_data = preprocess_input(np.expand_dims(np.array(img_data, dtype=np.float32), 0))
        preds = self.retinanet.predict(img_data)

        '''
        将预测结果进行解码
        '''
        results = self.bbox_util.decode_box(preds, self.anchors, img_shape, self.input_shape, self.letterbox_image,
                                            self.confidence)

        '''
        如果没有检测到物体，则返回原图
        '''
        if results[0] is None:
            return img

        top_label = np.array(results[0][:, 5], dtype=np.int32)
        top_conf = results[0][:, 4]
        top_boxes = results[0][:, :4]

        '''
        设置字体和边框厚度
        '''
        font = ImageFont.truetype(font='model_data/simhei.ttf',
                                  size=np.floor(3e-2 * np.shape(img)[1] + 0.5).astype('int32'))
        thickness = max((np.shape(img)[0] + np.shape(img)[1]) // self.input_shape[0], 1)

        '''
        是否进行裁剪
        '''
        if crop:
            for i, c in list(enumerate(top_label)):
                top, left, bottom, right = top_boxes[i]
                top = max(0, np.floor(top).astype('int32'))
                left = max(0, np.floor(left).astype('int32'))
                bottom = min(img.size[1], np.floor(bottom).astype('int32'))
                right = min(img.size[0], np.floor(right).astype('int32'))

                dir_save_path = "img_crop"
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                crop_image = img.crop([left, top, right, bottom])
                crop_image.save(os.path.join(dir_save_path, "crop_" + str(i) + ".png"), quality=95, subsampling=0)
                print("save crop_" + str(i) + ".png to " + dir_save_path)

        '''
        绘制图像
        '''
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = top_conf[i]

            top, left, bottom, right = box

            top = max(0, np.floor(top).astype('int32'))
            left = max(0, np.floor(left).astype('int32'))
            bottom = min(img.size[1], np.floor(bottom).astype('int32'))
            right = min(img.size[0], np.floor(right).astype('int32'))

            label = f'{predicted_class} {score:.2f}'
            draw = ImageDraw.Draw(img)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, str(label, 'utf-8'), fill=(0, 0, 0), font=font)

            del draw

        return img

    def get_FPS(self, img, test_interval):
        img_shape = np.array(np.shape(img)[0:2])
        '''
        将图像转化为RGB模式
        '''
        img = cvtColor(img)

        '''
        根据需求，觉得是否给图像进行不失真的resize
        '''
        img_data = resize_img(img, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)

        '''
        增加batch_size维度，并进行图像预处理，归一化
        '''
        img_data = preprocess_input(np.expand_dims(np.array(img_data, dtype=np.float32), 0))

        preds = self.get_pred(img_data).numpy()
        '''
        将预测结果进行解码
        '''
        results = self.bbox_util.decode_box(preds, self.anchors, img_shape, self.input_shape,
                                            self.letterbox_image, confidence=self.confidence)

        t1 = time.time()
        for _ in range(test_interval):
            preds = self.get_pred(img_data).numpy()
            results = self.bbox_util.decode_box(preds, self.anchors, image_shape,
                                                self.input_shape, self.letterbox_image, confidence=self.confidence)

        t2 = time.time()
        tact_time = (t2 - t1) / test_interval

        return tact_time

    def get_map_txt(self, img_id, img, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/" + img_id + ".txt"), "w")
        img_shape = np.array(np.shape(img)[0:2])
        '''
        将图像转化为RGB模式
        '''
        img = cvtColor(img)

        '''
        根据需求，觉得是否给图像进行不失真的resize
        '''
        img_data = resize_img(img, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)

        '''
        增加batch_size维度，并进行图像预处理，归一化
        '''
        img_data = preprocess_input(np.expand_dims(np.array(img_data, dtype=np.float32), 0))

        preds = self.get_pred(img_data).numpy()
        '''
        将预测结果进行解码
        '''
        results = self.bbox_util.decode_box(preds, self.anchors, img_shape, self.input_shape,
                                            self.letterbox_image, confidence=self.confidence)
        '''
        如果没有检测到物体，则返回
        '''
        if len(results[0]) <= 0:
            return

        top_label = results[0][:, 4]
        top_conf = results[0][:, 5]
        top_boxes = results[0][:, :4]

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = str(top_conf[i])

            top, left, bottom, right = box

            if predicted_class not in class_names:
                continue

            f.write(f"{predicted_class} {score[:6]} {str(int(left))} {int(top)} {int(right)} {int(bottom)} \n")

        f.close()
        return
