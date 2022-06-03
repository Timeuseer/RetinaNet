#! /usr/bin/python3
# -*- coding:utf-8 -*-

'''
@Author : Gabriel
@About : 
'''

import time
import cv2
import yaml
import tensorflow as tf
import numpy as np
from PIL import Image
from retinanet import RetinaNet

if __name__ == '__main__':
    retinanet = RetinaNet()
    config = yaml.load(open('config/predict.yaml', 'r', encoding='utf-8'))
    if config['mode'] == "predict":
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = retinanet.detect_img(image, crop=config['crop'])
                r_image.show()

    elif config['mode'] == "video":
        capture = cv2.VideoCapture(config['video_path'])
        if config['video_save_path'] != "":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(config['video_save_path'], fourcc, config['video_fps'], size)

        fps = 0.0
        while (True):
            t1 = time.time()
            # 读取某一帧
            ref, frame = capture.read()
            # 格式转变，BGRtoRGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 转变成Image
            frame = Image.fromarray(np.uint8(frame))
            # 进行检测
            frame = np.array(retinanet.detect_img(frame, crop=config.crop))
            # RGBtoBGR满足opencv显示格式
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            fps = (fps + (1. / (time.time() - t1))) / 2
            print("fps= %.2f" % (fps))
            frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("video", frame)
            c = cv2.waitKey(1) & 0xff
            if config['video_save_path'] != "":
                out.write(frame)

            if c == 27:
                capture.release()
                break
        capture.release()
        out.release()
        cv2.destroyAllWindows()

    elif config['mode'] == "fps":
        img = Image.open('img/street.jpg')
        tact_time = retinanet.get_FPS(img, config['test_interval'])
        print(str(tact_time) + ' seconds, ' + str(1 / tact_time) + 'FPS, @batch_size 1')

    elif config['mode'] == "dir_predict":
        import os

        from tqdm import tqdm

        img_names = os.listdir(config['dir_origin_path'])
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(
                    ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path = os.path.join(config['dir_origin_path'], img_name)
                image = Image.open(image_path)
                r_image = retinanet.detect_img(image, crop=config.crop)
                if not os.path.exists(config['dir_save_path']):
                    os.makedirs(config['dir_save_path'])
                r_image.save(os.path.join(config['dir_save_path'], img_name))

    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")
