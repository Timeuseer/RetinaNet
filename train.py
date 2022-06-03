#! /usr/bin/python3
# -*- coding:utf-8 -*-

'''
@Author : Gabriel
@About :
'''
import datetime
import os

import yaml
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam, SGD
from nets.retinanet import resnet_retinanet
from nets.loss import focal, get_lr_scheduler, smooth_l1
from utils.anchors import get_anchors
from utils.callbacks import (ExponentDecayScheduler, LossHistory)
from utils.load_data import RetinanetDatasets
from utils.utils import get_classes

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_virtual_device_configuration(
    gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
)

if __name__ == '__main__':
    '''
    加载配置文件
    '''
    config = yaml.load(open('config/train.yaml', 'r', encoding='utf-8'))

    '''
    获取classes和anchor
    '''
    class_names, num_classes = get_classes(classes_path=config['classes_path'])
    anchors = get_anchors(input_shape=config['input_shape'], anchors_size=config['anchors_size'])

    model = resnet_retinanet((config['input_shape'][0], config['input_shape'][1], 3), num_classes)

    if config['model_path'] != '':
        print(f'load weights {config["model_path"]}...')
        model.load_weights(config["model_path"], by_name=True, skip_mismatch=True)

    '''
    加载数据集
    '''
    with open(config['train_annotation_path'], encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(config['val_annotation_path'], encoding='utf-8') as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.DepthwiseConv2D):
            layer.add_loss(tf.keras.regularizers.l2(config['weight_decay'])(layer.depthwise_kernel))
        elif isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(tf.keras.regularizers.l2(config['weight_decay'])(layer.kernel))

    num_val = len(val_lines)

    if config['Freeze_Train']:
        freeze_layers = 174
        for i in range(freeze_layers): model.layers[i].trainable = False
        print(f'Freeze the first {freeze_layers} layers of total {len(model.layers)} layers.')

    '''
    主干特征提取网络特征通用，冻结训练可以加快速度，也可以防止训练初期破坏权值。
    Init_Epoch      为起始世代
    Freeze_Epoch    为冻结训练的世代
    Unfreeze_Epoch  总训练世代
    '''
    batch_size = config['Freeze_batch_size']
    start_epoch = config['Init_Epoch']
    end_epoch = config['Freeze_Epoch']

    '''
    判断当前batch_Size,自适应调整学习率
    '''
    nbs = 16
    lr_limit_max = 1e-4 if config['optimizer_type'] == 'adam' else 5e-2
    lr_limit_min = 1e-4 if config['optimizer_type'] == 'adam' else 5e-4
    Init_lr_fit = min(max(batch_size / nbs * config['Init_lr'], lr_limit_min), lr_limit_max)
    Min_lr_fit = min(max(batch_size / nbs * config['Init_lr'] * 0.01, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

    optimizer = {
        'adam': Adam(lr=Init_lr_fit, beta_1=config['momentum']),
        'sgd': SGD(lr=Init_lr_fit, momentum=config['momentum'], nesterov=True)
    }[config['optimizer_type']]
    model.compile(
        loss={
            'regression': smooth_l1(),
            'classification': focal()
        },
        optimizer=optimizer
    )

    '''
    获得学习率下降的公式
    '''
    lr_scheduler_func = get_lr_scheduler(config['lr_decay_type'], Init_lr_fit, Min_lr_fit, config['UnFreeze_Epoch'])

    epoch_step = num_train // batch_size
    epoch_step_val = num_val // batch_size

    if epoch_step == 0 or epoch_step_val == 0:
        raise ValueError('数据集过小，无法进行训练，请扩充数据集。')

    train_dataloader = RetinanetDatasets(train_lines, config['input_shape'], anchors, batch_size, num_classes, train=True)
    val_dataloader = RetinanetDatasets(val_lines, config['input_shape'], anchors, batch_size, num_classes, train=False)

    '''
    训练参数的设置
    logging         用于设置tensorboard的保存地址
    checkpoint      用于设置权值保存的细节，period用于修改多少epoch保存一次
    lr_scheduler       用于设置学习率下降的方式
    early_stopping  用于设定早停，val_loss多次不下降自动结束训练，表示模型基本收敛
    '''
    time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
    log_dir = os.path.join(config['save_dir'], "loss_" + str(time_str))
    logging = TensorBoard(log_dir)
    loss_history = LossHistory(log_dir)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(config['save_dir'], "ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5"),
        monitor='val_loss', save_weights_only=True, save_best_only=False, period=config['save_period'])
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_scheduler_func, verbose=1)
    callbacks = [logging, loss_history, checkpoint, lr_scheduler]

    if start_epoch < end_epoch:
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(
            generator=train_dataloader,
            steps_per_epoch=epoch_step,
            validation_data=val_dataloader,
            validation_steps=epoch_step_val,
            epochs=end_epoch,
            initial_epoch=start_epoch,
            use_multiprocessing=True if config['num_workers'] > 1 else False,
            workers=config['num_workers'],
            callbacks=callbacks
        )

    '''
    如果模型有冻结学习部分
    则解冻，并设置参数
    '''
    if config['Freeze_Train']:
        batch_size = config['UnFreeze_batch_size']
        start_epoch = config['Freeze_Epoch'] if start_epoch < config['Freeze_Epoch'] else start_epoch
        end_epoch = config['UnFreeze_Epoch']

        '''
        判断当前batch_size，自适应调整学习率
        '''
        nbs = 16
        lr_limit_max = 1e-4 if config['optimizer_type'] == 'adam' else 5e-2
        lr_limit_min = 1e-4 if config['optimizer_type'] == 'adam' else 5e-4
        Init_lr_fit = min(max(batch_size / nbs * config['Init_lr'], lr_limit_min), lr_limit_max)
        Min_lr_fit = min(max(batch_size / nbs * config['Init_lr'] * 0.01, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        '''
        获得学习率下降的公式
        '''
        lr_scheduler_func = get_lr_scheduler(config['lr_decay_type'], Init_lr_fit, Min_lr_fit, config['UnFreeze_Epoch'])
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_scheduler_func, verbose=1)
        callbacks = [logging, loss_history, checkpoint, lr_scheduler]

        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        model.compile(
            loss={
                'regression': smooth_l1(),
                'classification': focal()
            },
            optimizer=optimizer
        )

        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

        train_dataloader.batch_size = config['UnFreeze_batch_size']
        val_dataloader.batch_size = config['UnFreeze_batch_size']

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(
            generator=train_dataloader,
            steps_per_epoch=epoch_step,
            validation_data=val_dataloader,
            validation_steps=epoch_step_val,
            epochs=end_epoch,
            initial_epoch=start_epoch,
            use_multiprocessing=True if config['num_workers'] > 1 else False,
            workers=config['num_workers'],
            callbacks=callbacks
        )
