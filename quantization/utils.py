#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: utils.py
import multiprocessing
from abc import abstractmethod
import cv2, numpy as np
import tensorflow as tf
from tensorpack import imgaug, dataset, ModelDesc, InputDesc
from tensorpack.dataflow import AugmentImageComponent, PrefetchDataZMQ, BatchData, MultiThreadMapData
from tensorpack.models import regularize_cost
from tensorpack.predict import PredictConfig, SimpleDatasetPredictor
from tensorpack.tfutils import get_global_step_var
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.utils.stats import RatioCounter
from tensorpack.tfutils.tower import get_current_tower_context

__all__ = ['fbresnet_augmentor', 'normal_augmentor', 'get_imagenet_dataflow', 'ImageNetModel',
           'eval_on_ILSVRC12']

DEFAULT_IMAGE_SHAPE = 224
SOFTMAX_TEM = 1
DISTILL_TYPE = 'top-down'  # 'top-down' or 'direct'


class GoogleNetResize(imgaug.ImageAugmentor):
    """
    crop 8%~100% of the original image
    See `Going Deeper with Convolutions` by Google.
    """

    def __init__(self, crop_area_fraction=0.08, aspect_ratio_low=0.75, aspect_ratio_high=1.333,
                 target_shape=DEFAULT_IMAGE_SHAPE):
        self._init(locals())

    def _augment(self, img, _):
        h, w = img.shape[:2]
        area = h * w
        for _ in range(10):
            targetArea = self.rng.uniform(self.crop_area_fraction, 1.0) * area
            aspectR = self.rng.uniform(self.aspect_ratio_low, self.aspect_ratio_high)
            ww = int(np.sqrt(targetArea * aspectR) + 0.5)
            hh = int(np.sqrt(targetArea / aspectR) + 0.5)
            if self.rng.uniform() < 0.5:
                ww, hh = hh, ww
            if hh <= h and ww <= w:
                x1 = 0 if w == ww else self.rng.randint(0, w - ww)
                y1 = 0 if h == hh else self.rng.randint(0, h - hh)
                out = img[y1:y1 + hh, x1:x1 + ww]
                out = cv2.resize(out, (self.target_shape, self.target_shape), interpolation=cv2.INTER_CUBIC)
                return out
        out = imgaug.ResizeShortestEdge(self.target_shape, interp=cv2.INTER_CUBIC).augment(img)
        out = imgaug.CenterCrop(self.target_shape).augment(out)
        return out


def fbresnet_augmentor(isTrain):
    """
    Augmentor used in fb.resnet.torch, for BGR images in range [0,255].
    """
    if isTrain:
        augmentors = [
            GoogleNetResize(),
            # It's OK to remove the following augs if your CPU is not fast enough.
            # Removing brightness/contrast/saturation does not have a significant effect on accuracy.
            # Removing lighting leads to a tiny drop in accuracy.
            imgaug.RandomOrderAug([
                # imgaug.BrightnessScale((0.6, 1.4), clip=False),
                # imgaug.Contrast((0.6, 1.4), clip=False),
                # imgaug.Saturation(0.4, rgb=False),
                # rgb-bgr conversion for the constants copied from fb.resnet.torch
                imgaug.Lighting(0.1,
                                eigval=np.asarray(
                                    [0.2175, 0.0188, 0.0045][::-1]) * 255.0,
                                eigvec=np.array(
                                    [[-0.5675, 0.7192, 0.4009],
                                    [-0.5808, -0.0045, -0.8140],
                                    [-0.5836, -0.6948, 0.4203]],
                                    dtype='float32')[::-1, ::-1]
                                )
            ]),
            imgaug.Flip(horiz=True),
        ]
    else:
        augmentors = [
            imgaug.ResizeShortestEdge(256, cv2.INTER_CUBIC),
            imgaug.CenterCrop((DEFAULT_IMAGE_SHAPE, DEFAULT_IMAGE_SHAPE)),
        ]
    return augmentors


def normal_augmentor(isTrain):
    """
    Normal augmentor with random crop and flip only, for BGR images in range [0,255].
    """
    if isTrain:
        augmentors = [
            imgaug.ResizeShortestEdge(256, cv2.INTER_CUBIC),
            imgaug.RandomCrop((DEFAULT_IMAGE_SHAPE, DEFAULT_IMAGE_SHAPE)),
            imgaug.Flip(horiz=True),
        ]
    else:
        augmentors = [
            imgaug.ResizeShortestEdge(256, cv2.INTER_CUBIC),
            imgaug.CenterCrop((DEFAULT_IMAGE_SHAPE, DEFAULT_IMAGE_SHAPE)),
        ]
    return augmentors


def get_imagenet_dataflow(datadir, name, batch_size, augmentors, parallel=None):
    """
    See explanations in the tutorial:
    http://tensorpack.readthedocs.io/en/latest/tutorial/efficient-dataflow.html
    """
    assert name in ['train', 'val', 'test']
    assert datadir is not None
    assert isinstance(augmentors, list)
    isTrain = name == 'train'
    if parallel is None:
        parallel = min(30, multiprocessing.cpu_count())
    if isTrain:
        ds = dataset.ILSVRC12(datadir, name, shuffle=True)
        ds = AugmentImageComponent(ds, augmentors, copy=False)
        ds = PrefetchDataZMQ(ds, parallel)
        ds = BatchData(ds, batch_size, remainder=False)
    else:
        ds = dataset.ILSVRC12Files(datadir, name, shuffle=False)
        aug = imgaug.AugmentorList(augmentors)

        def mapf(dp):
            fname, cls = dp
            im = cv2.imread(fname, cv2.IMREAD_COLOR)
            im = aug.augment(im)
            return im, cls

        ds = MultiThreadMapData(ds, parallel, mapf, buffer_size=2000, strict=True)
        ds = BatchData(ds, batch_size, remainder=True)
        ds = PrefetchDataZMQ(ds, 1)
    return ds


def eval_on_ILSVRC12(model, scale, sessinit, dataflow):
    pred_config = PredictConfig(
        model=model,
        session_init=sessinit,
        input_names=['input', 'label'],
        output_names=['wrong-scale%03d-top1' % scale, 'wrong-scale%03d-top5' % scale]
    )
    pred = SimpleDatasetPredictor(pred_config, dataflow)
    acc1, acc5 = RatioCounter(), RatioCounter()
    for top1, top5 in pred.get_result():
        batch_size = top1.shape[0]
        acc1.feed(top1.sum(), batch_size)
        acc5.feed(top5.sum(), batch_size)
    print('Top1/Top5 Acc: %.1f/%.1f' % (100 - 100 * acc1.ratio, 100 - 100 * acc5.ratio))


class ImageNetModel(ModelDesc):
    """
    image_dtype: 
        uint8 instead of float32 is used as input type to reduce copy overhead.
        It might hurt the performance a liiiitle bit.
        The pretrained models were trained with float32.
    data_format:
        Either 'NCHW' or 'NHWC'
    image_bgr:
        Whether the image is BGR or RGB. If using DataFlow, then it should be BGR.
    weight_decay_pattern:
        To apply on normalization parameters, use '.*/W|.*/gamma|.*/beta'
    """
    image_dtype = tf.uint8
    data_format = 'NCHW'
    image_bgr = True
    weight_decay = 5e-5
    weight_decay_pattern = '.*/W'

    def __init__(self, scales, data_format='NCHW', wd=5e-5, learning_rate=0.1, 
                 data_aug=True, distill=False, double_iter=False):
        if data_format == 'NCHW':
            assert tf.test.is_gpu_available()
        self.scales = scales
        self.data_format = data_format
        self.weight_decay = wd
        self.data_aug = data_aug
        self.lr = learning_rate
        self.double_iter = double_iter
        self.image_shape = DEFAULT_IMAGE_SHAPE
        self.distill = distill

    def inputs(self):
        return [tf.TensorSpec([None, self.image_shape, self.image_shape, 3], self.image_dtype, 'input'),
                tf.TensorSpec([None], tf.int32, 'label')]

    def build_graph(self, image, label):
        is_training = get_current_tower_context().is_main_training_tower
        image_origin = ImageNetModel.image_preprocess(image, bgr=self.image_bgr)  # [N, H, W, C]
        loss, logit = 0, {}
        scales = sorted(self.scales, reverse=True)
        # sorted_scales = sorted(list(set(scales + self.scales)), reverse=True)
        for scale in scales:
            image = tf.image.resize_images(image_origin, [scale, scale],
                                           method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            if self.data_format == 'NCHW':
                image = tf.transpose(image, [0, 3, 1, 2])
            with tf.variable_scope('imagenet', reuse=tf.AUTO_REUSE):
                logit[scale] = self.get_logits(image, scale)
            loss_scale = self.compute_loss_and_error(logit[scale], label, scale, is_training)
            loss += loss_scale

        if self.distill:
            logit_ensemble = 0
            alpha = tf.get_variable('alpha', [len(scales)], initializer=tf.constant_initializer(1))
            alpha_soft = tf.nn.softmax(alpha)  # TODO: remove softmax
            for i, scale in enumerate(scales):
                logit_ensemble += alpha_soft[i] * tf.stop_gradient(logit[scale])
                tf.summary.scalar('alpha%03d' % scale, alpha_soft[i])
            loss_ensemble = self.compute_loss_and_error(logit_ensemble, label,
                                                        'ensemble', is_training)
            loss += loss_ensemble
            loss_distill = 0
            soft_label = tf.stop_gradient(tf.nn.softmax(logit_ensemble))
            for scale in scales:
                loss_distill += self.compute_distill_loss(logit[scale], soft_label)
            if DISTILL_TYPE == 'top-down':
                for i in range(len(scales) - 1):
                    soft_label = tf.stop_gradient(tf.nn.softmax(logit[scales[i]]))
                    for j in range(i + 1, len(scales)):
                        loss_distill += self.compute_distill_loss(logit[scales[j]], soft_label)
                distill_num = len(scales) * (len(scales) + 1) / 2
                loss += SOFTMAX_TEM ** 2 * loss_distill / distill_num * len(scales)
            else:
                loss += SOFTMAX_TEM ** 2 * loss_distill

        wd_loss = regularize_cost(self.weight_decay_pattern,
                                  tf.contrib.layers.l2_regularizer(self.weight_decay),
                                  name='l2_regularize_loss')
        add_moving_summary(loss, wd_loss)
        self.cost = tf.add_n([loss, wd_loss], name='cost')
        return self.cost

    @abstractmethod
    def get_logits(self, image, scale):
        """
        Args:
            image: 4D tensor of ``self.input_shape`` in ``self.data_format``

        Returns:
            Nx#class logits
        """

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=self.lr, trainable=False)
        if not self.data_aug:
            global_step = get_global_step_var()
            lr = tf.train.polynomial_decay(self.lr, global_step,
                                           640000 if self.double_iter else 320000, 1e-5, power=1.)
        tf.summary.scalar('learning_rate', lr)
        return tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)

    @staticmethod
    def image_preprocess(image, bgr=True):
        with tf.name_scope('image_preprocess'):
            if image.dtype.base_dtype != tf.float32:
                image = tf.cast(image, tf.float32)
            image = image * (1.0 / 255)
            mean = [0.485, 0.456, 0.406]  # rgb
            std = [0.229, 0.224, 0.225]
            if bgr:
                mean, std = mean[::-1], std[::-1]
            image_mean = tf.constant(mean, dtype=tf.float32)
            image_std = tf.constant(std, dtype=tf.float32)
            image = (image - image_mean) / image_std
            return image

    def compute_loss_and_error(self, logit, label, scale, is_training):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=label)
        loss = tf.reduce_mean(loss, name='xentropy-loss')

        def prediction_incorrect(logit, label, topk=1, name='incorrect_vector'):
            with tf.name_scope('prediction_incorrect'):
                x = tf.logical_not(tf.nn.in_top_k(logit, label, topk))
            return tf.cast(x, tf.float32, name=name)

        scale = '%03d' % scale if scale != 'ensemble' else '_%s' % scale
        wrong = prediction_incorrect(logit, label, 1, name='wrong-scale%s-top1' % scale)
        add_moving_summary(tf.reduce_mean(wrong, name='train-error-scale%s-top1' % scale))
        wrong = prediction_incorrect(logit, label, 5, name='wrong-scale%s-top5' % scale)
        add_moving_summary(tf.reduce_mean(wrong, name='train-error-scale%s-top5' % scale))

        return loss

    def compute_distill_loss(self, logit, soft_label):
        logit = logit / SOFTMAX_TEM
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=soft_label)
        loss = tf.reduce_mean(loss, name='kl-loss')
        return loss
