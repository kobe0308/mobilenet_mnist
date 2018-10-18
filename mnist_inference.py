#coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import mobilenet_v1

slim = tf.contrib.slim


#define the variables of nerual network
INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500

def get_weight_variable(shape, regularizer):
    weights = tf.get_variable("weights", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))

    if regularizer != None:
        tf.add_to_collection('losses', regularizer(weights))

    return weights

#define the forward network with MLPnet
def inference_MLP(input_tensor, regularizer):
    with tf.variable_scope('layer1'):
        weights = get_weight_variable([INPUT_NODE, LAYER1_NODE], regularizer)
        biases = tf.get_variable("biases", [LAYER1_NODE], initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

    with tf.variable_scope('layer2'):
        weights = get_weight_variable([LAYER1_NODE, OUTPUT_NODE], regularizer)
        biases = tf.get_variable("biases", [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1, weights) + biases

    return layer2

#define the forward network with mobilenet_v1
def inference_mobilenet(input_tensor, regularizer):
    #inputs = tf.random_uniform((batch_size, height, width, 3))
    with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                      normalizer_fn=slim.batch_norm):
        logits, end_points = mobilenet_v1.mobilenet_v1(
            input_tensor,
            num_classes=OUTPUT_NODE,
            dropout_keep_prob=0.8,
            is_training=True,
            min_depth=8,
            depth_multiplier=1.0,
            conv_defs=None,
            prediction_fn=tf.contrib.layers.softmax,
            spatial_squeeze=True,
            reuse=None,
            scope='MobilenetV1',
            global_pool=False
        )

    return logits
