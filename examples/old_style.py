#!/usr/bin/env python
import numpy as np
import tensorflow as tf
import tfcg


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=(shape,))
    return tf.Variable(initial)

def oldstype_dense(x, shape=(100, 10), name='fc'):
    with tf.name_scope(name):
        w = weight_variable(shape)
        b = bias_variable(shape[1])
        y = tf.nn.relu(tf.matmul(x, w) + b)
    return y

def build(x_p):
    x_p = oldstype_dense(x_p, shape=(500, 250))
    x_p = oldstype_dense(x_p, shape=(250, 100))
    x_p = oldstype_dense(x_p, shape=(100, 50))
    x_p = oldstype_dense(x_p, shape=(50, 10))
    return x_p

with tf.Graph().as_default() as graph:
    x = np.random.rand(128, 500)
    x_p = tf.placeholder(tf.float32, [None, 500])
    out_p = build(x_p)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        o = sess.run(out_p, feed_dict={x_p: x})
        parser = tfcg.from_graph_def(sess.graph_def)
        parser.dump_img('outputs/old_style_graph.png')
        parser.dump_yml('outputs/old_style_graph.yml')
        parser.dump_gml('outputs/old_style_graph.gml')
