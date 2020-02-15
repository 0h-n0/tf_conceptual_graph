import numpy as np
import tensorflow as tf
import tfcg


def build(x):
    x = tf.keras.layers.Conv2D(16, 3, input_shape=[28, 28, 3])(x)
    x = tf.keras.layers.Conv2D(32, 1)(x)
    x = tf.keras.layers.Conv2D(64, 2)(x)
    x = tf.keras.layers.Conv2D(128, 2)(x)
    x = tf.keras.layers.Flatten()(x)
    x1 = tf.keras.layers.Dense(32)(x)
    x1 = tf.keras.layers.ReLU()(x1)
    x1 = tf.keras.layers.Dense(16)(x1)

    x2 = tf.keras.layers.Dense(128)(x)
    x2 = tf.keras.layers.ReLU()(x2)
    x2 = tf.keras.layers.Dense(64)(x2)
    x2 = tf.keras.layers.ReLU()(x2)
    x2 = tf.keras.layers.Dense(32)(x2)
    x2 = tf.keras.layers.ReLU()(x2)
    x2 = tf.keras.layers.Dense(16)(x2)

    return x, x2

with tf.Graph().as_default() as graph:
    x = np.random.rand(128, 28, 28, 3)
    x_p = tf.placeholder(tf.float32, [None, 28, 28, 3])
    out1, out2 = build(x_p)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        o = sess.run((out1, out2), feed_dict={x_p: x})
        parser = tfcg.from_graph_def(sess.graph_def)
        parser.dump_img('outputs/multitask_graph.png')
        parser.dump_yml('outputs/multitask_graph.yml')
        parser.dump_gml('outputs/multitask_graph.gml')
