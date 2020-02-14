import numpy as np
import tensorflow as tf
import tfcg

def build(x1, x2):
    x1 = tf.keras.layers.Conv2D(16, 3, input_shape=[28, 28, 3])(x1)
    x1 = tf.keras.layers.Conv2D(32, 1)(x1)
    x1 = tf.keras.layers.Conv2D(64, 2)(x1)
    x1 = tf.keras.layers.Conv2D(128, 2)(x1)
    x1 = tf.keras.layers.Flatten()(x1)

    x2 = tf.keras.layers.Conv2D(16, 3, input_shape=[28, 28, 3])(x2)
    x2 = tf.keras.layers.Conv2D(32, 1)(x2)
    x2 = tf.keras.layers.Conv2D(64, 2)(x2)
    x2 = tf.keras.layers.Conv2D(128, 2)(x2)
    x2 = tf.keras.layers.Flatten()(x2)

    x = tf.concat([x1, x2], 1)

    y1 = tf.keras.layers.Dense(32)(x)
    y1 = tf.keras.layers.ReLU()(y1)
    y1 = tf.keras.layers.Dense(16)(y1)

    y2 = tf.keras.layers.Dense(128)(x)
    y2 = tf.keras.layers.ReLU()(y2)
    y2 = tf.keras.layers.Dense(64)(y2)
    y2 = tf.keras.layers.ReLU()(y2)
    y2 = tf.keras.layers.Dense(32)(y2)
    y2 = tf.keras.layers.ReLU()(y2)
    y2 = tf.keras.layers.Dense(16)(y2)
    return y1, y2

with tf.Graph().as_default() as graph:
    x = np.random.rand(128, 28, 28, 3)
    x1_p = tf.placeholder(tf.float32, [None, 28, 28, 3])
    x2_p = tf.placeholder(tf.float32, [None, 28, 28, 3])
    out1, out2 = build(x1_p, x2_p)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        o = sess.run((out1, out2), feed_dict={x1_p: x, x2_p: x})
        parser = tfcg.from_graph_def(sess.graph_def)
        parser.dump_img()
