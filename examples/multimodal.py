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
    x = tf.keras.layers.Dense(32)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dense(16)(x)
    return x

with tf.Graph().as_default() as graph:
    x = np.random.rand(128, 28, 28, 3)
    x_p = tf.placeholder(tf.float32, [None, 28, 28, 3])
    y_p = tf.placeholder(tf.float32, [None, 28, 28, 3])
    out_p = build(x_p, y_p)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        o = sess.run(out_p, feed_dict={x_p: x, y_p: x})
        parser = tfcg.from_graph_def(sess.graph_def)
        parser.dump_img()
