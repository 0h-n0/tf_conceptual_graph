import tfcg
import numpy as np
import tensorflow as tf


with tf.Graph().as_default() as graph:
    model = tf.keras.Sequential()
    x = np.random.rand(128, 500)
    model.add(tf.keras.layers.Dense(16))
    x_p = tf.placeholder(tf.float32, [None, 500])
    out_p = model(x_p)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        o = sess.run(out_p, feed_dict={x_p: x})
        parser = tfcg.from_graph_def(sess.graph_def)
