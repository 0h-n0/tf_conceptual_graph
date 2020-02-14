import numpy as np
import tensorflow as tf
import tfcg


with tf.Graph().as_default() as graph:
    model = tf.keras.Sequential()
    x = np.random.rand(128, 28, 28, 3)
    model.add(tf.keras.layers.Conv2D(16, 3, 1, input_shape=[28, 28, 3]))
    model.add(tf.keras.layers.Conv2D(32, 4, 2))
    model.add(tf.keras.layers.Conv2D(64, 5, 3))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(32))
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.Dense(16))
    x_p = tf.placeholder(tf.float32, [None, 28, 28, 3])
    out_p = model(x_p)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        o = sess.run(out_p, feed_dict={x_p: x})
        parser = tfcg.from_graph_def(sess.graph_def)
        parser.dump_img()
