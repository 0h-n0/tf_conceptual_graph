import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import tfcg


class MyLinear(layers.Layer):
    def __init__(self, units=32, input_dim=32):
        super(MyLinear, self).__init__()
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(initial_value=w_init(shape=(input_dim, units),
                                                  dtype='float32'),
                             trainable=True)
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(initial_value=b_init(shape=(units,),
                                                  dtype='float32'),
                             trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b


with tf.Graph().as_default() as graph:
    model = tf.keras.Sequential()
    x = np.random.rand(128, 28, 28, 3)
    model.add(tf.keras.layers.Conv2D(16, 3, 1, input_shape=[28, 28, 3]))
    model.add(tf.keras.layers.Conv2D(32, 4, 2))
    model.add(tf.keras.layers.Conv2D(64, 5, 3))
    model.add(tf.keras.layers.Flatten())
    model.add(MyLinear(32, 576))
    model.add(tf.keras.layers.ReLU())
    model.add(MyLinear(16, 32))
    x_p = tf.placeholder(tf.float32, [None, 28, 28, 3])
    out_p = model(x_p)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        o = sess.run(out_p, feed_dict={x_p: x})
        parser = tfcg.from_graph_def(sess.graph_def)
        parser.dump_img('outputs/keras_custom_layer_graph.png')
        parser.dump_yml('outputs/keras_custom_layer_graph.yml')
        parser.dump_gml('outputs/keras_custom_layer_graph.gml')
