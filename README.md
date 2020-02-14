[![Build Status](https://travis-ci.com/0h-n0/tf_conceptual_graph.svg?token=fnVzZYoHYzREzRx4L8BP&branch=master)](https://travis-ci.com/0h-n0/tf_conceptual_graph)
# tf_conceptual_graph
Create tensorflow conceptual graph.


## Installtion

```shell
$ pip install tfcg
```


## Usage

read a graph from object api(`sess.graph`)

```python
import numpy as np
import tensorflow as tf

import tfcg

with tf.Graph().as_default() as graph:
    model = tf.keras.Sequential()
    x = np.random.rand(128, 28, 28, 3)
    model.add(tf.keras.layers.Conv2D(16, 3, input_shape=[28, 28, 3], name='conv1'))
    model.add(tf.keras.layers.Conv2D(32, 1, name='conv2'))
    model.add(tf.keras.layers.Conv2D(64, 2, name='conv3'))
    model.add(tf.keras.layers.Conv2D(128, 2, name='conv4'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(32, name='dense1'))
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.Dense(16, name='dense2'))
    x_p = tf.placeholder(tf.float32, [None, 28, 28, 3], name='input')
    out_p = model(x_p)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        o = sess.run(out_p, feed_dict={x_p: x})
        _ = tf.identity(o, name="output")
        tf.io.write_graph(sess.graph, './', 'train.pbtxt')
        parser = tfcg.from_sess(sess.graph)
        parser.dump("conceptual_graph.json")

```

read a graph from a file, After dumpping a tensorflow graph file.

```python
import tfcg

parser = tfcg.from_file(sess.graph)
parser.dump("conceptual_graph.json")
```
