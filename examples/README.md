# Examples

## Dense
```python
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(16))
```

![dense](https://media.githubusercontent.com/media/0h-n0/tf_conceptual_graph/master/examples/outputs/dense_graph.png)

## Conv2d
```python
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(16, 3, 1, input_shape=[28, 28, 3]))
model.add(tf.keras.layers.Conv2D(32, 4, 2))
model.add(tf.keras.layers.Conv2D(64, 5, 3))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(32))
model.add(tf.keras.layers.ReLU())
model.add(tf.keras.layers.Dense(16))
```

![conv2d](https://media.githubusercontent.com/media/0h-n0/tf_conceptual_graph/master/examples/outputs/conv_graph.png)

## Old_style
```python
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
```

![old_style](https://media.githubusercontent.com/media/0h-n0/tf_conceptual_graph/master/examples/outputs/old_style_graph.png)


## Multitask
```python
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
```
![multitask](https://media.githubusercontent.com/media/0h-n0/tf_conceptual_graph/master/examples/outputs/multitask_graph.png)


## Multimodal
```python
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
```

![multimodal](https://media.githubusercontent.com/media/0h-n0/tf_conceptual_graph/master/examples/outputs/multimodal_graph.png)


## Multimodal_Multitask
```python
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
```

![multimodal_multitask](https://media.githubusercontent.com/media/0h-n0/tf_conceptual_graph/master/examples/outputs/multimodal_multitask_graph.png)
