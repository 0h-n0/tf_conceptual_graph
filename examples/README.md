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

## Mnist

```python
def create_model():
    model = tf.keras.models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.Conv2D(64, (3, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    return model

def main():
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train = X_train.astype(np.float32).reshape((60000, 28, 28, 1)) / 255.0
    X_test = X_test.astype(np.float32).reshape((10000, 28, 28, 1)) / 255.0

    model = create_model()
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    acc = tf.keras.metrics.SparseCategoricalAccuracy()
    optim = tf.keras.optimizers.Adam()

    # train
    model.compile(optimizer=optim, loss=loss, metrics=[acc])
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1, batch_size=2048)
    val_loss, val_acc = model.evaluate(X_test, y_test, batch_size=2048)
    parser = tfcg.from_graph_def(tf.get_default_graph().as_graph_def())
    parser.dump_img("outputs/mnist_graph.png")
    parser.dump_yml("outputs/mnist_graph.png")
```
![mnist](https://media.githubusercontent.com/media/0h-n0/tf_conceptual_graph/master/examples/outputs/mnist_graph.png)

## keras_custom_layer

```python
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
```

![keras_custom_layer](https://media.githubusercontent.com/media/0h-n0/tf_conceptual_graph/master/examples/outputs/keras_custom_layer_graph.png)
