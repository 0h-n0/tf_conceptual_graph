import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = "true"
import time
import tensorflow as tf
import tensorflow.keras.layers as layers
import numpy as np
import tfcg

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
    parser.dump_yml("outputs/mnist_graph.yml")

    print(val_loss, val_acc)

if __name__ == "__main__":
    main()
