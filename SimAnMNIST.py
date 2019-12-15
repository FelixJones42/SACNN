#Based on code from https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/quickstart/advanced.ipynb#scrollTo=i-2pkctU_Ci7

import tensorflow as tf
#import numpy as np

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
import numpy as np

tf.keras.backend.set_floatx('float64')

#print(tf.__version__)

def Do_Simulation(EPOCHS,T,version_title):

    tf.random.set_seed(1234)

    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Add a channels dimension
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).shuffle(10000).batch(32)

    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)


    class MyModel(Model):
        def __init__(self):
            super(MyModel, self).__init__()
            self.conv1 = Conv2D(32, 3, activation='relu')
            self.flatten = Flatten()
            self.d1 = Dense(128, activation='relu')
            self.d2 = Dense(10)
            self.d2.activation = lambda x: tf.exp(-x**2) / tf.reduce_sum(tf.exp(-x**2))


        def call(self, x):
            x = self.conv1(x)
            x = self.flatten(x)
            x = self.d1(x)
            return self.d2(x)


    # Create an instance of the model
    model = MyModel()
    #model.add(Activation(, name='softmax_temp')

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy() #Moving gradient in direction of KL Divergence
    optimizer = tf.keras.optimizers.Adam()

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    @tf.function
    def train_step(images, labels, model):
        with tf.GradientTape() as tape:
            predictions = model(images)
            loss = loss_object(labels, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)

    @tf.function
    def test_step(images, labels):
        predictions = model(images)
        t_loss = loss_object(labels, predictions)

        test_loss(t_loss)
        test_accuracy(labels, predictions)

    train_results=[]
    test_results = []


    for epoch in range(EPOCHS):

        for images, labels in train_ds:
            train_step(images, labels, model)

        for test_images, test_labels in test_ds:
            test_step(test_images, test_labels)

        template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print(template.format(epoch+1,
                              train_loss.result(),
                              train_accuracy.result()*100,
                              test_loss.result(),
                              test_accuracy.result()*100))

        train_results.append(train_accuracy.result()*100)
        test_results.append(test_accuracy.result()*100)

        # Reset the metrics for the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        @tf.function
        def train_step(images, labels, model):
            with tf.GradientTape() as tape:
                predictions = model(images)
                loss = loss_object(labels, predictions)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            train_loss(loss)
            train_accuracy(labels, predictions)

        model.d2.activation = lambda x: tf.exp(-(x**2)/T[epoch]) / tf.reduce_sum(tf.exp(-(x**2)/T[epoch]))

    @tf.function
    def test_step(images, labels):
        predictions = model(images)
        t_loss = loss_object(labels, predictions)

        test_loss(t_loss)
        test_accuracy(labels, predictions)

    model.d2.activation = lambda x: tf.exp(-x**2) / tf.reduce_sum(tf.exp(-x**2))

    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)

    template = 'Final Test {}, Test Loss: {}, Test Accuracy: {}'

    print(template.format(epoch+1,
                          test_loss.result(),
                          test_accuracy.result()*100))

    test_results.append(test_accuracy.result()*100)

    np.savetxt(version_title.format('train accuracy'),train_results)
    np.savetxt(version_title.format('test accuracy'),test_results)
