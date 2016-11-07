from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division

from baumkuchen import AutoEncoder
from baumkuchen.autoencoder import gaussian_noise

from baumkuchen.tests.fixtures import mnist_dataset


def test_plain_autoencoder(mnist_dataset):
    X_train, X_test =  mnist_dataset
    model = AutoEncoder(input_dim=784, encoding_dim=32)
    model.fit(X_train, X_train,
              nb_epoch=5,
              batch_size=256,
              shuffle=True,
              validation_data=(X_test, X_test))


def test_dropout_autoencoder(mnist_dataset):
    X_train, X_test =  mnist_dataset
    model = AutoEncoder(
            input_dim=784,
            encoding_dim=32,
            dropout_proba=0.5)

    model.fit(X_train, X_train,
              nb_epoch=5,
              batch_size=256,
              shuffle=True,
              validation_data=(X_test, X_test))


def test_denoising_autoencoder(mnist_dataset):
    X_train, X_test =  mnist_dataset
    model = AutoEncoder(
            input_dim=784,
            encoding_dim=32,
            add_noise=gaussian_noise,
            dropout_proba=0.5)

    model.fit(X_train, X_train,
              nb_epoch=5,
              batch_size=256,
              shuffle=True,
              validation_data=(X_test, X_test))

