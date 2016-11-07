import pytest
from keras.datasets import mnist
import numpy as np


@pytest.fixture
def mnist_dataset():
    (X_train, _), (X_test, _) = mnist.load_data()
    X_train = X_train.astype('float32') / 256.
    X_test = X_test.astype('float32') / 256.
    X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
    X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))
    return X_train, X_test
