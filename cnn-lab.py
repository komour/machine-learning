import keras
from keras.datasets import mnist
from keras.datasets import fashion_mnist

# load mnist dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
(X_train_fashion, y_train_fashion), (X_test_fashion, y_test_fashion) = fashion_mnist.load_data()

import matplotlib.pyplot as plt

for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.title(f'Number: {y_test[i]}')
    plt.imshow(X_test[i])
    plt.axis('off')

print(f"train data amount: {X_train.shape[0]}, each image has shape {X_train.shape[1]}x{X_train.shape[2]}")
print(f"test data amount: {X_test.shape[0]}, each image has shape {X_test.shape[1]}x{X_test.shape[2]}")
