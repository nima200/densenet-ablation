import src.net as net
from keras.datasets import cifar10
from keras.utils import to_categorical

(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

Y_test = to_categorical(Y_test, num_classes=10)
Y_train = to_categorical(Y_train, num_classes=10)

model = net.DenseNet(classes=10, input_dim=(32, 32))
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=30, epochs=10, verbose=2, validation_data=(X_test, Y_test))
