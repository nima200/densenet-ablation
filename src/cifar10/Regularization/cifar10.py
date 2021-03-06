from __future__ import print_function

import os.path
import sys
import densenet
import numpy as np
import sklearn.metrics as metrics
from random import shuffle

from keras.datasets import cifar10
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from keras import backend as K
import tensorflow as tf
# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

batch_size = 100
nb_classes = 10
nb_epoch = 40

img_rows, img_cols = 32, 32
img_channels = 3

img_dim = (img_channels, img_rows, img_cols) if K.image_dim_ordering() == "th" else (img_rows, img_cols, img_channels)
depth = 40
nb_dense_block = 3
growth_rate = 12
nb_filter = -1
dropout_rate = 0.0  # 0.0 for data augmentation

if not os.path.exists("weights"):
    os.makedirs("weights")

if len(sys.argv) > 1:
    augment = sys.argv[1]
else:
    augment = 'true'

load_models = False

model = densenet.DenseNet(img_dim, classes=nb_classes, depth=depth, nb_dense_block=nb_dense_block,
                          growth_rate=growth_rate, nb_filter=nb_filter, dropout_rate=dropout_rate, weights=None,
                          bottleneck=False)
print("Model created")

# model.summary()
optimizer = Adam(lr=1e-3)  # Using Adam instead of SGD to speed up training
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])
print("Finished compiling")
print("Building model...")

(trainX, trainY), (testX, testY) = cifar10.load_data()

trainX = trainX.astype('float32')
testX = testX.astype('float32')

trainX = densenet.preprocess_input(trainX)
testX = densenet.preprocess_input(testX)

Y_train = np_utils.to_categorical(trainY, nb_classes)
Y_test = np_utils.to_categorical(testY, nb_classes)

generator = ImageDataGenerator(rotation_range=15,
                               width_shift_range=5. / 32,
                               height_shift_range=5. / 32,
                               horizontal_flip=True)

# Here we zip the data and its classes so they get moved around together
train = zip(trainX, trainY)

# Then we sort the zipped list by its classes
train = sorted(train, key=lambda x: x[1])

# Then we turn each class into it's categorical representation
train = list(map((lambda d: (d[0], np_utils.to_categorical(d[1], nb_classes)[0])), train))

lr_reducer = ReduceLROnPlateau(monitor='val_acc', factor=np.sqrt(0.1),
                               cooldown=0, patience=5, min_lr=1e-5)

for data_size in [100, 1000, 10000, 50000]:

    # Load model
    weights_file = "weights/DenseNet-40-12-CIFAR10-%s-bottleneck=%s.h5" % (str(data_size), "False")

    if os.path.exists(weights_file) and load_models:
        model.load_weights(weights_file, by_name=True)
        print("Model loaded.")

    if not os.path.exists(weights_file):
        file = open(weights_file, 'w')

    model_checkpoint = ModelCheckpoint(weights_file, monitor="val_acc", save_best_only=True,
                                       save_weights_only=True, verbose=1)

    csv = CSVLogger("Densenet-40-12-CIFAR10-Size-%s-%s.csv" % (str(data_size), "False"), separator=',')

    callbacks = [lr_reducer, model_checkpoint, csv]

    training_data = []

    if data_size == 50000:
        training_data = train
    else:
        for x in range(0, 50000, 500):
            training_data = training_data + train[x: x + int(data_size/10)]

    shuffle(training_data)
    x, y = map(list, zip(*training_data))
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)

    generator.fit(x, seed=0)

    try:
        if augment == 'true':
            print("Training with data augmentation...")
            model.fit_generator(generator.flow(x, y, batch_size=batch_size),
                                steps_per_epoch=len(x) // batch_size, epochs=nb_epoch,
                                callbacks=callbacks,
                                validation_data=(testX, Y_test),
                                validation_steps=testX.shape[0] // batch_size, verbose=1)
        else:
            print("Training without data augmentation...")
            model.fit(x, y, batch_size=batch_size, epochs=nb_epoch, callbacks=callbacks,
                      validation_data=(testX, Y_test), verbose=2)
    except KeyboardInterrupt:
        print("Training interrupted")
        sys.exit(1)

    yPreds = model.predict(testX)
    yPred = np.argmax(yPreds, axis=1)
    yTrue = testY

    accuracy = metrics.accuracy_score(yTrue, yPred) * 100
    error = 100 - accuracy
    print("Accuracy : ", accuracy)
    print("Error : ", error)
