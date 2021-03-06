from __future__ import print_function

import os.path
import sys
import densenet
import numpy as np
import sklearn.metrics as metrics

from keras.datasets import fashion_mnist
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from keras import backend as K

batch_size = 100 
nb_classes = 10
nb_epoch = 40

img_rows, img_cols = 28, 28
img_channels = 1

img_dim = (img_channels, img_rows, img_cols) if K.image_dim_ordering() == "th" else (img_rows, img_cols, img_channels)
depth = 40


nb_dense_block = [2, 3, 4, 6]
growth_rate = 12
nb_filter = -1
dropout_rate = 0.0 # 0.0 for data augmentation
if len(sys.argv) > 1:
    augment = sys.argv[1]
else:
    augment = 'false'
for nb_db in nb_dense_block:
    count = int((depth - 4))
    nb_dense_layers = [int(count/nb_db) for _ in range(nb_db)]
    model = densenet.DenseNet(img_dim, classes=nb_classes, depth=depth, nb_dense_block=nb_db,
                              nb_layers_per_block=nb_dense_layers, growth_rate=growth_rate, nb_filter=nb_filter,
                              dropout_rate=dropout_rate, weights=None)
    print("Model created")

    model.summary()
    optimizer = Adam(lr=1e-3)  # Using Adam instead of SGD to speed up training
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])
    print("Finished compiling")
    print("Building model...")

    (trainX, trainY), (testX, testY) = fashion_mnist.load_data()

    trainX = trainX.astype('float32')
    testX = testX.astype('float32')

    trainX = densenet.preprocess_input(trainX)
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], trainX.shape[2], img_channels))
    testX = densenet.preprocess_input(testX)
    testX = np.reshape(testX, (testX.shape[0], testX.shape[1], testX.shape[2], img_channels))

    Y_train = np_utils.to_categorical(trainY, nb_classes)
    Y_test = np_utils.to_categorical(testY, nb_classes)

    generator = ImageDataGenerator(rotation_range=15,
                                   width_shift_range=5. / 32,
                                   height_shift_range=5. / 32,
                                   horizontal_flip=True)

    generator.fit(trainX, seed=0)

    # Load model
    weights_file = "weights/DenseNet-40-12-SVHN-%d_blocks.h5" % nb_db
    if os.path.exists(weights_file):
        # model.load_weights(weights_file, by_name=True)
        print("Model loaded.")

    out_dir = "weights/"

    lr_reducer = ReduceLROnPlateau(monitor='val_acc', factor=np.sqrt(0.1),
                                   cooldown=0, patience=5, min_lr=1e-5)
    model_checkpoint = ModelCheckpoint(weights_file, monitor="val_acc", save_best_only=True,
                                       save_weights_only=True, verbose=1)

    csv = CSVLogger("Densenet-40-12-SVHN-%d_blocks.csv" % nb_db, separator=',')

    callbacks = [lr_reducer, model_checkpoint, csv]
    try:
        if augment == 'true':
            print("Training with data augmentation...")
            model.fit_generator(generator.flow(trainX, Y_train, batch_size=batch_size),
                                steps_per_epoch=len(trainX) // batch_size, epochs=nb_epoch,
                                callbacks=callbacks,
                                validation_data=(testX, Y_test),
                                validation_steps=testX.shape[0] // batch_size, verbose=1)
        else:
            print("Training without data augmentation...")
            model.fit(trainX, Y_train, batch_size=batch_size, epochs=nb_epoch, callbacks=callbacks,
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
