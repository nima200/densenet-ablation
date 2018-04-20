from __future__ import print_function

import sys
from densenet_slim import DenseNet, preprocess_input
import numpy as np
import sklearn.metrics as metrics

from keras.datasets import cifar10
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from keras import backend as K

batch_size = 100
nb_classes = 10
nb_epoch = 100

img_rows, img_cols = 32, 32
img_channels = 3

img_dim = (img_channels, img_rows, img_cols) if K.image_dim_ordering() == "th" else (img_rows, img_cols, img_channels)
depth = 40
nb_dense_block = 3
growth_rate = 12
nb_filter = -1

dropout_befores = [False, False, True, True]
dropout_rates = [5, 15, 5, 15]
configs = list(zip(dropout_rates, dropout_befores))
for config in configs:
    dropout = config[0]
    before = config[1]
    model = DenseNet(img_dim, classes=nb_classes, depth=depth, nb_dense_block=nb_dense_block,
                     growth_rate=growth_rate, nb_filter=nb_filter, dropout_rate=dropout, weights=None,
                     dropout_before=before, bottleneck=True)
    print("Model created")
    print("Dropout before: %s, Dropout rate: %d" % (before, dropout))

    model.summary()
    optimizer = Adam(lr=1e-3)  # Using Adam instead of SGD to speed up training
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])
    print("Finished compiling")
    print("Building model...")

    (trainX, trainY), (testX, testY) = cifar10.load_data()

    trainX = trainX.astype('float32')
    testX = testX.astype('float32')

    trainX = preprocess_input(trainX)
    testX = preprocess_input(testX)

    Y_train = np_utils.to_categorical(trainY, nb_classes)
    Y_test = np_utils.to_categorical(testY, nb_classes)

    # Load model
    weights_file="weights/DenseNet-40-12-CIFAR10-dropout_before=%s-rate=%d.h5" % (before, dropout)

    out_dir="weights/"

    lr_reducer      = ReduceLROnPlateau(monitor='val_acc', factor=np.sqrt(0.1),
                                        cooldown=0, patience=5, min_lr=1e-5)
    model_checkpoint= ModelCheckpoint(weights_file, monitor="val_acc", save_best_only=True,
                                      save_weights_only=True, verbose=1)

    csv = CSVLogger("Densenet-40-12-CIFAR10-dropout_before=%s-rate=%d.csv" % (before, dropout), separator=',')

    callbacks=[lr_reducer, model_checkpoint, csv]
    try:
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
