from __future__ import print_function

import os.path
import sys
import densenet
import numpy as np
import sklearn.metrics as metrics

from keras.datasets import cifar10
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras import backend as K

batch_size = 100
nb_classes = 10
nb_epoch = 300

img_rows, img_cols = 32, 32
img_channels = 3

img_dim = (img_channels, img_rows, img_cols) if K.image_dim_ordering() == "th" else (img_rows, img_cols, img_channels)
depth = 40
nb_dense_block = 3
growth_rate = 12
nb_filter = -1
dropout_rate = 0.0  # 0.0 for data augmentation
if len(sys.argv) > 1:
    augment = sys.argv[1]
else:
    augment = 'false'

(trainX, trainY), (testX, testY) = cifar10.load_data()

trainX = trainX.astype('float32')
testX = testX.astype('float32')

trainX = densenet.preprocess_input(trainX)
testX = densenet.preprocess_input(testX)

Y_train = np_utils.to_categorical(trainY, nb_classes)
Y_test = np_utils.to_categorical(testY, nb_classes)

# GENERATOR
generator = ImageDataGenerator(rotation_range=15,
                               width_shift_range=5. / 32,
                               height_shift_range=5. / 32,
                               horizontal_flip=True)

generator.fit(trainX, seed=0)

# MODELS

model2k = densenet.DenseNet(img_dim, classes=nb_classes, depth=depth, nb_dense_block=nb_dense_block,
                            growth_rate=growth_rate, nb_filter=nb_filter, dropout_rate=dropout_rate, weights=None,
                            bottleneck=True, growth_rate_factor=2)

print("Models created")

# 2K MODEL
print("Building model 2k...")
model2k.summary()
optimizer = Adam(lr=1e-3)  # Using Adam instead of SGD to speed up training
model2k.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])
print("Finished compiling")


# Load model
weights_file_2k = "weights/DenseNet-40-12-CIFAR10-2K.h5"
if os.path.exists(weights_file_2k):
    # model.load_weights(weights_file, by_name=True)
    print("Model loaded.")

out_dir = "weights/"
tb_dir_2k = "tensorboard/2k"

lr_reducer = ReduceLROnPlateau(monitor='val_acc', factor=np.sqrt(0.1),
                               cooldown=0, patience=5, min_lr=1e-5, verbose=1)
model_checkpoint_2k = ModelCheckpoint(weights_file_2k, monitor="val_acc", save_best_only=True,
                                      save_weights_only=True, verbose=1)
# tensorboard_2k = TensorBoard(log_dir=tb_dir_2k+"/logs", histogram_freq=5, batch_size=batch_size, write_graph=False,
#                              write_images=True, write_grads=True)

callbacks_2k = [lr_reducer, model_checkpoint_2k]
try:
    if augment == 'true':
        print("Training with data augmentation...")
        model2k.fit_generator(generator.flow(trainX, Y_train, batch_size=batch_size),
                              steps_per_epoch=len(trainX) // batch_size, epochs=nb_epoch,
                              callbacks=callbacks_2k,
                              validation_data=(testX, Y_test),
                              validation_steps=testX.shape[0] // batch_size, verbose=1)
    else:
        print("Training without data augmentation...")
        model2k.fit(trainX, Y_train, batch_size=batch_size, epochs=nb_epoch, callbacks=callbacks_2k,
                    validation_data=(testX, Y_test), verbose=2)
except KeyboardInterrupt:
    print("Training interrupted")
    sys.exit(1)

yPreds_2k = model2k.predict(testX)
yPred_2k = np.argmax(yPreds_2k, axis=1)
yTrue = testY

accuracy_2k = metrics.accuracy_score(yTrue, yPred_2k) * 100
error_2k = 100 - accuracy_2k
print("2K Accuracy : ", accuracy_2k)
print("2K Error : ", error_2k)
del model2k

# 4K MODEL
model4k = densenet.DenseNet(img_dim, classes=nb_classes, depth=depth, nb_dense_block=nb_dense_block,
                            growth_rate=growth_rate, nb_filter=nb_filter, dropout_rate=dropout_rate, weights=None,
                            bottleneck=True, growth_rate_factor=4)
print("Building model 4k...")
model4k.summary()
optimizer = Adam(lr=1e-3)  # Using Adam instead of SGD to speed up training
model4k.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])
print("Finished compiling")


# Load model
weights_file_4k = "weights/DenseNet-40-12-CIFAR10-4K.h5"
if os.path.exists(weights_file_4k):
    # model.load_weights(weights_file, by_name=True)
    print("Model loaded.")

out_dir = "weights/"
tb_dir_4k = "tensorboard/4k"

lr_reducer = ReduceLROnPlateau(monitor='val_acc', factor=np.sqrt(0.1),
                               cooldown=0, patience=5, min_lr=1e-5, verbose=1)
model_checkpoint_4k = ModelCheckpoint(weights_file_4k, monitor="val_acc", save_best_only=True,
                                      save_weights_only=True, verbose=1)

# tensorboard_4k = TensorBoard(log_dir=tb_dir_4k+"/logs", histogram_freq=5, batch_size=batch_size, write_graph=False,
#                              write_images=True, write_grads=True)

callbacks_4k = [lr_reducer, model_checkpoint_4k]
try:
    if augment == 'true':
        print("Training with data augmentation...")
        model4k.fit_generator(generator.flow(trainX, Y_train, batch_size=batch_size),
                              steps_per_epoch=len(trainX) // batch_size, epochs=nb_epoch,
                              callbacks=callbacks_4k,
                              validation_data=(testX, Y_test),
                              validation_steps=testX.shape[0] // batch_size, verbose=1)
    else:
        print("Training without data augmentation...")
        model4k.fit(trainX, Y_train, batch_size=batch_size, epochs=nb_epoch, callbacks=callbacks_4k,
                    validation_data=(testX, Y_test), verbose=2)
except KeyboardInterrupt:
    print("Training interrupted")
    sys.exit(1)

yPreds_4k = model4k.predict(testX)
yPred_4k = np.argmax(yPreds_4k, axis=1)
yTrue = testY

accuracy_4k = metrics.accuracy_score(yTrue, yPred_4k) * 100
error_4k = 100 - accuracy_4k
print("4K Accuracy : ", accuracy_4k)
print("4K Error : ", error_4k)
del model4k

# 6K MODEL
print("Building model 6k...")

model6k = densenet.DenseNet(img_dim, classes=nb_classes, depth=depth, nb_dense_block=nb_dense_block,
                            growth_rate=growth_rate, nb_filter=nb_filter, dropout_rate=dropout_rate, weights=None,
                            bottleneck=True, growth_rate_factor=6)

model6k.summary()
optimizer = Adam(lr=1e-3)  # Using Adam instead of SGD to speed up training
model6k.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])
print("Finished compiling")

# Load model
weights_file_6k = "weights/DenseNet-40-12-CIFAR10-6K.h5"
if os.path.exists(weights_file_6k):
    # model.load_weights(weights_file, by_name=True)
    print("Model loaded.")

out_dir = "weights/"
tb_dir_6k = "tensorboard/6k"

model_checkpoint_6k = ModelCheckpoint(weights_file_6k, monitor="val_acc", save_best_only=True,
                                      save_weights_only=True, verbose=1)

# tensorboard_6k = TensorBoard(log_dir=tb_dir_6k+"/logs", histogram_freq=5, batch_size=batch_size, write_graph=False,
#                              write_images=True, write_grads=True)

callbacks_6k = [lr_reducer, model_checkpoint_6k]
try:
    if augment == 'true':
        print("Training with data augmentation...")
        model6k.fit_generator(generator.flow(trainX, Y_train, batch_size=batch_size),
                              steps_per_epoch=len(trainX) // batch_size, epochs=nb_epoch,
                              callbacks=callbacks_6k,
                              validation_data=(testX, Y_test),
                              validation_steps=testX.shape[0] // batch_size, verbose=1)
    else:
        print("Training without data augmentation...")
        model6k.fit(trainX, Y_train, batch_size=batch_size, epochs=nb_epoch, callbacks=callbacks_6k,
                    validation_data=(testX, Y_test), verbose=2)
except KeyboardInterrupt:
    print("Training interrupted")
    sys.exit(1)

yPreds_6k = model6k.predict(testX)
yPred_6k = np.argmax(yPreds_6k, axis=1)

accuracy_6k = metrics.accuracy_score(yTrue, yPred_6k) * 100
error_6k = 100 - accuracy_6k
print("6K Accuracy : ", accuracy_6k)
print("6K Error : ", error_6k)
del model6k

# 8K MODEL
print("Building model 8k...")

model8k = densenet.DenseNet(img_dim, classes=nb_classes, depth=depth, nb_dense_block=nb_dense_block,
                            growth_rate=growth_rate, nb_filter=nb_filter, dropout_rate=dropout_rate, weights=None,
                            bottleneck=True, growth_rate_factor=8)

model8k.summary()
optimizer = Adam(lr=1e-3)  # Using Adam instead of SGD to speed up training
model8k.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])
print("Finished compiling")

# Load model
weights_file_8k = "weights/DenseNet-40-12-CIFAR10-8K.h5"
if os.path.exists(weights_file_8k):
    # model.load_weights(weights_file, by_name=True)
    print("Model loaded.")

out_dir = "weights/"
tb_dir_8k = "tensorboard/8k"

model_checkpoint_8k = ModelCheckpoint(weights_file_8k, monitor="val_acc", save_best_only=True,
                                      save_weights_only=True, verbose=1)


# tensorboard_8k = TensorBoard(log_dir=tb_dir_8k+"/logs", histogram_freq=5, batch_size=batch_size, write_graph=False,
#                              write_images=True, write_grads=True)

callbacks_8k = [lr_reducer, model_checkpoint_8k]
try:
    if augment == 'true':
        print("Training with data augmentation...")
        model8k.fit_generator(generator.flow(trainX, Y_train, batch_size=batch_size),
                              steps_per_epoch=len(trainX) // batch_size, epochs=nb_epoch,
                              callbacks=callbacks_8k,
                              validation_data=(testX, Y_test),
                              validation_steps=testX.shape[0] // batch_size, verbose=1)
    else:
        print("Training without data augmentation...")
        model8k.fit(trainX, Y_train, batch_size=batch_size, epochs=nb_epoch, callbacks=callbacks_8k,
                    validation_data=(testX, Y_test), verbose=2)
except KeyboardInterrupt:
    print("Training interrupted")
    sys.exit(1)


yPreds_8k = model8k.predict(testX)
yPred_8k = np.argmax(yPreds_8k, axis=1)

accuracy_8k = metrics.accuracy_score(yTrue, yPred_8k) * 100
error_8k = 100 - accuracy_8k
print("8K Accuracy : ", accuracy_8k)
print("8K Error : ", error_8k)
del model8k