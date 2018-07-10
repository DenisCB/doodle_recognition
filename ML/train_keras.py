import json
import numpy as np
import pandas as pd
import os
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import PIL
from PIL import ImageDraw


os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils.np_utils import to_categorical
import keras.backend as K


def im2arr(drawing):
    scale = np.random.beta(6, 2)

    # Original images are 255x255, add extra 5 to each edge.
    im = PIL.Image.new(mode='L', size=(260, 260))
    draw = PIL.ImageDraw.Draw(im)

    # Shift the strokes from edges by 5 pixels, convert them to valid format.
    for stroke in drawing:
        stroke_shifted = list(map(lambda x: tuple([(i+2.5)*scale for i in x]),
                                  tuple(zip(stroke[0], stroke[1]))))
        draw.line(stroke_shifted, fill=255, width=4)
    # Find the bounding box.
    bbox = PIL.Image.eval(im, lambda x: x).getbbox()
    width = bbox[2] - bbox[0]  # right minus left
    height = bbox[3] - bbox[1]  # bottom minus top
    # Center after croping.
    diff = width - height
    if diff >= 0:
        bbox = (bbox[0], bbox[1]-diff/2, bbox[2], bbox[3]+diff/2)
    else:
        bbox = (bbox[0]+diff/2, bbox[1], bbox[2]-diff/2, bbox[3])
    # Add borders.
    bbox = (bbox[0]-border_px, bbox[1]-border_px, bbox[2]+border_px, bbox[3]+border_px)

    # Crop and resize.
    im = im.crop(bbox)
    im = im.resize((px, px), resample=3)

    # Clip max values to make lines less blury.
    im = np.array(im).astype(float)
    im /= im.max()/2

    return im.clip(0, 1)


def batch_generator(X, y, weights, BATCH_SIZE, EPOCH_PARTION, mean_global):
    """
    Batch generator for nnet training
    input:
        X - train dataset,  numpy array or csr matrix
        y - target, numpy array
        BATCH_SIZE - int, number of objects in batch. If X is csr matrix, it will be transformed 
        to dense array so batch size must be small enough for this array to fit in memory
        EPOCH_PARTION - float, share of objects that will be used for training in epoch

    Important: in this implementation each batch is constructed from random objects from train.
    """

    batch_number = 0
    batches_per_epoch = int(X.shape[0]/BATCH_SIZE*EPOCH_PARTION)
    while True:
        batch_indexes = np.random.choice(X.shape[0], BATCH_SIZE)
        X_batch = np.array([im2arr(drawing) for drawing in X[batch_indexes]]) - mean_global
        X_batch = X_batch.reshape(-1, px, px, 1)
        y_batch = to_categorical(y[batch_indexes], num_classes=num_classes)
        w_batch = weights[batch_indexes]
        batch_number += 1

        yield X_batch, y_batch, w_batch
        if batch_number == batches_per_epoch-1:
            batch_number = 0


def batch_generator_p(X, BATCH_SIZE):
    """
    Batch generator for nnet predictions
    input:
        X - train dataset,  numpy array or csr matrix
        BATCH_SIZE - number of objects in batch. If X is csr matrix, it will be transformed 
        to dense array so batch size must be small enough for this array to fit in memory        
    """

    number_of_batches = X.shape[0] / np.ceil(X.shape[0]/BATCH_SIZE)
    batch_number = 0
    sample_index = np.arange(X.shape[0])
    while True:
        batch_indexes = sample_index[BATCH_SIZE*batch_number : BATCH_SIZE*(batch_number+1)]
        X_batch = X[batch_indexes]
        X_batch = np.array([im2arr(drawing) for drawing in X_batch])
        X_batch = X_batch.reshape(-1, px, px, 1)
        batch_number += 1
        yield (X_batch)
        if batch_number == number_of_batches:
            batch_number = 0


px = 96
border_px = 2
result_path = '../model/'
df = pd.read_csv('data/df.csv')
train = np.load('data/strokes_raw.npy')
print(df.shape[0])


# We'll need to substract mean global from each pixel, but we can't calculate it,
# so we will be updating mean global during training. For now initialize it with 100 random batches.
batch_counter = 0
mean_global = np.zeros((px, px))
for i in range(100):
    batch = np.random.choice(train, 1000)
    batch = np.array([im2arr(drawing) for drawing in batch])

    mean_global = (mean_global * batch_counter + batch.mean(axis=0)) / (batch_counter+1)
    batch_counter += 1
mean_global = mean_global.mean()


le = LabelEncoder()
ytrain = le.fit_transform(df.target.values)
(train, valid,
 ytrain, yvalid,
 df, dfvalid) = train_test_split(train, ytrain, df, test_size=0.05, random_state=0)
train_weights = np.array([1 if recognized==1 else 0.25 for recognized in df.recognized])
valid_weights = np.array([1 if recognized==1 else 0.25 for recognized in dfvalid.recognized])
num_classes = ytrain.max()+1

np.save(result_path+'processing_params.npy', np.array([px, border_px]))
np.save(result_path+'mean_global.npy', np.array([mean_global]))
np.save(result_path+'classes.npy', le.classes_)




model = Sequential()
model.add(Conv2D(filters = 128, kernel_size = (5, 5), activation='relu',
                 input_shape = (px, px, 1)))
model.add(BatchNormalization())
model.add(MaxPool2D())
model.add(Dropout(0.05))

model.add(Conv2D(filters = 128, kernel_size = (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D())
model.add(Dropout(0.05))

model.add(Conv2D(filters = 256, kernel_size = (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D())
model.add(Dropout(0.05))

model.add(Conv2D(filters = 128, kernel_size = (1, 1), activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.05))

model.add(Conv2D(filters = 256, kernel_size = (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D())
model.add(Dropout(0.05))

model.add(Conv2D(filters = 256, kernel_size = (3, 3), activation='relu'))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.05))

model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', 
              optimizer = Adam(lr=1e-4),
              metrics=["accuracy"])
model.summary()

# model.load_weights(result_path+'nnet_96_aug_v1.h5')

print('started training')
EPOCH_PARTION = 0.1
VALID_EPOCH_PARTION = 0.25
BATCH_SIZE = 256

early_stopper = EarlyStopping(monitor='val_acc', patience=2, verbose=0, mode='auto')
checkpoint = ModelCheckpoint(filepath=result_path+'nnet_96_aug_v2.h5', monitor='val_acc', save_best_only=True)

model.fit_generator(
    generator=batch_generator(train, ytrain, train_weights, BATCH_SIZE, EPOCH_PARTION, mean_global),
    samples_per_epoch=np.ceil(train.shape[0]/BATCH_SIZE*EPOCH_PARTION),
    
    validation_data=batch_generator(valid, yvalid, valid_weights, BATCH_SIZE, EPOCH_PARTION, mean_global),
    validation_steps = np.ceil(valid.shape[0]/BATCH_SIZE*VALID_EPOCH_PARTION),
    
    verbose=1, nb_epoch=200,
    callbacks=[early_stopper, checkpoint],

#     use_multiprocessing=True,
#     workers=5,
    max_queue_size=10
)



K.set_value(model.optimizer.lr, 1e-5)

EPOCH_PARTION = 0.1
VALID_EPOCH_PARTION = 0.25
BATCH_SIZE = 256

checkpoint = ModelCheckpoint(filepath=result_path+'nnet_96_aug_v2.h5', monitor='val_acc', save_best_only=True)

model.fit_generator(
    generator=batch_generator(train, ytrain, train_weights, BATCH_SIZE, EPOCH_PARTION, mean_global),
    samples_per_epoch=np.ceil(train.shape[0]/BATCH_SIZE*EPOCH_PARTION),
    
    validation_data=batch_generator(valid, yvalid, valid_weights, BATCH_SIZE, EPOCH_PARTION, mean_global),
    validation_steps = np.ceil(valid.shape[0]/BATCH_SIZE*VALID_EPOCH_PARTION),
    
    verbose=1, nb_epoch=200,
    callbacks=[early_stopper, checkpoint],

#     use_multiprocessing=True,
#     workers=5,
    max_queue_size=10
)
