import numpy as np
import cv2
import os
import sys
import keras
from keras.applications.mobilenet import MobileNet
from keras.models import Model, Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import keras.layers as L
from keras.optimizers import SGD, Adam
from os.path import join
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer
import matplotlib.pyplot as plt
import sys

temperature = 10

def pretrained_small():
    model = MobileNet(input_shape=(128, 128, 3), alpha=1.0, depth_multiplier=1, dropout=1e-3, include_top=False, weights='imagenet', input_tensor=None, pooling=None, classes=8)
    layer_name = 'conv_pw_6'
    intermodel = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    return intermodel

def shuffle_in_unison_scary(X, Y1, Y2):
    rng_state = np.random.get_state()
    np.random.shuffle(X)
    np.random.set_state(rng_state)
    np.random.shuffle(Y1)
    np.random.set_state(rng_state)
    np.random.shuffle(Y2)
    return X, Y1, Y2

def getdata():
    num_images=3105
    X = np.zeros((num_images, 128, 128, 3))
    Y1 = np.zeros((num_images,), dtype=int)
    Y2 = np.zeros((num_images,), dtype=int)
    mydir1 = 'finaldata/paths/'
    mydir2 = 'finaldata/'
    counter=0
    summ=0
    for labelfile in os.listdir(mydir1):
        print(labelfile)
        fullpath = join(mydir1, labelfile)
        data = pd.read_csv(fullpath)
        length = len(data)
        for i in range(length):
            imgname = str(data["filepath"][i])
            if imgname == 'nan': continue
            imgname = imgname[2:]
            direction = data['direction'][i]
            distracted = data['distracted'][i]
            imagepath=join(mydir2, imgname)
            img = cv2.imread(imagepath)
            img = cv2.resize(img, (128, 128))
            X[counter, :, :, :] = img[:, :, :]
            Y1[counter] = int(direction)
            Y2[counter] = int(distracted)
            counter+=1
    print("done")
    print(counter)
    X -= 128
    X /= 128
    enc=LabelBinarizer()
    Y1=enc.fit_transform(Y1.reshape(Y1.shape[0], 1))
    Y2=enc.fit_transform(Y2.reshape(Y2.shape[0], 1))
    np.save('distractiondata/X.npy', X)
    np.save('distractiondata/Y1.npy', Y1)
    np.save('distractiondata/Y2.npy', Y2)
    return

def getsaveddata():
    X = np.load('distractiondata/X.npy')
    Y1 = np.load('distractiondata/Y1.npy')
    Y2 = np.load('distractiondata/Y2.npy')
    return X, Y1, Y2

def trainRealTransfer2_withoutgen():
    batchsize = 64
    X, Y1, Y2= getsaveddata()
    X, Y1, Y2= shuffle_in_unison_scary(X, Y1, Y2)
    intermodel = pretrained_small()
    print(Y2.shape)
    x = intermodel.output
    x = L.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = L.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = L.Conv2D(filters=512, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = L.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = L.Flatten()(x)
    x = L.Dense(1024, activation='relu')(x)
    x = L.Dropout(0.5)(x)
    x = L.Dense(512, activation='relu')(x)
    x = L.Dropout(0.5)(x)
    x1 = L.Dense(128, activation='relu')(x)
    x1 = L.Dropout(0.5)(x1)
    x1 = L.Dense(8)(x1)
    x1 = L.Activation('softmax', name='dir_out')(x1)
    x2 = L.Dense(128, activation='elu')(x)
    x2 = L.Dropout(0.5)(x2)
    x2 = L.Dense(1)(x2)
    x2 = L.Activation('sigmoid', name='dis_out')(x2)
    model = Model(inputs = intermodel.input, outputs = [x1, x2])
    filepath="models/distraction.h5"
    loss = {
        'dir_out':'categorical_crossentropy',
        'dis_out':'binary_crossentropy'
    }
    model.compile(optimizer=Adam(lr=0.00005, decay=0.001), loss=loss, metrics=['accuracy'])
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callback_list = [checkpoint]
    history = model.fit(x=X, y={'dir_out':Y1, 'dis_out':Y2}, batch_size=64, epochs=1, verbose=1, validation_split=0.2, callbacks=callback_list)
    # summarize history for accuracy


def inference_KD():
    model = load_model('models/distraction.h5')
    path = 'test/'
    correct=0
    l = len(os.listdir(path))
    while True:
        imgp = input('imgpath: ')
        img = cv2.imread(imgp)
        img = cv2.resize(img, (128, 128)).astype(np.float32)
        # cv2.imshow('win', img)
        # cv2.waitKey(0)
        img -= 128
        img = img/128
        
        output = model.predict(np.reshape(img, (1, 128, 128, 3)))
        prediction = np.array(output[0]).argmax()
        prediction2 = output[1]
        print('direction:', prediction, 'distraction score:', prediction2[0][0])
        

def main():
    choice = sys.argv[1]
    if choice == 'train':
        trainRealTransfer2_withoutgen()
    if choice == 'infer':
        inference_KD()

main()