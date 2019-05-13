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
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer
import matplotlib.pyplot as plt
from keras.losses import categorical_crossentropy as logloss
from keras.metrics import categorical_accuracy
import sys

temperature = 1.0

def pretrained_model(layer_name):
    model = MobileNet(input_shape=(128, 128, 3), alpha=1.0, depth_multiplier=1, dropout=1e-3, include_top=False, weights='imagenet', input_tensor=None, pooling=None, classes=8)
    intermodel = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    return intermodel

def getKerasModel():
    model = Sequential()
    #model.add(L.Conv2D(1024, kernel_size=(3, 3), strides=(1, 1), padding='same', input_shape=(8, 8, 512)))
    #model.add(L.Flatten())
    model.add(L.Dense(1000, activation='elu', input_dim=8*8*512))
    model.add(L.Dropout(0.5))
    model.add(L.Dense(128, activation='elu'))
    model.add(L.Dense(8, activation='softmax'))
    return model


def normalize(X):
    for i in range(X.shape[1]):
        maxelem = X[:, i].max()
        X[:, i] = (X[:, i]-maxelem)/maxelem

    return X

def create_data(X, Y):
    data = np.c_[X, Y]
    np.random.shuffle(data)
    train = data[:int(data.shape[0]*0.8), :]
    val = data[int(data.shape[0]*0.8):, :]
    Xtrain = train[:, :-8]
    Ytrain = train[:, -8:]
    Xval = val[:, :-8]
    Yval = val[:, -8:]
    return Xtrain, Ytrain, Xval, Yval


def shuffledata(X, Y):
    data = np.c_[X, Y]
    np.random.shuffle(data)
    X = data[:, :-8]
    Y = data[:, -8:]
    return X, Y


def shuffle_in_unison_scary(X, Y):
    rng_state = np.random.get_state()
    np.random.shuffle(X)
    np.random.set_state(rng_state)
    np.random.shuffle(Y)
    return X, Y


def trainRealTransfer2_withoutgen():
    batchsize = 64
    Xtrain, Ytrain, Xtest, Ytest= getsaveddata()
    intermodel = pretrained_model('conv_pw_6')
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
    x = L.Dense(128, activation='relu')(x)
    x = L.Dropout(0.5)(x)
    x = L.Dense(8)(x)
    x = L.Activation('softmax')(x)
    model = Model(inputs = intermodel.input, outputs = x)
    print(model.summary())
    filepath="models/fullmodel_transfer2.h5"
    model.compile(optimizer=Adam(lr=0.00005, decay=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callback_list = [checkpoint]
    history = model.fit(x=Xtrain, y=Ytrain, batch_size=64, epochs=35, verbose=1, validation_data=(Xtest, Ytest), callbacks=callback_list)
    print(model.evaluate(Xtest, Ytest))
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('models/fullmodel_transfer2_history/accuracy.jpg')
    plt.gcf().clf()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('models/fullmodel_transfer2_history/loss.jpg')

def checkKDcompare():
    batchsize = 64
    Xtrain, Ytrain, Xtest, Ytest= getsaveddata()
    intermodel = pretrained_model('conv_pw_4')
    x = intermodel.output
    x = L.Conv2D(filters=512, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='tanh')(x)
    x = L.Dropout(0.5)(x)
    # x = L.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='tanh')(x)
    # x = L.Dropout(0.5)(x)
    x = L.Conv2D(filters=512, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='tanh')(x)
    x = L.Dropout(0.5)(x)
    x = L.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = L.Flatten()(x)
    x = L.Dense(8)(x)
    x = L.Activation('softmax')(x)
    model = Model(inputs = intermodel.input, outputs = x)
    print(model.summary())
    filepath="models/test.h5"
    model.compile(optimizer=Adam(lr=0.00005, decay=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callback_list = [checkpoint]
    history = model.fit(x=Xtrain, y=Ytrain, batch_size=64, epochs=150, verbose=1, validation_data=(Xtest, Ytest), callbacks=callback_list)
    print(model.evaluate(Xtest, Ytest))
    # summarize history for accuracy



def custom_loss(y_true, y_pred):
    lamda=0.7
    y_true, logits = y_true[:, :8], y_true[:, 8:]
    y_pred, y_pred_soft = y_pred[:, :8], y_pred[:, 8:]
    return lamda*logloss(y_true, y_pred) + (1-lamda)*logloss(logits, y_pred_soft)
#     return logloss(logits, y_pred_soft)

def soft_logloss(y_true, y_pred):     
    logits = y_true[:, 8:]
    y_soft = L.Softmax(logits/temperature)
    y_pred_soft = y_pred[:, 8:]    
    return logloss(y_soft, y_pred_soft)

def categorical_crossentropy(y_true, y_pred):
    y_true = y_true[:, :8]
    y_pred = y_pred[:, :8]
    return logloss(y_true, y_pred)


def accuracy(y_true, y_pred):
    y_true = y_true[:, :8]
    y_pred = y_pred[:, :8]
    return categorical_accuracy(y_true, y_pred)

  
def create_KD_model():
    intermodel = pretrained_model('conv_pw_4')
    x = intermodel.output
    o1 = L.Conv2D(filters=512, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu', name='o1')(x)
    x = L.Dropout(0.5)(o1)
    # x = L.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='tanh')(x)
    # x = L.Dropout(0.5)(x)
    x = L.Conv2D(filters=512, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = L.Dropout(0.5)(x)
    o2 = L.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='o2')(x)
    x = L.Flatten()(o2)
    x = L.Dense(8)(x)
    # hardtargets = L.Activation('softmax')(x)
    # x_soft = L.Lambda(lambda x: x/7.0)(x)
    softtargets = L.Activation('softmax', name='final')(x)
    # output = L.concatenate([hardtargets, softtargets], name='final')
    # model = Model(inputs = intermodel.input, outputs = [intermodel.get_layer('conv_pw_2').output, o2, x])
    model = Model(inputs = intermodel.input, outputs=[o1, o2, softtargets])
    # losses = {'o1':'MSE', 'o2':'MSE', 'final':lambda y_true, y_pred: custom_loss(y_true, y_pred)}
    losses = {'o1':'MSE', 'o2':'MSE', 'final':'categorical_crossentropy'}
    model.compile(optimizer=Adam(lr=0.00004, decay=0.001), loss=losses, metrics=['accuracy'])
    # model.compile(optimizer=Adam(lr=0.00003, decay=0.001), loss=lambda y_true, y_pred: custom_loss(y_true, y_pred), metrics=[accuracy, categorical_crossentropy])
    print(model.summary())
    return model

def getrandomsampleKD(X, Y):
    rng_state = np.random.get_state()
    np.random.shuffle(X)
    np.random.set_state(rng_state)
    np.random.shuffle(Y)
    X_train = X[:int(0.8*3105), :, :, :]
    Y_train = Y[:int(0.8*3105), :]
    X_test = X[int(0.8*3105):, :, :, :]
    Y_test = Y[int(0.8*3105):, :]
    return X_train, Y_train, X_test, Y_test
  
def trainRealTransfer_small_KD(num_epochs):
    Xtrain, Ytrain, Xtest, Ytest, convtrain, convtest, conv4train, conv4test = getsavedKDdata()
    model = create_KD_model()
    filepath = 'models/KDmodel.h5'
    yt = {'o1':convtrain, 'o2':conv4train, 'final':Ytrain[:, :8]}
    ytt = {'o1':convtest, 'o2':conv4test, 'final':Ytest[:,:8]}
    checkpoint = ModelCheckpoint(filepath, monitor='val_final_loss', verbose=1, save_best_only=True, mode='min')
    # model.fit(x=X_train, y={'conv_pw_2':Y1_train, 'Y2':Y2_train, 'Y3':Y3_train}, batch_size=64, epochs=40, verbose=1, validation_data=(X_test, {'conv_pw_2':Y1_test, 'Y2':Y2_test, 'Y3':Y3_test}), callbacks=callback_list)
    history=model.fit(x=Xtrain, y=yt, batch_size=64, epochs=num_epochs, verbose=1, validation_data=(Xtest, ytt), callbacks=[checkpoint])
    model.save(filepath)
    plt.gcf().clf()
    plt.plot(history.history['final_acc'])
    plt.plot(history.history['val_final_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('models/KDmodelhistory/accuracy.jpg')
    plt.gcf().clf()
    # summarize history for loss
    plt.plot(history.history['final_loss'])
    plt.plot(history.history['val_final_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('models/KDmodelhistory/loss.jpg')
    output = model.predict(Xtest)[2]
    predictions = output[:, :8]
    predictions = np.array(predictions)
    for i in range(predictions.shape[0]):
        amax = predictions[i, :].argmax()
        predictions[i, :] = 0
        predictions[i, amax] = 1
    total = predictions.shape[0]
    correct=0
    for i in range(total):
        if np.array_equal(predictions[i, :].astype(int),Ytest[i, :8]):
            correct+=1

    print(1.0*correct/total)
    
# AMR2018190101542199343

def checktest():
    Xtrain, Ytrain, Xtest, Ytest = getsaveddata()
    model = load_model('mobilemodel.h5')
    output = model.predict(Xtest)
    predictions = output
    predictions = np.array(predictions)
    for i in range(predictions.shape[0]):
        amax = predictions[i, :].argmax()
        predictions[i, :] = 0
        predictions[i, amax] = 1
    total = predictions.shape[0]
    correct=0
    for i in range(total):
        if np.array_equal(predictions[i, :].astype(int),Ytest[i, :]):
            correct+=1

    print(1.0*correct/total)

def getsaveddata():
    Xtrain = np.load('KDdata/Xtrain.npy')
    Ytrain = np.load('KDdata/Ytrain.npy')
    Xtest = np.load('KDdata/Xtest.npy')
    Ytest = np.load('KDdata/Ytest.npy')
    return Xtrain, Ytrain, Xtest, Ytest

# custom dataset generation 
def getdata():
    mydir = 'finaldata/extracted/'
    X = np.zeros((3105, 128, 128, 3))
    Y = np.zeros((3105,), dtype=int)
    counter = 0
    for direcs in os.listdir(mydir):
        subpath = os.path.join(mydir, direcs)
        print(direcs)
        for imagename in os.listdir(subpath):
            imagepath = os.path.join(subpath, imagename)
            img = cv2.imread(imagepath)
            img = cv2.resize(img, (128, 128))
            X[counter, :, :, :] = img
            Y[counter] = int(direcs)
            counter+=1
    print('done')
    X /= 255
    X -= 0.5
    X *= 2
    enc=LabelBinarizer()
    Y=enc.fit_transform(Y.reshape(Y.shape[0], 1))
    X, Y = shuffle_in_unison_scary(X, Y)
    Xtrain, Ytrain, Xtest, Ytest = getrandomsampleKD(X, Y)
    np.save('KDdata/Xtrain.npy', Xtrain)
    np.save('KDdata/Xtest.npy', Xtest)
    np.save('KDdata/Ytrain.npy', Ytrain)
    np.save('KDdata/Ytest.npy', Ytest)

def getsavedKDdata():
    Xtrain = np.load('KDdata/Xtrain.npy')
    Ytrain = np.load('KDdata/Y3train.npy')
    Xtest = np.load('KDdata/Xtest.npy')
    Ytest = np.load('KDdata/Y3test.npy')
    convtrain = np.load('KDdata/convtrain.npy')
    convtest = np.load('KDdata/convtest.npy')
    conv4train = np.load('KDdata/conv4train.npy')
    conv4test = np.load('KDdata/conv4test.npy')
    return Xtrain, Ytrain, Xtest, Ytest, convtrain, convtest, conv4train, conv4test

from keras.utils.generic_utils import CustomObjectScope

def softmax(x):
    ex = np.exp(x)
    sum_ex = np.sum(np.exp(x))
    return ex/sum_ex

def gethighestaccuracymodeloutput():
    # model = load_model('models/fullmodel_transfer2.h5')
    # with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
    model = load_model('models/fullmodel_transfer2.h5')
    # intermodel1 = Model(inputs=model.input, outputs=model.get_layer('conv_pw_2').output)
    # intermodel2 = Model(inputs=model.input, outputs=model.get_layer('conv_pw_6').output)
    intermodel3 = Model(inputs=model.input, outputs=model.get_layer('dense_4').output)
    intermodel2 = Model(inputs=model.input, outputs=model.get_layer('conv2d_1').output)
    intermodel1 = Model(inputs=model.input, outputs=model.get_layer('conv2d_4').output)
    print(model.summary())
    Xtrain , Ytrain, Xtest, Ytest = getsaveddata()
    Y3 = intermodel3.predict(Xtrain)
    Y3_b = softmax(Y3/temperature)
    Y3train = np.c_[Ytrain, Y3_b]
    Y3t = intermodel3.predict(Xtest)
    Y3t_b = softmax(Y3t/temperature)
    Y3test = np.c_[Ytest, Y3t_b]
    convtrain = intermodel2.predict(Xtrain)
    convtest = intermodel2.predict(Xtest)
    conv4train = intermodel1.predict(Xtrain)
    conv4test = intermodel1.predict(Xtest)
    np.save('KDdata/convtrain.npy', convtrain)
    np.save('KDdata/convtest.npy', convtest)
    np.save('KDdata/conv4train.npy', conv4train)
    np.save('KDdata/conv4test.npy', conv4test)
    np.save('KDdata/Y3train.npy', Y3train)
    np.save('KDdata/Y3test.npy', Y3test)

def inference_KD():
    model = load_model('models/fullmodel_transfer2.h5')
    ans = {
        '0': 0,
        '1': 45,
        '2': 90,
        '3': 135,
        '4': 180,
        '5': 225,
        '6': 270,
        '7': 315,
    }
    correct=0
    while True:
        imgp = input('imgpath: ')
        img = cv2.imread(imgp)
        img = cv2.resize(img, (128, 128)).astype(np.float32)
        # cv2.imshow('win', img)
        # cv2.waitKey(0)
        img -= 128
        img = img/128
        
        output = model.predict(np.reshape(img, (1, 128, 128, 3)))
        prediction = np.array(output).argmax()
        print(ans[str(prediction)])

# checktest()
# inference_KD()
# gethighestaccuracymodeloutput()
# trainRealTransfer2_withoutgen()
# getdata()
# create_KD_model()
# trainRealTransfer_small_KD(80)
# checkKDcompare()
# trainRealTransfer_small_KD()

def main():
    choice = sys.argv[1]
    if choice == 'teacher':
        trainRealTransfer2_withoutgen()
    if choice == 'student':
        gethighestaccuracymodeloutput()
        trainRealTransfer_small_KD(130)
    if choice == 'infer':
        inference_KD()

main()