#Author:Wallace He 

import matplotlib
#this comment is used for letting matplotlib showing picture in AWS non-gui enviroment
#if you used the gui-virtual machine environment, delete that one.
matplotlib.use('Agg')
import numpy as np
import cv2
#school's mac is hard to install cv2, you may think abput using PIL
import matplotlib.pyplot as plt
import pickle
import h5py
import glob
import time
from random import shuffle
from collections import Counter

from sklearn.model_selection import train_test_split

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.models import model_from_json
from keras.models import load_model


#input fo 
map_characters ={0:'CarolinaChickadee' , 1:'EmptyFeeder',2:'EuropeanStarling',3:'GrayCatbird',4:'NorthernCardinal',5:'RedBelliedWoodpecker',6:'TuftedTitmouse',7:'YellowRumpedWarbler'}

pic_size = 140
batch_size = 32
epochs = 40
num_classes = len(map_characters)
test_size = 0.15



def load_pictures(BGR):
    """
    Load pictures from folders for characters from the map_characters dict and create a numpy dataset and 
    a numpy labels set. Pictures are re-sized into picture_size square.
    :param BGR: boolean to use true color for the picture (RGB instead of BGR for plt)
    :return: dataset, labels set
    """
    pics = []
    labels = []
    for k, char in map_characters.items():
        pictures = [k for k in glob.glob('./data3/%s/*' % char)]
        #print(pictures)
        nb_pic = len(pictures)
        print(nb_pic)
        for pic in np.random.choice(pictures, nb_pic):
        	#we dont need to use shuffle
            a = cv2.imread(pic)
            if BGR:
                a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
            a = cv2.resize(a, (pic_size,pic_size))
            pics.append(a)
            labels.append(k)
    return np.array(pics), np.array(labels) 

def get_dataset(save=True, load=False, BGR=False):
    """
    Create the actual dataset split into train and test, pictures content is as float32 and
    normalized (/255.) from cs231. 
    :param save: saving or not the created dataset
    :param load: loading or not the dataset
    :param BGR: boolean to use true color for the picture (RGB instead of BGR for plt)
    :return: X_train, X_test, y_train, y_test (numpy arrays)
    """
    if load:
        h5f = h5py.File('dataset.h5','r')
        X_train = h5f['X_train'][:]
        X_test = h5f['X_test'][:]
        h5f.close()    

        h5f = h5py.File('labels.h5','r')
        y_train = h5f['y_train'][:]
        y_test = h5f['y_test'][:]
        h5f.close()    
    else:
        X, y = load_pictures(BGR)
        y = keras.utils.to_categorical(y, num_classes)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        #random splict the train/test set  
        if save:
            h5f = h5py.File('dataset.h5', 'w')
            h5f.create_dataset('X_train', data=X_train)
            h5f.create_dataset('X_test', data=X_test)
            h5f.close()

            h5f = h5py.File('labels.h5', 'w')
            h5f.create_dataset('y_train', data=y_train)
            h5f.create_dataset('y_test', data=y_test)
            h5f.close()
            
    X_train = X_train.astype('float32') / 255.
    X_test = X_test.astype('float32') / 255.
    print("Train", X_train.shape, y_train.shape)
    print("Test", X_test.shape, y_test.shape)
    if not load:
    	#figure of the class training and testing numbers
        dist = {k:tuple(d[k] for d in [dict(Counter(np.where(y_train==1)[1])), dict(Counter(np.where(y_test==1)[1]))]) 
                for k in range(num_classes)}
        print('\n'.join(["%s : %d train pictures & %d test pictures" % (map_characters[k], v[0], v[1]) 
            for k,v in sorted(dist.items(), key=lambda x:x[1][0], reverse=True)]))
    return X_train, X_test, y_train, y_test


def create_four_conv_layer(input_shape):
    """
    CNN Keras model with 4 convolution layers.
    :param input_shape: input shape, generally X_train.shape[1:]
    :return: Keras model, RMS prop optimizer
    """
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    return model, opt

def create_six_conv_layer(input_shape):
    """
    CNN Keras model with 6 convolutions.
    :param input_shape: input shape, generally X_train.shape[1:]
    :return: Keras model, RMS prop optimizer
    """
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(256, (3, 3), padding='same')) 
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    return model, opt


def lr_schedule(epoch):
	#used for the learning rate decay. 
    lr = 0.01
    return lr*(0.1**int(epoch/10))

def load_model_from_checkpoint(weights_path, six_conv=False, input_shape=(pic_size,pic_size,3)):
	
    if six_conv:
        model, opt = create_model_six_conv(input_shape)
    else:
        model, opt = create_model_four_conv(input_shape)
    model.load_weights(weights_path)
    model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
    return model
    

def VGG16(input_shape):

    """CNN Keras model with VGG16 network.
    :param input_shape: input shape, generally X_train.shape[1:]
    :return: Keras model, RMS prop optimizer"""
    

    model = VGG16(include_top=True, weights='None', input_tensor=None, input_shape=input_shape, pooling=None, classes=num_classes)
    opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    return model, opt


def VGG19(input_shape):

    """CNN Keras model with VGG19 network.
    :param input_shape: input shape, generally X_train.shape[1:]
    :return: Keras model, RMS prop optimizer"""
    

    model = VGG19(include_top=True, weights='None', input_tensor=None, input_shape=input_shape, pooling=None, classes=num_classes)
    opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    return model, opt





def training(model, X_train, X_test, y_train, y_test, data_augmentation=True):
    """
    Training.
    :param model: Keras sequential model
    :param data_augmentation: boolean for using the pre image-processing from keras or not(default:True)
    :param callback: boolean for saving model checkpoints and get the best saved mode
    :return: model and epochs history (acc, loss, val_acc, val_loss for every epochs)
    """
    if data_augmentation:
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images
        # Compute quantities required for feature-wise normalization
        # (std, mean, and principal components if ZCA whitening is applied).


        datagen.fit(X_train)
        filepath="weight test2 for 6 convolutionslayers-{epoch:03d}-{loss:.4f}-{val_acc:.4f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
        callbacks_list = [LearningRateScheduler(lr_schedule) ,checkpoint]
        history = model.fit_generator(datagen.flow(X_train, y_train,
                                    batch_size=batch_size),
                                    steps_per_epoch=X_train.shape[0] // batch_size,
                                    epochs=epochs,
                                    validation_data=(X_test, y_test),
                                    callbacks=callbacks_list)


                        
    else:
        history = model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(X_test, y_test),
          shuffle=True)
    return model, history


'''

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_dataset(load=False)
    model, opt = create_six_conv_layer(X_train.shape[1:])
    model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
    model, hist = training(model, X_train, X_test, y_train, y_test, data_augmentation=True)
    
    print(hist.history.keys())
    train_loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    train_acc = hist.history['acc']
    val_acc = hist.history['val_acc']
    xc = range(epochs)
    
    plt.figure(1,figsize=(7,5))
    plt.plot(xc,train_loss)
    plt.plot(xc,val_loss)
    plt.xlabel('num of Epochs')
    plt.ylabel('loss')
    plt.title('train_loss vs val_loss')
    plt.grid(True)
    plt.legend(['train','val'])
    #print plt.style.available # use bmh, classic,ggplot for big pictures
    plt.style.use(['classic'])

    plt.figure(2,figsize=(7,5))
    plt.plot(xc,train_acc)
    plt.plot(xc,val_acc)
    plt.xlabel('num of Epochs')
    plt.ylabel('accuracy')
    plt.title('train_acc vs val_acc')
    plt.grid(True)
    plt.legend(['train','val'],loc=4)
    #print plt.style.available # use bmh, classic,ggplot for big pictures
    plt.style.use(['classic'])
    plt.savefig('result.jpg')



    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test Loss:', score[0])
    print('Test accuracy:', score[1])

    test_image = X_test[0:1]
    print (test_image.shape)

    print(model.predict(test_image))
    print(model.predict_classes(test_image))
    print(y_test[0:1])

    model_json = model.to_json()
    with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")

                
	
'''






