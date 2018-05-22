from keras.layers import Input, Dropout, Conv2D, Flatten, MaxPooling2D, Dense, BatchNormalization, Activation, SeparableConv2D, MaxPool2D, Reshape,Activation,Flatten, Dense, Permute
from keras.models import Sequential, Model
from keras.regularizers import l2
from keras.layers.advanced_activations import PReLU
import keras.layers as layers

def simple_CNN(input_shape):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(1, activation='linear'))
    return model

def simpler_CNN(input_shape):
    model = Sequential()
    model.add(Conv2D(8, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(1, activation='linear'))
    return model

def simpler_nodropout_CNN(input_shape, output_size):
    model = Sequential()
    model.add(Conv2D(8, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(output_size))
    return model

def simple_nodropout_CNN(input_shape, output_size):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(output_size))
    return model

def Kao_Onet(input_shape, output_size):
    input = Input(shape = input_shape)
    x = Conv2D(32, (3, 3), strides=1, padding='valid', name='conv1')(input)
    # x = PReLU(name='prelu1')(x)
    x = PReLU(shared_axes=[1,2],name='prelu1')(x)
    x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)
    x = Conv2D(64, (3, 3), strides=1, padding='valid', name='conv2')(x)
    x = PReLU(shared_axes=[1,2],name='prelu2')(x)
    # x = PReLU(name='prelu2')(x)
    x = MaxPool2D(pool_size=3, strides=2)(x)
    x = Conv2D(64, (3, 3), strides=1, padding='valid', name='conv3')(x)
    x = PReLU(shared_axes=[1,2],name='prelu3')(x)
    # x = PReLU(name='prelu3')(x)
    x = MaxPool2D(pool_size=2)(x)
    x = Conv2D(128, (2, 2), strides=1, padding='valid', name='conv4')(x)
    x = PReLU(shared_axes=[1,2],name='prelu4')(x)
    x = Flatten()(x)
    x = Dense(256, name='conv5') (x)
    x = PReLU(name='prelu5')(x)
    out = Dense(output_size)(x)
    model = Model(inputs=input, outputs=out)
    return model


def mini_inception(input_shape, output_size):
    input_img = Input(input_shape)
    tower_1 = Conv2D(64, (1,1), padding='same', activation='relu')(input_img)
    tower_1 = Conv2D(64, (3,3), padding='same', activation='relu')(tower_1)
    tower_2 = Conv2D(64, (1,1), padding='same', activation='relu')(input_img)
    tower_2 = Conv2D(64, (5,5), padding='same', activation='relu')(tower_2)
    tower_3 = MaxPooling2D((3,3), strides=(1,1), padding='same')(input_img)
    tower_3 = Conv2D(64, (1,1), padding='same', activation='relu')(tower_3)
    output = layers.concatenate([tower_1, tower_2, tower_3], axis = 3)
    output = Flatten()(output)
    out = Dense(output_size, activation='linear')(output)
    model = Model(inputs = input_img, outputs = out)
    return model
