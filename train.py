from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pickle
import time
import cv2
import os

from model import build_pretrained_model
import utils 

def create_dataset(train_path, test_path=None, valid_size=0.1, batch_size=32):
    X_train, X_val, y_train, y_val = utils.read_data(
        data_path=train_path,
        valid_size=valid_size)

    enc = LabelEncoder()
    enc.fit(y_train)
    print(enc.classes_)
    
    with open("models/label_enc.pkl", "wb") as f:
        pickle.dump(enc, f)

    y_train = enc.transform(y_train)
    y_train = to_categorical(y_train)

    train_gen = image.ImageDataGenerator(
        preprocessing_function=preprocess_input,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    train_gen.fit(X_train)

    y_val = enc.transform(y_val)
    y_val = to_categorical(y_val)

    valid_gen = image.ImageDataGenerator(
        preprocessing_function=preprocess_input)

    valid_gen.fit(X_val)

    print("train: {}".format(len(y_train)))
    print("valid: {}".format(len(y_val)))

    if test_path is not None:
        X_test, y_test = utils.read_data(
            data_path=test_path,
            valid_size=0)

        y_test = enc.transform(y_test)
        y_test = to_categorical(y_test)

        test_gen = image.ImageDataGenerator(
            preprocessing_function=preprocess_input)
        
        test_gen.fit(X_test)

        train_gen = train_gen.flow(X_train, y_train, batch_size)
        valid_gen = valid_gen.flow(X_val, y_val, batch_size)
        test_gen = test_gen.flow(X_test, y_test, batch_size)
        
        print("test: {}".format(len(y_test)))

        return train_gen, valid_gen, test_gen

    valid_gen = valid_gen.flow(X_val, y_val, batch_size)
    test_gen = test_gen.flow(X_test, y_test, batch_size) 

    return train_gen, valid_gen

def train(model, train_gen, valid_gen, test_gen, 
          train_steps, valid_steps, 
          optim=Adam(lr=1e-4), epochs=30):
    
    model.compile(optimizer=optim, 
                  loss='categorical_crossentropy', 
                  metrics=['acc'])
    
    model.fit_generator(
        generator=train_gen, 
        steps_per_epoch=train_steps,
        validation_data=valid_gen,
        validation_steps=valid_steps,
        epochs=epochs)

    test_loss, test_acc = model.evaluate_generator(test_gen)
    print('accuracy: {}'. format(test_acc))
    print('loss: {}'.format(test_loss))

    model_path = 'models/model_{}acc.h5'.format(int(test_acc*100))
    tf.keras.models.save_model(model, model_path)

    print("Successfully Saved {}".format(model_path))

    return model


if __name__ == "__main__":

    # HyperParameters
    learning_rate = 1e-4
    hidden_dim = 1024
    batch_size = 32
    n_classes = 3
    epochs = 10

    train_gen, valid_gen, test_gen = create_dataset(
        train_path="data/data3a/training/",
        test_path="data/data3a/validation/",
        valid_size=0.1,
        batch_size=batch_size)

    base_model = ResNet50(
        input_shape=(224,224,3), 
        weights='imagenet', 
        include_top=False)

    model =  build_pretrained_model(
        base_model=base_model,
        hidden_dim=hidden_dim,
        n_classes=n_classes)

    # train only the top layers 
    for layer in base_model.layers:
        layer.trainable = False
        if "5c_" in layer.name:
            layer.trainable = True

    train_steps = np.ceil(train_gen.n/batch_size)
    valid_steps = np.ceil(valid_gen.n/batch_size)

    train(model=model,
        train_gen=train_gen,
        valid_gen=valid_gen,
        test_gen=test_gen,
        train_steps=train_steps,
        valid_steps=valid_steps,
        optim=Adam(lr=learning_rate), 
        epochs=epochs)