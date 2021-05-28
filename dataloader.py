# -*- coding: utf-8 -*-
"""

"""

# pip install tensorflow-datasets
import os
import sys
import cv2
import keras
import numpy as np
from skimage import io
from keras.datasets import cifar10, mnist, cifar100
from sklearn.model_selection import StratifiedShuffleSplit
import tensorflow as tf

import config

def load_cifar():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train/255.0, x_test/255.0     # divide by 255 normalization

    # # # divide by mean normalization
    # x_train_mean = np.mean(x_train, axis=0)
    # x_train -= x_train_mean
    # x_test -= x_train_mean

    # sss = StratifiedShuffleSplit(n_splits=1, test_size=0.0002, random_state=config.random_state)
    # sss.get_n_splits(x_train, y_train)

    # for train_index, test_index in sss.split(x_train, y_train):
    #   x_train, _ = x_train[train_index], x_train[test_index]
    #   y_train, _ = y_train[train_index], y_train[test_index]
    
    #if config.loss == 'mean_squared_error':
        # Convert class vectors to binary class matrices. This is required only for pretrained models.
    y_train = keras.utils.to_categorical(y_train, config.num_classes)
    y_test = keras.utils.to_categorical(y_test, config.num_classes)

    print(x_train.shape, x_test.shape)
    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)

def resize_images(images):
    new_images = []
    for image in images:
        image = cv2.resize(image, (config.image_size, config.image_size), interpolation=cv2.INTER_CUBIC)
        new_images.append(image)
    return np.array(new_images)
    
def load_cifar100():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data(label_mode="fine")
    x_train, x_test = resize_images(x_train), resize_images(x_test)
    x_train, x_test = x_train/255.0, x_test/255.0

    y_train = keras.utils.to_categorical(y_train, config.num_classes)
    y_test = keras.utils.to_categorical(y_test, config.num_classes)

    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)

def stack_channels(image_list):
    images = []
    for image in image_list:
        image = cv2.resize(image, (32,32), interpolation=cv2.INTER_CUBIC)
        image = np.stack([image, image, image], axis=2)
        images.append(image)
    return np.array(images)

def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = np.expand_dims(x_train, axis=3), np.expand_dims(x_test, axis=3)
    #x_train, x_test = stack_channels(x_train), stack_channels(x_test)
    x_train, x_test = x_train/255.0, x_test/255.0
    #print("Maximum value", np.max(x_train))

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.001, random_state=config.random_state)
    sss.get_n_splits(x_train, y_train)

    for train_index, test_index in sss.split(x_train, y_train):
      x_train, _ = x_train[train_index], x_train[test_index]
      y_train, _ = y_train[train_index], y_train[test_index]

    #if config.loss == 'mean_squared_error':
    #Convert class vectors to binary class matrices. This is required only for pretrained models.
    y_train = keras.utils.to_categorical(y_train, config.num_classes)
    y_test = keras.utils.to_categorical(y_test, config.num_classes)

    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)

def load_colored_mnist(color_var = 0.020):
    
    data_dic = np.load(os.path.join(config.colored_mnist_path,'mnist_10color_jitter_var_%.03f.npy'%color_var),encoding='latin1',allow_pickle=True).item()
    x_train = data_dic['train_image']
    y_train = data_dic['train_label']
    x_test = data_dic['test_image']
    y_test = data_dic['test_label']

    x_train, x_test = x_train/255.0, x_test/255.0     # divide by 255 normalization
    y_train = keras.utils.to_categorical(y_train, config.num_classes)
    y_test = keras.utils.to_categorical(y_test, config.num_classes)

    print(x_train.shape, x_test.shape)
    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)

def load_corrupted_testset(dataset_name, corrution_type):
    if dataset_name == 'mnist':
        # print("loading corrupted data from:", dataset_name, "corruption name:", corrution_type)
        npys = os.listdir(os.path.join(config.mnsitc_path, corrution_type))
        # print("npys:", npys)
        x_test = np.load(os.path.join(config.mnsitc_path, corrution_type, npys[npys.index('test_images.npy')]))
        y_test = np.load(os.path.join(config.mnsitc_path, corrution_type, npys[npys.index('test_labels.npy')]))
        x_test = x_test/255.0

    elif dataset_name == 'cifar':
        x_test = np.load(os.path.join(config.cifarc_path, corrution_type+".npy"))
        y_test = np.load(os.path.join(config.cifarc_path, "labels.npy"))
        x_test = x_test/255.0
        
    elif dataset_name == 'cifar100':
        x_test = np.load(os.path.join(config.cifarc100_path, corrution_type+".npy"))
        y_test = np.load(os.path.join(config.cifarc100_path, "labels.npy"))
        x_test = resize_images(x_test)
        x_test = x_test/255.0
    return x_test, y_test

def load_occluded_testset(dataset_name, occlusion_color, occlusion_range, set_='test'):
    '''
    Occlusion ranges from 20% to 80% with a step of 20%
    occlusion range:
    20.0: 20% to 40%
    40.0: 40% to 60%
    60.0: 60% to 80%

    occlusion_color(str) can be black or white: 0 for black and 1 for white

    set_: trainset or testset (Presently, the dataset is prepared only for testset)
    '''
    path = os.path.join('../occluded_dataset', dataset_name, occlusion_color, str(occlusion_range), set_)
    images, labels = [], []

    for label in os.listdir(path):
        # print(label)
        for image in os.listdir(os.path.join(path, label)):
            # print(image)
            image = io.imread(os.path.join(path, label, image))
            images.append(image)
            labels.append(label)

    # Currently the dataset loaded is unshuffled because we are going to use this testset only.
    images, labels = np.array(images), np.array(labels)
    if dataset_name == 'mnist':
        images = np.expand_dims(images, axis=3)
    labels = keras.utils.to_categorical(labels, config.num_classes)
    print(images.shape, labels.shape)
    return images, labels

def load_mnist_new(pert_name):
    '''
    Dataloader to load mnist-new dataset for a specified perturbation
    '''
    path = os.path.join('../data/new_mnist', pert_name)
    x_train, y_train, x_test, y_test = [], [], [], []

    for amat in os.listdir(path):
        if 'test.amat' in amat:
            test_data = np.loadtxt(os.path.join(path, amat))
            for row in test_data:
                flat_image = row[:-1]
                image = np.reshape(flat_image, (28, 28))
                label = row[-1]
                x_test.append(image)
                y_test.append(label)

        if 'train.amat' in amat:
            train_data = np.loadtxt(os.path.join(path, amat))
            for row in train_data:
                flat_image = row[:-1]
                image = np.reshape(flat_image, (28, 28))
                label = row[-1]
                x_train.append(image)
                y_train.append(label)

    x_train = np.expand_dims(x_train, axis=3)
    x_test = np.expand_dims(x_test, axis=3)

    return np.array(x_train), np.array(y_train), np.arrat(x_test), np.array(y_test)   


if __name__ == "__main__":
    # load_corrupted_testset("cifar", "brightness")
    load_colored_mnist()
