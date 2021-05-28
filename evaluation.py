
import os
import sys
import cv2
import keras
import config
import numpy as np
import scipy.io as io

import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report

import config
import models
import dataloader


def resize_images(images, size=72):
    new_images = []
    for im in images:
        x = cv2.resize(im, (size,size), interpolation=cv2.INTER_CUBIC)
        # x = resize(im, (size,size), anti_aliasing=True)
        new_images.append(x)
    new_images = np.array(new_images)
    return new_images
'''
def compute_accuracy(model, x_val, y_val, resize=None):
    Pred = []
    if resize is not None:
        x_val = resize_images(x_val)
    for i in range(0, x_val.shape[0], 1000):
        ind = 1000+i
        Pred.extend(model.predict(x_val[i:ind]))
   # Pred = model(x_val)
    PredLab = np.argmax(Pred, axis = 1)
    TrueLab = np.argmax(y_val, axis = 1)
    # return accuracy_score(TrueLab, PredLab)
    Acc = (TrueLab - PredLab)
    acc = np.sum(Acc == 0)/float(len(x_val))
    return acc
'''

def compute_accuracy(model, x_val, y_val, resize=None):
    
    if resize is not None:
        x_val = resize_images(x_val)

    Pred = model(x_val)
    PredLab = np.argmax(Pred, axis = 1)
    TrueLab = np.argmax(y_val, axis = 1)
    # return accuracy_score(TrueLab, PredLab)
    Acc = (TrueLab - PredLab)
    acc = np.sum(Acc == 0)/float(len(x_val))
    return acc

# # for performance on corrupted datasets
def compute_corruption_accuracy(model, resize=None):
    accs = []
    for corruption_type in config.corruption_list:
        x_test_c, y_test_c = dataloader.load_corrupted_testset(config.dataset, corruption_type)
        y_test_c = keras.utils.to_categorical(y_test_c, config.num_classes)
        acc = compute_accuracy(model, x_test_c, y_test_c, resize)
        accs.append(acc)
        print("Corruption type:", corruption_type, "Accuracy: ", acc)
    print("Mean accuracy: ", np.mean(np.array(accs)))

# # for performance on data manipulated for training the algorithm
def compute_accuracy_manipulated_data(model):
    for iteration in range(config.iterations):
        mat = io.loadmat(os.path.join(config.mats_path, "M_" + config.dataset + "_Jacobian_test_"+str(iteration)+".mat"))
        x_test_c = mat['manipulated_data']              # # eval on test set
        y_test_c = mat['manipulated_data_labels']
        acc = compute_accuracy(model, x_test_c, y_test_c)
        print("Accuracy on manipulated set: ", iteration, acc)

def ce(new_model, base_model, x_test, y_test, resize=None):

    base_clean_acc = compute_accuracy(base_model, x_test, y_test, resize)
    new_clean_acc = compute_accuracy(new_model, x_test, y_test)
    base_clean_error = 100.0 - base_clean_acc
    new_clean_error = 100.0 - new_clean_acc

    print("Clean Base Accuracy   : ", base_clean_acc)
    print("Clean Unified Accuracy: ", new_clean_acc)
    avg_relative_ce = 0.0
    for corruption_type in config.corruption_list:
        x_test_c, y_test_c = dataloader.load_corrupted_testset(config.dataset, corruption_type)
        y_test_c = keras.utils.to_categorical(y_test_c, config.num_classes)
        base_cacc = compute_accuracy(base_model, x_test_c, y_test_c, resize)
        new_cacc = compute_accuracy(new_model, x_test_c, y_test_c)

        base_cerror = 100.0 - base_cacc 
        new_cerror = 100.0 - new_cacc
        rel_ce = new_cerror/base_cerror
        print("corruption type: {:>12}, Base accuracy: {:5f},  Unified accuracy: {:5f}, Corruption: {:5f}".format(corruption_type, base_cacc, new_cacc, rel_ce))
        avg_relative_ce += rel_ce
    
    avg_relative_ce = avg_relative_ce/float(len(config.corruption_list))
    print("Mean CE: ", avg_relative_ce)

def relative_ce(new_model, base_model, x_test, y_test, resize=None):
    base_clean_acc = compute_accuracy(base_model, x_test, y_test, resize)
    new_clean_acc = compute_accuracy(new_model, x_test, y_test)
    base_clean_error = 100.0 - base_clean_acc
    new_clean_error = 100.0 - new_clean_acc

    print("Clean Base Accuracy   : ", base_clean_acc)
    print("Clean Unified Accuracy: ", new_clean_acc)
    avg_relative_ce = 0.0
    for corruption_type in config.corruption_list:
        x_test_c, y_test_c = dataloader.load_corrupted_testset(config.dataset, corruption_type)
        y_test_c = keras.utils.to_categorical(y_test_c, config.num_classes)
        base_cacc = compute_accuracy(base_model, x_test_c, y_test_c, resize)
        new_cacc = compute_accuracy(new_model, x_test_c, y_test_c)

        base_cerror = 100.0 - base_cacc 
        new_cerror = 100.0 - new_cacc
        rel_ce = (new_cerror - new_clean_error)/(base_cerror - base_clean_error)
        print("corruption type: {:>12}, Base accuracy: {:5f},  Unified accuracy: {:5f}, Rel. Corruption: {:5f}".format(corruption_type, base_cacc, new_cacc, rel_ce))
        avg_relative_ce += rel_ce
    avg_relative_ce = avg_relative_ce/float(len(config.corruption_list))
    print("Mean Relative CE: ", avg_relative_ce)