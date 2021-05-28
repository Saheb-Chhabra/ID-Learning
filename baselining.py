#baselining

import os
import sys
import cv2
import keras
import config
import numpy as np
import scipy.io as io

import tensorflow as tf
#from keras.applications.resnet50 import ResNet50
from keras.applications.xception import Xception
from sklearn.metrics import accuracy_score, classification_report

import config
import models
import dataloader

#tf.config.gpu.set_per_process_memory_fraction(0.65)
#tf.config.gpu.set_per_process_memory_growth(True)


def check_dir():
    if not os.path.isdir('proposed_models++'):
        os.mkdir('proposed_models++')
    if not os.path.isdir(os.path.join('proposed_models++', 'resnet')):
        os.mkdir(os.path.join('proposed_models++', 'resnet'))
    
    if not os.path.isdir(os.path.join('proposed_models++', 'xception')):
        os.mkdir(os.path.join('proposed_models++', 'xception'))

    if not os.path.isdir('models++'):
        os.mkdir('models++')
    if not os.path.isdir(os.path.join('models++', 'resnet')):
        os.mkdir(os.path.join('models++', 'resnet'))
    
    if not os.path.isdir(os.path.join('models++', 'xception')):
        os.mkdir(os.path.join('models++', 'xception'))


def evaluate(x_test, y_test, model_physical, model_digital):
	# score-level fusion

	physical_preds = np.argmax(model_physical.predict(x_test), axis=1)
	digital_preds = np.argmax(model_digital.predict(x_test), axis=1)

	print(physical_preds.shape, digital_preds.shape)

	pred = []
	for p_pred, d_pred in zip(physical_preds, digital_preds):
		if p_pred == d_pred and p_pred == 0: #If both model predicts real
			pred.append(0)
		elif p_pred == 1 or d_pred == 1:     #If eithe of the model predicts fake
			pred.append(1)
		else:
			print("In else", p_pred, d_pred)
	acc = sum(pred == y_test)/len(y_test)
	
	return acc


network = ['resnet', 'xception']

#n = int(sys.argv[1])
#d = int(sys.argv[2])

net = network[0]
dataset_name = config.dataset

#config.load_model = True

if config.dataset == 'mnist':
	x_train, y_train, x_val, y_val = dataloader.load_mnist()
elif config.dataset == 'cifar':
	x_train, y_train, x_val, y_val = dataloader.load_cifar()
else:
	print("ye konsa dataset hai be:", config.dataset)

print("shape of final dataset:", x_train.shape, y_train.shape)
print("shape of valset:", x_val.shape, y_val.shape)

check_dir()

print("network:", net, " Datset:", dataset_name)

# model = models.train_baseline(x_train, y_train, x_val, y_val, net)
model = models.train_long_model(x_train, y_train, x_val, y_val, net)

print("model training complete")
model.save(os.path.join('models++', net, net+'_'+config.dataset+'.h5'))

print("Models saved")

'''
#_, model = init_models(net)
model = tf.keras.models.load_model(os.path.join('models++', net, net+.h5'))

'''

x_test, y_test = np.array(x_val), np.array(y_val)

print("Accuracy on valset: ", model.evaluate(x_val, y_val))

print("Done")