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
import evaluation
print(tf.test.is_gpu_available())

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6024)])
  except RuntimeError as e:
    print(e)
#main

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu_devices[0], True)

opts = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
sess = tf.Session(config=tf.ConfigProto(gpu_options=opts))

tf.config.gpu.set_per_process_memory_fraction(0.3)
tf.config.gpu.set_per_memory_growth(True)

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8096)])

def check_dir():
    pass
    # if not os.path.isdir('proposed_models++'):
    #     os.mkdir('proposed_models++')
    # if not os.path.isdir(os.path.join('proposed_models++', 'resnet')):
    #     os.mkdir(os.path.join('proposed_models++', 'resnet'))

    # if not os.path.isdir(os.path.join('proposed_models++', 'xception')):
    #     os.mkdir(os.path.join('proposed_models++', 'xception'))

    # if not os.path.isdir('models++'):
    #     os.mkdir('models++')
    # if not os.path.isdir(os.path.join('models++', 'resnet')):
    #     os.mkdir(os.path.join('models++', 'resnet'))

    # if not os.path.isdir(os.path.join('models++', 'xception')):
    #     os.mkdir(os.path.join('models++', 'xception'))

    # if not os.path.isdir(os.path.join('models++', 'layers_4')):
    #     os.mkdir(os.path.join('models++', 'layers_4'))


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


# network = ['resnet', 'xception', 'layers_4']

#n = int(sys.argv[1])
#d = int(sys.argv[2])

# net = network[2]
net = config.net
dataset_name = config.dataset

#config.load_model = True

if config.dataset == 'mnist':
        x_train, y_train, x_val, y_val = dataloader.load_mnist()
elif config.dataset == 'cifar':
        x_train, y_train, x_val, y_val = dataloader.load_cifar()
elif config.dataset == 'cifar100':
        x_train, y_train, x_val, y_val = dataloader.load_cifar100()
else:
        print("ye konsa dataset hai be:", config.dataset)

print("shape of final dataset:", x_train.shape, y_train.shape)
print("shape of valset:", x_val.shape, y_val.shape)

check_dir()

print("network:", net, " Datset:", dataset_name)

if config.stage_1 == True:
    print("Entering stage 1")
    # stage - 1
    model = None
    model = models.train_baseline(x_train, y_train, x_val, y_val, net)
    print("model training complete")
    # model.save(os.path.join(config.baseline_model_path, net, net+'_'+config.dataset+'.h5'))
    model.save(os.path.join(config.baseline_model_path, net+'_'+config.dataset+'.h5'))
    print("Models saved")

manipulated_x_train, manipulated_y_train = x_train, y_train
if config.stage_2_3 == True:
    print("Entering stage 2 and stage 3")

    model = tf.keras.models.load_model(os.path.join(config.baseline_model_path, config.net+'_'+config.dataset+".h5"))
    print("Model loaded.")
    
    '''
    for i in range(config.iterations):
            # stage - 2 and 3 combined
            jacobian_list, jacobian_preserved, manipulated_x_train, manipulated_y_train = models.compute_jacobian(manipulated_x_train, manipulated_y_train, net, model, i, 'train')
            print("Jacobian computation complete.. shape=", jacobian_list.shape)
            print("jacobian_preserved.shape:", jacobian_preserved.shape, "manipulated_x_train.shape:", manipulated_x_train.shape, "manipulated_y_train:", manipulated_y_train.shape)

            model = models.train_baseline(manipulated_x_train, manipulated_y_train, x_val, y_val, net)
            print("model training complete")
            model.save(os.path.join(config.inter_model_path, net+'_'+config.dataset+'iteration_'+str(i)+'.h5'))

            print("Models saved")
    '''
               
    manipulated_x_train, manipulated_y_train = x_val, y_val
    for i in range(config.iterations):
            # stage - 2 and 3 combined
            jacobian_list, jacobian_preserved, manipulated_x_train, manipulated_y_train = models.compute_jacobian(manipulated_x_train, manipulated_y_train, net, model, i, 'test')
            print("Jacobian computation complete.. shape=", jacobian_list.shape)
            print("jacobian_preserved.shape:", jacobian_preserved.shape, "manipulated_x_train.shape:", manipulated_x_train.shape, "manipulated_y_train:", manipulated_y_train.shape)

            model = tf.keras.models.load_model(os.path.join(config.inter_model_path, net+'_'+config.dataset+'iteration_'+str(i)+'.h5'))

            print("Models saved")
    

'''
#_, model = init_models(net)
model = tf.keras.models.load_model(os.path.join('models++', net, net+.h5'))
'''

if config.stage_4 == True:
    print("Entering stage 4")
    # Stage - 4
    jacobian_preserved, mask_preserved = models.load_jacobian_list(config.iterations)
    model = models.unified_model_learning(x_train, y_train, x_val, y_val, jacobian_preserved, mask_preserved, net)
    # model.save(os.path.join(config.proposed_model_path, net, "unified_"+net+'_'+config.dataset+'.h5'))
    model.save(os.path.join(config.proposed_model_path, "unified_"+net+'_'+config.dataset+'.h5'))
    print("Unified model saved successfully..")
    
if config.stage_5 == True:
    print("Entering stage 5")
    # Stage - 4
    jacobian_preserved, mask_preserved = models.load_jacobian_list(config.iterations)
    model = models.jr_model_learning(x_train, y_train, x_val, y_val, jacobian_preserved, mask_preserved, net)
    model.save(os.path.join(config.jr_model_path, "jr_"+net+'_'+config.dataset+'.h5'))
    print("Jacobian Regularized model saved successfully..")
    
if config.stage_6 == True:
    print("Entering stage 6")
    # Stage - 4
    jacobian_preserved, mask_preserved = models.load_jacobian_list(config.iterations)
    model = models.dropout_model_learning(x_train, y_train, x_val, y_val, jacobian_preserved, mask_preserved, net)
    model.save(os.path.join(config.dropout_model_path, "dropout_"+net+'_'+config.dataset+'.h5'))
    print("Dropout model saved successfully..")
    
if config.stage_7 == True:
    print("Entering stage 7")
    # Stage - 4
    baseline_model = tf.keras.models.load_model(os.path.join(config.baseline_model_path, config.net+'_'+config.dataset+".h5"))
    #evaluation.compute_accuracy_manipulated_data(baseline_model)

    jr_model = tf.keras.models.load_model(os.path.join(config.jr_model_path, "jr_"+net+'_'+config.dataset+'.h5'))
    evaluation.relative_ce(jr_model, baseline_model, x_val, y_val)
    #evaluation.compute_accuracy_manipulated_data(jr_model)

    dropout_model = tf.keras.models.load_model(os.path.join(config.dropout_model_path, "dropout_"+net+'_'+config.dataset+'.h5'))
    evaluation.relative_ce(dropout_model, baseline_model, x_val, y_val)
    #evaluation.compute_accuracy_manipulated_data(dropout_model)

    unified_model = tf.keras.models.load_model(os.path.join(config.proposed_model_path, "unified_"+net+'_'+config.dataset+'.h5'))
    evaluation.relative_ce(unified_model, baseline_model, x_val, y_val)
    #evaluation.compute_accuracy_manipulated_data(unified_model)

                                                

# x_test, y_test = np.array(x_val), np.array(y_val)

# print("Accuracy on valset: ", model.evaluate(x_val, y_val))

# for corruption_type in config.corruption_list:
#     x_test_c, y_test_c = dataloader.load_corrupted_testset(config.dataset, corruption_type)
#     print("corruption type:", corruption_type, "Accuracy on corrupted set: ", model.evaluate(x_test_c, y_test_c))

# print("Done")
