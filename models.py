import os
import itertools
import numpy as np
from scipy import io

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

import config
import dataloader
from resnet_model import resnet_v1, lr_schedule
from evaluation import compute_accuracy, compute_corruption_accuracy


def init_models(net):
    # modify this function to get feature extracture and end-to-end moddel
    if net == 'resnet':   
        depth = 56
        input_shape=(config.image_size, config.image_size, config.num_channels)     
        model = resnet_v1(input_shape=input_shape, depth=depth, num_classes=config.num_classes)
        return None, model
    elif net == 'resnet50':
        feature_extractor = tf.keras.applications.ResNet50(include_top=False, input_shape=(config.image_size, config.image_size, config.num_channels), pooling='avg') # pooling: None, avg or max
    elif net == 'xception':
        feature_extractor = tf.keras.applications.Xception(include_top=False, input_shape=(config.image_size, config.image_size, config.num_channels), pooling='avg') # pooling: None, avg or max
    elif net == 'layers_4':
        model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(config.image_size, config.image_size, config.num_channels)),
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        #keras.layers.Dropout(0.3),
        tf.keras.layers.Conv2D(256, (3,3), activation='relu'),
        #keras.layers.Dropout(0.3),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(512, (3,3), activation='relu'),
        #keras.layers.Dropout(0.3),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        #keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(64, activation='relu'),
        #keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(config.num_classes, activation='softmax')
        ])
        return None, model
    elif net == 'layers_5':
        model = tf.keras.models.Sequential([
            Conv2D(32, (3,3), activation="relu", padding="same", input_shape=(config.image_size, config.image_size, config.num_channels)),
            MaxPooling2D(pool_size=(2,2), strides=1),
            Conv2D(64, (3,3), activation="relu", padding="same"),
            MaxPooling2D(pool_size=(2,2), strides=1),
            Conv2D(64, (3,3), activation="relu", padding="same"),
            MaxPooling2D(pool_size=(2,2), strides=1),
            Conv2D(128, (3,3), activation="relu", padding="same"),
            Conv2D(256, (3,3), activation="relu", padding="same"),
            MaxPooling2D(pool_size=(2,2), strides=1),
            Flatten(),
            Dense(512, activation='relu'),
            Dense(64, activation='relu'),
            Dense(config.num_classes, activation='softmax')
        ])
        print(model.summary())
        return None, model
    else:
        # model = tf.keras.models.Sequential([
        #     Conv2D(32, (3,3), activation="relu", padding="same", input_shape=(config.image_size, config.image_size, config.num_channels)),
        #     MaxPooling2D(pool_size=(2,2), strides=1),
        #     Conv2D(64, (3,3), activation="relu", padding="same"),
        #     MaxPooling2D(pool_size=(2,2), strides=1),
        #     Conv2D(64, (3,3), activation="relu", padding="same"),
        #     MaxPooling2D(pool_size=(2,2), strides=1),
        #     Conv2D(128, (3,3), activation="relu", padding="same"),
        #     Conv2D(256, (3,3), activation="relu", padding="same"),
        #     MaxPooling2D(pool_size=(2,2), strides=1),
        #     Flatten(),
        #     Dense(512, activation='relu'),
        #     Dense(64, activation='relu'),
        #     Dense(config.num_classes, activation='softmax')
        # ])
        # print(model.summary())
        # return None, model
        print("model ka name gadbad hai!")

    x = tf.keras.layers.Dense(512, activation='relu')(feature_extractor.output)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dense(config.num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=feature_extractor.input, outputs=x)
    return feature_extractor, model


#############################################training module###########################################
# This is for Stage - 1

def train_baseline(x_train, y_train, x_val, y_val, net):

    _, model = init_models(net)

    if config.dataset == "cifar":        
        model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=lr_schedule(0)), metrics=['accuracy'])
        model.summary()

        lr_scheduler = LearningRateScheduler(lr_schedule)
        #lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
        callbacks = [lr_scheduler]
        model.fit(x_train, y_train, batch_size=config.b_batch_size, epochs=config.b_num_epochs,
                    validation_data=(x_val, y_val), shuffle=False, callbacks = callbacks)
        # , callbacks=callbacks
    elif config.dataset == "mnist" or "colored_mnist":
         model.compile(optimizer=tf.keras.optimizers.Adam(lr=config.b_lr), loss='binary_crossentropy', metrics=['accuracy'])
         model.fit(x_train, y_train, batch_size=config.b_batch_size, epochs=config.b_num_epochs, validation_data=(x_val, y_val))
   
    return model


########################################################################################################
# This is for Stage - 2, 3

def custom_loss(batch_pred, batch_labels, batch_size): 
    cce = tf.keras.losses.CategoricalCrossentropy()
    # cce = tf.keras.losses.BinaryCrossentropy()
    # cce = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    loss = cce(batch_labels, batch_pred)
    #loss = tf.reduce_sum(tf.abs(batch_pred - batch_labels))
    #return loss/batch_size
    return loss 

# # where is this used?
def jacobian_matrix(y_flat, x, num_classes=config.num_classes):
    for i in range(num_classes):
        if i==0:
            Jacobian = tf.gradients(y_flat[i], x)
        else:
            Jacobian = tf.concat([Jacobian, tf.gradients(y_flat[i], x)], axis=0)
    return Jacobian

def create_mask_single_channel(batch_jacobians, drop_rate, batch_size, net):
    batch_k = []
    k = abs(batch_jacobians[:1]) # to keep it 4-dimentional	
    r = k[0,:,:,0]
    rgb = tf.reshape((r), [-1])
    ind = int(drop_rate * rgb.shape[0])
    rgb = tf.sort(rgb)
    thresh = rgb[-ind]
    batch_preserve = tf.where(k<thresh, 0.0, 1.0)
    batch_drop = tf.where(k<thresh, 1.0, 0.0)

    for i in range(1, batch_size):
        k = abs(batch_jacobians[i:i+1]) # to keep it 4-dimentional
        r = k[0, :, :, 0]
        rgb = tf.reshape((r), [-1])
        ind = int(drop_rate * rgb.shape[0]) # 30% drop
        rgb = tf.sort(rgb)
        thresh = rgb[-ind]
        mask_preserve = tf.where(k<thresh, 0.0, 1.0)
        batch_preserve = tf.concat([batch_preserve, mask_preserve], axis=0)
        mask_drop = tf.where(k<thresh, 1.0, 0.0)
        batch_drop = tf.concat([batch_drop, mask_drop], axis=0)

    return batch_preserve, batch_drop    

def create_mask(batch_jacobians, drop_rate, batch_size, net):
    num_channels = batch_jacobians.shape[-1]
    if num_channels == 1:
        return create_mask_single_channel(batch_jacobians, drop_rate, batch_size, net)

    bj_c1 = tf.expand_dims(batch_jacobians[:, :, :, 0], axis=3)
    bj_c2 = tf.expand_dims(batch_jacobians[:, :, :, 1], axis=3)
    bj_c3 = tf.expand_dims(batch_jacobians[:, :, :, 2], axis=3)

    bp1, bd1 =  create_mask_single_channel(bj_c1, drop_rate, batch_size, net)
    bp2, bd2 =  create_mask_single_channel(bj_c2, drop_rate, batch_size, net)
    bp3, bd3 =  create_mask_single_channel(bj_c3, drop_rate, batch_size, net)

    batch_preserve = tf.concat([bp1, bp2, bp3], axis=3)
    batch_drop = tf.concat([bd1, bd2, bd3], axis=3)
    return batch_preserve, batch_drop    

def compute_jacobian(x_train, y_train, net, model=None, iteration=0, fname='train'):
    batch_size, drop_rate = config.jacobian_batch_size, config.drop_rate

    no_img = x_train.shape[0]
    # no_img = 3
    jacobian_list, manipulated_data, jacobian_preserved, manipulated_data_labels, image_list, mask_preserve_list, mask_drop_list = [], [], [], [], [], [], [] # Dropped and preserved Top 30% of the total sensitive pixels.
    manipulated_data_pred = []
    for epoch in range(1):

        Indices = range(no_img)
        for idx in range(no_img//batch_size):   # # resolve

            Ind = Indices[idx*batch_size : (idx+1)*batch_size]
            batch_images = tf.convert_to_tensor(x_train[Ind], tf.float32)
            batch_labels = tf.convert_to_tensor(y_train[Ind], tf.float32) # bcoz labels are already one-hot vectors from dataloader

            # Stage - 2

            with tf.GradientTape(persistent = True) as tape:
                #with tf.GradientTape(watch_accessed_variables=False) as t1:
                tape.watch(batch_images)
                batch_pred = model(batch_images)
                batch_pred = batch_pred*batch_labels
            jacobian = tape.gradient(batch_pred, batch_images)

            # Stage - 3
            #print("Creating masks..")
            mask_preserve, mask_drop = create_mask(jacobian, drop_rate, batch_size, net)
            jacobian_list.extend(jacobian.numpy())

            jacobian_preserved.extend(jacobian.numpy() * tf.convert_to_tensor(mask_preserve, tf.float32).numpy()) # tracking the sensitive pixels, independent jacobians/patterns.
            manipulated_data.extend(batch_images.numpy() * tf.convert_to_tensor(mask_drop, tf.float32).numpy()) # Manipulating the data
            manipulated_data_labels.extend(batch_labels.numpy()) # saving the corresponding labels
            image_list.extend(batch_images.numpy())
            mask_preserve_list.extend(mask_preserve.numpy())
            mask_drop_list.extend(mask_drop.numpy())
            manipulated_data_pred.extend(batch_pred.numpy())

    jacobian_list, jacobian_preserved = np.array(jacobian_list), np.array(jacobian_preserved)
    manipulated_data, manipulated_data_labels = np.array(manipulated_data), np.array(manipulated_data_labels)

    io.savemat(os.path.join(config.mats_path, "M_" + config.dataset + "_Jacobian_"+fname+'_'+str(iteration)+".mat"), dict(jacobian_preserved = jacobian_preserved, mask_preserve = mask_preserve, manipulated_data = manipulated_data, manipulated_data_labels = manipulated_data_labels, jacobian_list = jacobian_list, mask_preserve_list = mask_preserve_list, mask_drop_list = mask_drop_list, image_list = image_list, manipulated_data_pred = manipulated_data_pred))
   # io.savemat(os.path.join(config.mats_path, "M_" + config.dataset + "_Jacobian_"+fname+'_'+str(iteration)+".mat"), dict(jacobian_preserved = jacobian_preserved, mask_preserve = mask_preserve, jacobian_list = jacobian_list, mask_preserve_list = mask_preserve_list))

    return jacobian_list, jacobian_preserved, manipulated_data, manipulated_data_labels

########################################################################################################
# This is for Stage - 4

def load_jacobian_list(iterations):
    mat = io.loadmat(os.path.join(config.mats_path,"M_" + config.dataset + "_Jacobian_train_"+str(0)+".mat"))
    jacobian_preserved = mat['jacobian_list']
    if config.dataset == 'mnist': mask_preserve = mat['mask_preserve_list'] # for cifar mask-preserve fir mnist mask-preserve-list
    else:mask_preserve = mat['mask_preserve_list'] # for cifar mask-preserve fir mnist mask-preserve-list

    for iteration in range(1, config.iterations):
        mat = io.loadmat(os.path.join(config.mats_path, "M_" + config.dataset + "_Jacobian_train_"+str(iteration)+".mat"))
        next_jacobian_preserved = mat['jacobian_list']
       # jacobian_preserved += config.jac_weight*iteration*next_jacobian_preserved
        jacobian_preserved += next_jacobian_preserved
        if config.dataset == 'mnist': next_mask_preserve = mat['mask_preserve_list'] # for cifar mask-preserve fir mnist mask-preserve-list
        else: next_mask_preserve = mat['mask_preserve_list'] # for cifar mask-preserve fir mnist mask-preserve-list
        mask_preserve += next_mask_preserve
    return jacobian_preserved, mask_preserve

def drop_pixel(batch_x_train, drop_percent):
    masks = []
    total_pixels = config.image_size*config.image_size*config.num_channels
    for i in range(batch_x_train.shape[0]):
        mask = np.ones((config.image_size, config.image_size, config.num_channels))
        random_pixels = np.random.choice(total_pixels, int(total_pixels*drop_percent))
        np.put(mask, random_pixels, np.zeros(random_pixels.shape))
        masks.append(mask)

    pixels_dropped_images = batch_x_train * np.array(masks)
    return pixels_dropped_images


def dropout_model_learning(x_train, y_train, x_val, y_val, jacobian_preserved, mask_preserved, net):
    batch_size = config.dropout_batch_size
    optimizer = tf.keras.optimizers.Adam(lr=config.dropout_lr)
    # cosine_loss = tf.keras.losses.CosineSimilarity(axis=1)
    print("Training a Dropout model")
    _, model = init_models(net)
   # x_train = drop_pixel(x_train, config.drop_percent)
    no_img = x_train.shape[0]
    for epoch in range(config.dropout_epochs):
        print("Epoch ", epoch)
        if epoch == 6: optimizer = tf.keras.optimizers.Adam(lr=config.dropout_lr*0.1)
        if epoch == 12: optimizer = tf.keras.optimizers.Adam(lr=config.dropout_lr*0.01)
        for idx in range(no_img//batch_size - 2):
            # Ind = idx
            Indices = range(no_img)
            Ind = Indices[idx*batch_size : (idx+1)*batch_size]
            batch_x_train = x_train[Ind]
            # batch_x_train = drop_pixel(batch_x_train, config.drop_percent)
            batch_images = tf.convert_to_tensor(batch_x_train, tf.float32)
            batch_labels = tf.convert_to_tensor(y_train[Ind], tf.float32) 

            for batch_iteration in range(1):
                # with tf.GradientTape(persistent = True) as tape:
                with tf.GradientTape() as tape:
                    # tape.watch(batch_images)
                    batch_pred = model(batch_images, training=True)
                        
                    loss_cce = custom_loss(batch_pred, batch_labels, batch_size)
                    
                    total_loss = loss_cce 

                    BAcc = compute_accuracy(model, batch_images, batch_labels)                    
                    print("batch id, total loss, cce_loss, batch acc = ", idx, total_loss.numpy(), loss_cce.numpy(), BAcc)

                
                grads = tape.gradient(total_loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
        
       # model.save("drop.h5")         
       # Acc1 = compute_accuracy(model, x_train, y_train)
        Acc = compute_accuracy(model, x_val, y_val)
        print("Accuracy ", Acc)
        #print("val accuracy:", model.evaluate(x_val, y_val))

    return model

def jr_model_learning(x_train, y_train, x_val, y_val, jacobian_preserved, mask_preserved, net):
    batch_size = config.jr_batch_size
    optimizer = tf.keras.optimizers.Adam(lr=config.jr_lr)
    # cosine_loss = tf.keras.losses.CosineSimilarity(axis=1)
    print("Training a J-R model")
    _, model = init_models(net)
    
    no_img = x_train.shape[0]
    for epoch in range(config.jr_epochs):
        print("Epoch ", epoch)
        # optimizer = tf.keras.optimizers.Adam(lr=config.jr_lr)
        if epoch == 6: optimizer = tf.keras.optimizers.Adam(lr=config.jr_lr*0.1)
        if epoch == 12: optimizer = tf.keras.optimizers.Adam(lr=config.jr_lr*0.01)
        for idx in range(no_img//batch_size - 2):

            # Ind = idx
            Indices = range(no_img)
            Ind = Indices[idx*batch_size : (idx+1)*batch_size]
            batch_images = tf.convert_to_tensor(x_train[Ind], tf.float32)
            batch_labels = tf.convert_to_tensor(y_train[Ind], tf.float32) # bcoz labels are already one-hot vectors from dataloader
            batch_jacobian_preserved = tf.convert_to_tensor(jacobian_preserved[Ind], tf.float32)
            batch_mask_preserved = tf.convert_to_tensor(mask_preserved[Ind], tf.float32)


            for batch_iteration in range(1):
                with tf.GradientTape(persistent = True) as tape:
                    with tf.GradientTape(watch_accessed_variables=False) as t1:
                        t1.watch(batch_images)
                        batch_pred = model(batch_images, training=True)
                        
                        batch_pred_1 = batch_pred*batch_labels
                    jacobian = t1.gradient(batch_pred_1, batch_images)
                    
                    jacobian_loss = tf.reduce_sum(tf.norm(jacobian, ord="euclidean"))
                    
                    loss_cce = custom_loss(batch_pred, batch_labels, batch_size)
                    if epoch > 5:
                        total_loss = loss_cce + jacobian_loss
                    else:
                        total_loss = loss_cce
                    
                    print("batch id, total loss, jacobian_loss, cce_loss = ", idx, total_loss.numpy(), jacobian_loss.numpy(), loss_cce.numpy())

                
                grads = tape.gradient(total_loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
        

        Acc = compute_accuracy(model, x_val, y_val)
        print(Acc)
        # print("val accuracy:", model.evaluate(x_val, y_val))

    return model

def unified_model_learning(x_train, y_train, x_val, y_val, jacobian_preserved, mask_preserved, net):
    batch_size = config.uni_batch_size
    optimizer = tf.keras.optimizers.Adam(lr=config.uni_lr)
    # cosine_loss = tf.keras.losses.CosineSimilarity(axis=1)
    print("Training a unified model")
    _, model = init_models(net)
    
    no_img = x_train.shape[0]
    for epoch in range(config.uni_epochs):
        print("Epoch ", epoch)
        # if epoch == 5: optimizer = tf.keras.optimizers.Adam(lr=config.jr_lr*0.1)
        # if epoch == 8: optimizer = tf.keras.optimizers.Adam(lr=config.jr_lr*0.01)
        # if epoch == 11: optimizer = tf.keras.optimizers.Adam(lr=config.jr_lr*0.001)
        for idx in range(no_img//batch_size - 1):

            # Ind = idx
            Indices = range(no_img-1000)
            Ind = Indices[idx*batch_size : (idx+1)*batch_size]
            batch_images = tf.convert_to_tensor(x_train[Ind], tf.float32)
            batch_labels = tf.convert_to_tensor(y_train[Ind], tf.float32) # bcoz labels are already one-hot vectors from dataloader
            batch_jacobian_preserved = tf.convert_to_tensor(jacobian_preserved[Ind], tf.float32)
            batch_mask_preserved = tf.convert_to_tensor(mask_preserved[Ind], tf.float32)


            for batch_iteration in range(1):
                with tf.GradientTape(persistent = True) as tape:
                    with tf.GradientTape(watch_accessed_variables=False) as t1:
                        t1.watch(batch_images)
                        batch_pred = model(batch_images, training=True)
                        batch_pred_1 = batch_pred*batch_labels
                    jacobian = t1.gradient(batch_pred_1, batch_images)
                    # jacobian_loss = tf.reduce_sum(tf.abs(tf.square(jacobian - batch_jacobian_preserved))) / batch_size
                    jacobian_loss = tf.reduce_sum(tf.norm(tf.square(tf.abs(jacobian - batch_jacobian_preserved)), ord="euclidean"))
                    loss_cce = custom_loss(batch_pred, batch_labels, batch_size)

                    # if epoch < 1:
                    #     total_loss = loss_cce
                    # else:
                    #     total_loss = loss_cce + (epoch+1)*(jacobian_loss/batch_size)
                    # BAcc = compute_accuracy(model, batch_images, batch_labels)                    
                    # print("batch id, total loss, cce_loss, jacobian_loss, batch acc = ", idx, total_loss.numpy(), jacobian_loss.numpy(), loss_cce.numpy(), BAcc)
                    if epoch > 2:
                        total_loss = loss_cce 
                    elif epoch >= 0:
                        total_loss = jacobian_loss + loss_cce 
                    else:
                        total_loss = 10*jacobian_loss
                    BAcc = compute_accuracy(model, batch_images, batch_labels)                    
                    print("batch id, total loss, cce_loss, jacobian_loss, batch acc = ", idx, total_loss.numpy(), jacobian_loss.numpy(), loss_cce.numpy(), BAcc)

                    # if epoch < 5:
                    #     total_loss = loss_cce 
                    #       #Acc = Compute_Accuracy_manipulated(x_val, y_val, model)
                    # elif epoch < 11:
                    #     total_loss = jacobian_loss 
                    # elif epoch < 17:
                    #     total_loss = jacobian_loss + loss_cce 
                    # else:
                    #     total_loss = loss_cce
                    # print("batch id, total loss, jacobian_loss, cce_loss = ", idx, total_loss.numpy(), jacobian_loss.numpy(), loss_cce.numpy())

                
                grads = tape.gradient(total_loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
        
        # model.save("test.h5")
        Acc = compute_accuracy(model, x_val, y_val)
        print(Acc)
        # print("val accuracy:", model.evaluate(x_val, y_val))

    return model


# #####################################################################################################
# # # UNUSED
# #####################################################################################################

# def train_long_model(x_train, y_train, x_val, y_val, net):
#     batch_size = 1

#     #train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(buffer_size=len(x_train)).batch(batch_size)
#     #train_dataset = tf.data.Dataset.from_tensor_slices((x_train_df, x_train_mlfp))
#     #train_dataset = train_dataset.shuffle(buffer_size=765).batch(32)

#     feature_extractor, model = init_models(net)
#     optimizer = tf.keras.optimizers.Adam(lr=0.0001)
#     cce_loss = tf.keras.losses.CategoricalCrossentropy()
#     #model.compile(optimizer=optimizer, loss=custom_loss(batch_size))
#     no_img = x_train.shape[0]

#     for epoch in range(config.num_epochs):
#         total_loss = 0.0
#         loss_avg = tf.keras.metrics.Mean()
#         loss_avg.reset_states()

#         for idx in range(no_img//batch_size):
#         #Indices = np.random.permutation(x_train.shape[0])
#             Indices = range(no_img)
#             Ind = Indices[idx*batch_size : (idx+1)*batch_size]
#             batch_images = tf.convert_to_tensor(x_train[Ind], tf.float32)
#             batch_labels = tf.convert_to_tensor(y_train[Ind], tf.float32) # bcoz labels are already one-hot vectors from dataloader

#             #print("batch_images.shape", batch_images.shape, "batch_labels.shape", batch_labels.shape, type(batch_images), type(batch_labels))

#             #for step, (x_p, x_d) in enumerate(zip(batch_mlfp, batch_df)):
#             with tf.GradientTape(persistent = True) as tape:
#                 with tf.GradientTape(watch_accessed_variables=False) as t1:
#                     t1.watch(batch_images)
#                     batch_pred = model(batch_images)
#                 print("batch_pred.shape:", batch_pred.shape)
#                 jacobian = t1.jacobian(batch_pred, batch_images)
#                 print("jacobian.shape:", jacobian.shape)
#                 loss_cce = custom_loss(batch_pred, batch_labels, batch_size)
#                 loss_jacobian = tf.norm(jacobian)
#                 total_loss = loss_jacobian + loss_cce
#                 print("total loss = ", total_loss, loss_jacobian, loss_cce)
#             grads = tape.gradient(total_loss, model.trainable_variables)
#             optimizer.apply_gradients(zip(grads, model.trainable_variables))

#             # with tf.GradientTape(persistent = True) as tape:
#             #     tape.watch(batch_images)
#             #     batch_pred = model(batch_images, training=True)
#             #     loss_cce = custom_loss(batch_pred, batch_labels, batch_size)
#             #     print("batch_pred.shape:", batch_pred.shape)

#             #     if idx > 50:
#             #         total_loss = loss_cce
#             #         print("batch_number:", idx, "cce_loss:", loss_cce.numpy())
#             #     else:
#             #         argmax = tf.math.argmax(batch_labels, axis=1)
#             #         print("argmax:", argmax, "argmax.shape:", argmax.shape)
#             #         batch_pred_1 = tf.expand_dims(tf.gather(batch_pred[0], argmax[0]), axis=0)

#             #         print("batch:", batch_pred_1.shape)
#             #         for i in range(batch_size-1):
#             #             batch_pred_1 = tf.concat([batch_pred_1, tf.expand_dims(tf.gather(batch_pred[i+1], argmax[i+1]), axis = 0)], axis=0)
#             #         batch_pred_1 = tf.expand_dims(batch_pred_1, axis = 1)
#             #         print("batch_pred_1:", batch_pred_1, "batch_pred_1.shape:", batch_pred_1.shape)
#             #         with
#             #         jacobian = tape.gradient(batch_pred_1, batch_images)

#             #         print("jacobian.shape:", jacobian.shape)
#             #         loss_jacobian = tf.norm(tf.abs(jacobian))
#             #         total_loss = loss_cce
#                    #      #total_loss = loss_cce + loss_jacobian/1000000.0
#                    #      #print("jacobian:", jacobian)

#             #         print("batch_number:", idx, "cce_loss:", loss_cce.numpy(), "loss_jacobian:", loss_jacobian.numpy(), "total_loss:", total_loss.numpy())
#             # jacobian_ver = tape.jacobian(batch_pred, batch_images)
#             # io.savemat("Jac.mat", dict(jacobian = jacobian.numpy(), jacobian_ver = jacobian_ver.numpy(), argmax = argmax.numpy()))
#             # # The crucial step - train the model.
#             # grads = tape.gradient(total_loss, model.trainable_variables)
#             # # print("grads.shape:", grads)
#             # optimizer.apply_gradients(zip(grads, model.trainable_variables))


# def plot_accuracy(history, model_name):
#     if not os.path.isdir('plots'):
#         os.mkdir('plots')

#     d = history.history
#     loss, val_loss = d['loss'], d['val_loss']
#     acc, val_acc = d['accuracy'], d['val_accuracy']

#     plt.figure()
#     plt.plot(list(range(len(loss))), loss)
#     plt.plot(list(range(len(val_loss))), val_loss)
#     plt.xlabel("Num of Epochs")
#     plt.ylabel("Loss")
#     plt.title("Training Loss vs Validation Loss")
#     plt.legend(['train','validation'])
#     plt.savefig(os.path.join('plots', 'loss_'+model_name+'.jpg'), bbox_inches='tight', dpi=150)
#     plt.show()

#     plt.figure()
#     plt.plot(list(range(len(loss))), loss)
#     plt.plot(list(range(len(val_loss))), val_loss)
#     plt.xlabel("Num of Epochs")
#     plt.ylabel("Loss")
#     plt.title("Training Loss vs Validation Loss")
#     plt.legend(['train','validation'])
#     plt.savefig(os.path.join('plots', 'loss_'+model_name+'.jpg'), bbox_inches='tight', dpi=150)
#     plt.show()

#     plt.figure()
#     plt.plot(list(range(len(acc))), acc)
#     plt.plot(list(range(len(val_acc))), val_acc)
#     plt.xlabel("Num of Epochs")
#     plt.ylabel("Accuracy")
#     plt.title("Training Accuracy vs Validation Accuracy")
#     plt.legend(['train','validation'])
#     plt.savefig(os.path.join('plots', 'accuracy'+model_name+'.jpg'), bbox_inches='tight', dpi=150)
#     plt.show()


# def create_mask(batch_jacobians, drop_rate, batch_size, net):
#     if net == "layers_4":
#         batch_k = []
#         k = abs(batch_jacobians[:1]) # to keep it 4-dimentional	
#         r = k[0,:,:,0]
#         rgb = tf.reshape((r), [-1])
#         ind = int(drop_rate * rgb.shape[0])
#         rgb = tf.sort(rgb)
#         thresh = rgb[-ind]
#         batch_preserve = tf.where(k<thresh, 0.0, 1.0)
#         batch_drop = tf.where(k<thresh, 1.0, 0.0)

#         for i in range(1, batch_size):
#             k = abs(batch_jacobians[i:i+1]) # to keep it 4-dimentional
#             r = k[0, :, :, 0]
#             rgb = tf.reshape((r), [-1])
#             ind = int(drop_rate * rgb.shape[0]) # 30% drop
#             rgb = tf.sort(rgb)
#             thresh = rgb[-ind]
#             mask_preserve = tf.where(k<thresh, 0.0, 1.0)
#             batch_preserve = tf.concat([batch_preserve, mask_preserve], axis=0)
#             mask_drop = tf.where(k<thresh, 1.0, 0.0)
#             batch_drop = tf.concat([batch_drop, mask_drop], axis=0)
#         return batch_preserve, batch_drop    

#     batch_k = []
#     k = abs(batch_jacobians[:1]) # to keep it 4-dimentional
#     r, g, b = k[0, :, :, 0], k[0, :, :, 1], k[0, :, :, 2]
#     rgb = tf.reshape((r+g+b), [-1])
#     ind = int(drop_rate * rgb.shape[0]) # 30% drop
#     rgb = tf.sort(rgb)
#     thresh = rgb[-ind]
#     batch_preserve = tf.where(k<thresh, 0.0, 1.0)
#     batch_drop = tf.where(k<thresh, 1.0, 0.0)

#     for i in range(1, batch_size):
#         k = abs(batch_jacobians[i:i+1]) # to keep it 4-dimentional
#         r, g, b = k[0, :, :, 0], k[0, :, :, 1], k[0, :, :, 2]
#         rgb = tf.reshape((r+g+b), [-1])
#         ind = int(drop_rate * rgb.shape[0]) # 30% drop
#         rgb = tf.sort(rgb)
#         thresh = rgb[-ind]
#         mask_preserve = tf.where(k<thresh, 0.0, 1.0)
#         batch_preserve = tf.concat([batch_preserve, mask_preserve], axis=0)
#         mask_drop = tf.where(k<thresh, 1.0, 0.0)
#         batch_drop = tf.concat([batch_drop, mask_drop], axis=0)
#     return batch_preserve, batch_drop