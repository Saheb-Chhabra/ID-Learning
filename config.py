# uncompyle6 version 3.7.4
# Python bytecode 3.8 (3413)
# Decompiled from: Python 3.6.8 (default, Jan 14 2019, 11:02:34) 
# [GCC 8.0.1 20180414 (experimental) [trunk revision 259383]]
# Embedded file name: /workspace/code/config.py
# Compiled at: 2021-01-21 12:38:33
# Size of source mod 2**32: 5249 bytes
import os

def CuDNN_error():
    c = tf.compat.v1.ConfigProto()
    c.gpu_options.allow_growth = True
    session = tf.compat.v1.InteractiveSession(config=c)
    return (c, session)


def set_GPU(num):
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(num)


root = '/workspace/'
baseline_model_path = os.path.join(root, 'models_baseline')
jr_model_path = os.path.join(root, 'models_jr')
dropout_model_path = os.path.join(root, 'models_dropout')
inter_model_path = os.path.join(root, 'models_inter')
proposed_model_path = os.path.join(root, 'models_proposed')
mats_path = os.path.join(root, 'mats')

dataset_type = 'clean'
#dataset = 'mnist'   # mnist, cifar100
dataset = 'cifar100'

mnsitc_path = '../corrupted_data/mnist_c/'
cifarc_path = '../corrupted_data/CIFAR-10-C/'
cifarc100_path = '../corrupted_data/CIFAR-100-C/CIFAR-100-C-sz96/'

colored_mnist_path = '../data/colored_mnist/'
if dataset == 'cifar100':
    cuda_device = 0
else:
    if dataset == 'mnist':
        cuda_device = 2
    else:
        cuda_device = 0
set_GPU(cuda_device)

load_model = True
random_state = 42
stage_1, stage_2_3, stage_4, stage_5, stage_6, stage_7 = (False, False, False, False,
                                                          False, True)
loss = 'cross_entropy'
#net = 'layers_5'
net = 'xception'


b_num_epochs = 10
b_lr = 0.0001
b_batch_size = 32

drop_rate = 0.1
jacobian_batch_size = 1000
iterations = 4
jac_weight = 1

uni_epochs = 4
uni_lr = 0.001
uni_batch_size = 500

jr_epochs = 15
jr_lr = 0.0001
jr_batch_size = 32

dropout_epochs = 10
dropout_lr = 0.0001
drop_percent = 0.03
dropout_batch_size = 32

def init_dataset(dataset_type, dataset_name):
    if dataset_name == 'mnist':
        image_size = 28
        num_channels = 1
        num_classes = 10
        corruption_dict = {'brightness':True, 
         'dotted_line':True,  'glass_blur':True,  'impulse_noise':True,  'rotate':True,  'shear':True, 
         'spatter':True,  'translate':True,  'canny_edges':True,  'fog':True,  'motion_blur':True, 
         'scale':True,  'shot_noise':True,  'stripe':True,  'zigzag':True}
        corruption_list = [key for key, value in corruption_dict.items() if value == True]
    elif dataset_name == 'cifar':
        image_size = 32
        num_channels = 3
        num_classes = 10
        corruption_dict = {'defocus_blur':False, 
            'contrast':False,  'pixelate':True,  'snow':False, 
            'fog':False,  'speckle_noise':False,  'glass_blur':False,  'brightness':False, 
            'elastic_transform':False,  'frost':False, 
            'jpeg_compression':False,  'shot_noise':False,  'spatter':False,  'saturate':False, 
            'impulse_noise':True,  'zoom_blur':False,  'gaussian_noise':False,  'motion_blur':False, 
            'gaussian_blur':False}
        corruption_list = [key for key, value in corruption_dict.items() if value == True]
    elif dataset_name == 'cifar100':
        image_size = 72
        num_channels = 3
        num_classes = 100
        corruption_dict = {'defocus_blur':True, 
            'contrast':True,  'pixelate':True,  'snow':True, 
            'fog':True,  'speckle_noise':False,  'glass_blur':True,  'brightness':True, 
            'elastic_transform':True,  'frost':True, 
            'jpeg_compression':True,  'shot_noise':True,  'spatter':False,  'saturate':False, 
            'impulse_noise':True,  'zoom_blur':True,  'gaussian_noise':True,  'motion_blur':True, 
            'gaussian_blur':False}
        corruption_list = [key for key, value in corruption_dict.items() if value == True]
    elif dataset_name == 'colored_mnist':
            color_var = 0.02
            image_size = 28
            num_channels = 3
            num_classes = 10
            corruption_list = []
    
    if dataset_type == 'occluded':
        occlusion_color = 'white'
        occlusion_range = '20.0'
        set_ = 'test'
        return (
         image_size, num_channels, num_classes, corruption_list, occlusion_color, occlusion_range, set_)
    return (
     image_size, num_channels, num_classes, corruption_list)


if dataset_type == 'clean':
    image_size, num_channels, num_classes, corruption_list = init_dataset(dataset_type, dataset)
else:
    if dataset_type == 'occluded':
        image_size, num_channels, num_classes, corruption_list, occlusion_color, occlusion_range, set_ = init_dataset(dataset_type, dataset)
# okay decompiling config.cpython-38.pyc
