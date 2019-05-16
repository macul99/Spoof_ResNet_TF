import os
from os import path
import tensorflow as tf
from tensorflow.keras.regularizers import l1, l2

BASE_PATH = './dataset' # soft link to '/media/macul/black/spoof_db/train_test_all/tfrecord'
BASE_PATH = os.path.abspath(BASE_PATH)

CFG_PATH = './config'
CFG_PATH = os.path.abspath(CFG_PATH)

OUT_PATH = './output'
OUT_PATH = os.path.abspath(OUT_PATH)

#DATASET_MEAN = path.sep.join([BASE_PATH, "mean.json"])

TRAIN_REC = [path.sep.join([BASE_PATH, "train6-00000-of-00005"]),
             path.sep.join([BASE_PATH, "train6-00001-of-00005"]),
             path.sep.join([BASE_PATH, "train6-00002-of-00005"]),
             path.sep.join([BASE_PATH, "train6-00003-of-00005"]),
             path.sep.join([BASE_PATH, "train6-00004-of-00005"]),]
VAL_REC = [path.sep.join([BASE_PATH, "validation6-00000-of-00005"]),
           path.sep.join([BASE_PATH, "validation6-00001-of-00005"]),
           path.sep.join([BASE_PATH, "validation6-00002-of-00005"]),
           path.sep.join([BASE_PATH, "validation6-00003-of-00005"]),
           path.sep.join([BASE_PATH, "validation6-00004-of-00005"])]

NUM_CLASSES = 3 


PREFIX = 'train_17'
NET_SCOPE = 'SpoofLdConvNet'
Classifier = "new" # old, new
Mask_Cmp_Loss = True

# input parameter
Image_Height = 224
Image_Width = 224
Channel_Number = 2
Draw_Line = False
Fill_Value = 127
Fill_Ratio_H = 0.4
Fill_Ratio_W = 0.35



# training parameters
BUFFER_SIZE = 10000
BATCH_SIZE = 128
DEVICE_IDS = "0,"
NUM_DEVICES = len(DEVICE_IDS.split(","))
NUM_EPOCH = 40
Update_Interval = 10


# resnet parameter
Resnet_stages = (1,1,1,1,1)
Resnet_filters = (8,8,8,16,16,8)
Resnet_input_layer = 'v2' # specify input layer version number
Resnet_residue_module = 'v3' # specify residue module version number
Resnet_use_se = False # use squeeze_excite network
Resnet_bottle_neck = False # use bottle_neck structure

# classifer parameter
Clf_layer_unit = [16,]
Drop_rate = 0.9

#
Regularizer = l2(5e-4)             
Initializer = tf.contrib.layers.xavier_initializer(uniform=False)
Activation = tf.nn.relu

# model parameter
Embedding_size = 64


# optimizer parameter
#Opt_name = 'SGD' # SGD, Adam
Opt_lr = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
Opt_lr_steps = [15000, 25000, 30000, 35000]
Opt_momentum = 0.9
#Opt_weight_decay = 0.0005
#Opt_rescale_grad = 1.0

# loss type
LOSS_TYPE = 'softmax' # 'arc' or 'softmax'
Arc_margin_scale = 64.0 # for 'arc' only
Arc_margin_angle = 0.25 # for 'arc' only
Loss_clf_weight = 0.5
Loss_cmp_weight = 1.0


# augmentation
Augm_rotate = 0.3 # set to 0 to disable
Augm_brightness = 32.0/255 # set to 0 to disable
Augm_saturation = 0.5 # set to 0 to disable
Augm_hue = 0.5 # set to 0 to disable
Augm_contrast = 0.5 # set to 0 to disable
Augm_crop = 0 # set to 0 to disable
