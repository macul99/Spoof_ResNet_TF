import tensorflow as tf
import numpy as np
import os
import argparse
from os.path import isdir
from os import mkdir
from shutil import copyfile
import time
from PIL import Image
import pickle
from SpoofLineConvTf import SpoofLineConvTf
from config import net_config_ld_conv as config

os.environ["CUDA_VISIBLE_DEVICES"] = config.DEVICE_IDS

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--start-epoch", required=False, type=int, default=-1, help="start epoch")

    args = vars(ap.parse_args())
    return args

if __name__ == '__main__':
    args = parse_args()
    SpoofTrain = SpoofLineConvTf(config)

    copyfile(config.CFG_PATH+'/net_config_ld_conv.py', os.path.sep.join([SpoofTrain.checkpointsPath,'net_config_ld_conv_{}.py'.format(0)]))
    copyfile(os.path.abspath('./SpoofLineConvTf.py'), os.path.sep.join([SpoofTrain.checkpointsPath,'SpoofLineConvTf_{}.py'.format(0)]))

    SpoofTrain.train_net(args['start_epoch'])
