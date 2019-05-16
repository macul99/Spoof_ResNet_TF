import tensorflow as tf
import numpy as np
import os
from os.path import isdir
from os import mkdir
from shutil import copyfile
import time
from PIL import Image
import pickle
import sys
import argparse


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--src", required=True, help="path to source folder")
    ap.add_argument("-c", "--checkpoints", required=True, type=int, help="checkpoints number") 


    args = vars(ap.parse_args())
    args['src'] = os.path.abspath(args['src'])
    return args


if __name__ == '__main__':

    args = parse_args()
    sys.path.append(args['src'])

    from SpoofLineConvTf_0 import SpoofLineConvTf
    import net_config_ld_conv_0 as config

    #config.OUT_PATH = args['src'][0:args['src'].rfind('/')]
    #config.DEVICE_IDS = "0,"

    os.environ["CUDA_VISIBLE_DEVICES"] = config.DEVICE_IDS

    SpoofVal = SpoofLineConvTf(config)

    rst_val = SpoofVal.save_pb(args['checkpoints'])

# python -m save_pb -s /home/macul/libraries/mk_utils/tf_spoof/dgxout/train_2 -c 7
