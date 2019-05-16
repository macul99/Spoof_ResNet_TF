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



def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--src", required=True, help="path to source folder")
    ap.add_argument("-d", "--dest", required=True, help="path to destination folder") # if it is a pickle file then call to_df(), if a folder then call to_txt() which will be used for IJBC eval()
    ap.add_argument("-c", "--checkpoints", required=True, type=int, help="checkpoints number") 


    args = vars(ap.parse_args())
    args.src = os.path.abspath(args.src)
    args.dest = os.path.abspath(args.dest)
    return args


if __name__ == '__main__':

	args = parse_args()
	sys.path.append(args.src)
	
	from SpoofLineConvTf_0 import SpoofLineConvTf
	import net_config_ld_conv_0 as config

	os.environ["CUDA_VISIBLE_DEVICES"] = config.DEVICE_IDS

    SpoofVal = SpoofLineConvTf(config)

    rst_val = SpoofVal.validate_net(args['checkpoints'], args['dest'])