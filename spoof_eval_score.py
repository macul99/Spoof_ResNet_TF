import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,"
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
#from mpl_toolkits.mplot3d import Axes3D
#from mpl_toolkits import mplot3d
#from sklearn.cluster import KMeans
#from scipy.interpolate import Rbf, interp2d
#from scipy import fftpack, ndimage
from scipy.optimize import minimize
#from tfFaceDet import tfFaceDet
import time
import argparse
import pickle
from imutils import paths
from SpoofLineConvTf import SpoofLineConvTf

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--ckpt-num", required=True, help="checkpoint number")
ap.add_argument("-p", "--prefix", required=True, help="name of model prefix")
ap.add_argument("-o", "--output-path", required=True, help="output folder path")
args = vars(ap.parse_args())

ckpt_num = args['ckpt_num']
prefix = args['prefix']
import sys
sys.path.append('/home/macul/libraries/mk_utils/tf_spoof/output1/{}'.format(prefix))
import net_config_ld_conv_0 as config
config.BASE_PATH = '/media/macul/black/spoof_db/train_test_all/tfrecord'
config.CFG_PATH = '/home/macul/libraries/mk_utils/tf_spoof/config'
config.OUT_PATH = '/home/macul/libraries/mk_utils/tf_spoof/output1'
config.PREFIX = prefix

SpoofVal = SpoofLineConvTf(config)

sess, _, pred = SpoofVal.deploy_net(ckpt_num)

# for COLOR_MOMENT_S3 only
input_dic = {   'real'          : [ 
                                    '/media/macul/black/spoof_db/train_test_all/test5/real',
                                  ], 
                'image_attack'  : [ 
                                    '/media/macul/black/spoof_db/train_test_all/test5/image_attack',
                                  ], 
                'video_attack'  : [ 
                                    '/media/macul/black/spoof_db/train_test_all/test5/video_attack',
                                  ]
            }
output_dic = {}

time_elaps = []

for key in input_dic.keys():
    fd_list = input_dic[key]
    tmp_dic = {}
    for fd in fd_list:
        tmp_rst = []
        imagePaths = list(paths.list_images(fd))
        count = 0
        for imgP in imagePaths:
            count += 1            
            frame = cv2.imread(imgP)
            print(key, fd, count, imgP, frame.shape) 
            [h, w] = frame.shape[:2]
            start_time = time.time()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_fill = SpoofVal.fill_c(frame, target_size=SpoofVal.config.Image_Height, fill_value=SpoofVal.config.Fill_Value, 
                    ratio_h=SpoofVal.config.Fill_Ratio_H, ratio_w=SpoofVal.config.Fill_Ratio_W)
            #print('crop_flag: ', crop_flag)            
            rst = SpoofVal.eval(frame_fill, sess, pred)
            time_elaps += [time.time() - start_time]
            tmp_rst += [np.squeeze(rst[0])]
        tmp_dic[fd] = np.array(tmp_rst)
    output_dic[key] = tmp_dic


with open(args['output_path']+'/{}_{}.pkl'.format(prefix, ckpt_num),'wb') as f:
    pickle.dump(output_dic, f)

print(np.mean(time_elaps))
    #plt.subplot(122)
    #plt.imshow(crop_color)
    #plt.show()

#f_log.close()
