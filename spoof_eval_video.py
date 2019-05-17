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
from locallib.utils.tfFaceDet import tfFaceDet
from locallib.utils.tfMtcnnFaceDet import tfMtcnnFaceDet
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,"
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-mf", "--model-folder", required=True, help="model folder path")
ap.add_argument("-c", "--ckpt-num", required=False, help="checkpoint number")
ap.add_argument("-mpb", "--model-pb", required=False, default="", help="pb model")
#ap.add_argument("-p", "--prefix", required=True, help="name of model prefix")
ap.add_argument("-v", "--video-path", required=True, help="input video full path")
ap.add_argument("-t", "--threshold", required=True, type=float, help="threshold")
ap.add_argument("-tf", "--threshold-false", required=True, type=float, help="threshold for false detection")
ap.add_argument("-s", "--min-size", required=False, type=int, default=100, help="threshold")
#ap.add_argument("-of", "--out-folder", required=False, default="output1", help="output folder name")
ap.add_argument("-um", "--use-mtcnn", required=False, type=int, default=0, help="use mtcnn face detector")
ap.add_argument("-fdm", "--face-det-model", required=False, default='../mymodels', help="use mtcnn face detector")
ap.add_argument("-cs", "--crop-scale", required=False, type=float, default=2.5, help="crop scale")

args = vars(ap.parse_args())

args['model_folder'] = os.path.abspath(args['model_folder'])

prefix = args['model_folder'][args['model_folder'].rfind('/')+1:]
out_folder = args['model_folder'][0:args['model_folder'].rfind('/')]

import sys
sys.path.append(args['model_folder'])
import net_config_ld_conv_0 as config
config.OUT_PATH = out_folder

from SpoofLineConvTf_0 import SpoofLineConvTf
#from spoofing_ld_conv.SpoofLineConvTf import SpoofLineConvTf

SpoofVal = SpoofLineConvTf(config)

if args['model_pb'] == "":
    ckpt_num = args['ckpt_num']
    sess, _, pred = SpoofVal.deploy_net(ckpt_num)
else:
    sess, img_ph, pred, _, _ = SpoofVal.load_pb(args['model_pb'])

useMtcnn = args['use_mtcnn']>0
if useMtcnn:
    faceDet = tfMtcnnFaceDet(args['face_det_model']+'/tf_mtcnn/frozen_model_mtcnn_all.pb')
else:
    faceDet = tfFaceDet(args['face_det_model']+'/tf_face_detection/frozen_inference_graph_face.pb')

cap = cv2.VideoCapture(args['video_path'])


currentFrame = 0
face_cnt = 0
real_cnt = 0
fake_cnt = 0
while True:
    # Create a pipeline object. This object configures the streaming camera and owns it's handle
    ret, frame = cap.read()

    if type(frame) == type(None):
        break

    if useMtcnn:
        faces = faceDet.getModelOutput(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = faceDet.getModelOutput(frame)

    [h, w] = frame.shape[:2]

    for face in faces:  # face [ymin, xmin, ymax, xmax]  

        if useMtcnn:
            bbox = [max(face[1],0), max(face[0],0), min(face[3],w-1), min(face[2],h-1)]
            p1 = (bbox[0],bbox[1])
            p2 = (bbox[2],bbox[3])
            #p1 = (face[1],face[0])
            #p2 = (face[3],face[2])     
        else:
            p1 = (int(face[1]*w),int(face[0]*h))
            p2 = (int(face[3]*w),int(face[2]*h))  

        if (p2[0]-p1[0]) < args['min_size'] or (p2[1]-p1[1]) < args['min_size']:
            continue

        face_crop = SpoofVal.preprocess_img(frame, [p1[0], p1[1], p2[0], p2[1]], crop_scale_to_bbox=args['crop_scale'])

        if args['model_pb'] == "":
            rst = SpoofVal.eval(face_crop, sess, pred)
        else:
            rst = SpoofVal.eval_pb(face_crop, sess, img_ph, pred)
        rst = np.squeeze(rst[0])
        idx = np.argmax(rst)
        print("spoof: ", idx, rst)
        print('threshold: ', args['threshold'])

        face_cnt += 1
        if idx == 0:
            if rst[0]>args['threshold']:
                frame=cv2.rectangle(frame, p1, p2, (0,255,0), 3)
                real_cnt += 1
            elif rst[0]<(1.0-args['threshold_false']):
                frame=cv2.rectangle(frame, p1, p2, (255,0,0), 3)
                fake_cnt += 1
        else:
            if rst[0]<(1.0-args['threshold_false']):
                frame=cv2.rectangle(frame, p1, p2, (255,0,0), 3)
                fake_cnt += 1
        
        #cv2.imshow('frame',face_crop)
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break

        #lbp(gray, p=8, split=3)
        '''
        gray = cv2.cvtColor(crop_color, cv2.COLOR_BGR2GRAY)
        lbp = feature.local_binary_pattern(gray, 8, 2, method="nri_uniform")
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 59))
        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)
        print('hist:', hist.shape)
        plt.hist(lbp.ravel(),bins=range(59))
        plt.show()
        '''

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
#cap1.release()
cv2.destroyAllWindows()
    #plt.subplot(122)
    #plt.imshow(crop_color)
    #plt.show()

print('face cnt: {}, real cnt: {}, fake cnt: {}'.format(face_cnt, real_cnt, fake_cnt))
#f_log.close()


# python -m spoof_eval_video1 -c 7 -mf /home/macul/libraries/mk_utils/tf_spoof/dgxout/train_2 -v /home/macul/Screencast_2019-05-06_10-28-48.mp4  -t 0.95 -tf 0.96 -um 1 -cs 2.0
# python -m spoof_eval_video1 -mf /home/macul/libraries/mk_utils/tf_spoof/dgxout/train_2 -mpb /home/macul/libraries/mk_utils/tf_spoof/dgxout/train_2/train_2_7.pb -v /home/macul/Screencast_2019-05-06_10-28-48.mp4  -t 0.95 -tf 0.96 -um 1 -cs 2.0
