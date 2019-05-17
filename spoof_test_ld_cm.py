###################### pointgrey python repo ##############################
# https://github.com/jordens/pyflycapture2/tree/master/src

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
import argparse
from SpoofLineConvTf import SpoofLineConvTf
from locallib.utils.tfSpoofLd import TfSpoofLd

def prepare_img(img, bbox, crop_scale_to_bbox=2.2, tgt_bbox_width=90):
    img_h, img_w, _ = img.shape
    bbx_x, bbx_y, x2, y2 = bbox
    bbx_w = x2 - bbx_x + 1
    bbx_h = y2 - bbx_y + 1

    img_cm = img[bbx_y:y2,bbx_x:x2,:].copy()
    img_cm = cv2.cvtColor(img_cm, cv2.COLOR_BGR2RGB)
    #print('orig_img_shape: ', img.shape)

    crp_w = int(bbx_w*crop_scale_to_bbox)
    crp_h = int(bbx_h*crop_scale_to_bbox)
    crp_x1 = max(0, int(bbx_x-(crp_w-bbx_w)/2.0))
    crp_y1 = max(0, int(bbx_y-(crp_h-bbx_h)/2.0))
    crp_x2 = min(crp_x1+crp_w-1, img_w-1)
    crp_y2 = min(crp_y1+crp_h-1, img_h-1)

    #print('crop_x: ', crp_x1)
    #print('crop_y: ', crp_y1)
    #print('crop_w: ', crp_w)
    #print('crop_h: ', crp_h)

    img_ld = img[crp_y1:crp_y2+1,crp_x1:crp_x2+1,:].copy()
    img_h, img_w, _ = img_ld.shape
    #print('crop_img_shape: ', img.shape)

    tmpScale = 1.0*tgt_bbox_width/bbx_w
    bbx_x = bbx_x - crp_x1
    bbx_y = bbx_y - crp_y1

    #print('crop_bbx_x: ', bbx_x)
    #print('crop_bbx_y: ', bbx_y)

    new_img_w = int(tmpScale*img_w)
    new_img_h = int(tmpScale*img_h)
    img_ld = cv2.resize(img_ld,(new_img_w,new_img_h))
    img_h, img_w, _ = img_ld.shape

    #print('resized_img_shape: ', img.shape)

    bbx_x = int(bbx_x*tmpScale)
    bbx_y = int(bbx_y*tmpScale)          
    bbx_w = int(bbx_w*tmpScale)
    bbx_h = int(bbx_h*tmpScale)

    #print('resize_bbx_x: ', bbx_x)
    #print('resize_bbx_y: ', bbx_y)
    #print('resize_bbx_width: ', bbx_w)
    #print('resize_bbx_height: ', bbx_h)

    # crop image
    

    bbx_x1 = bbx_x
    bbx_x2 = bbx_x1 + bbx_w -1
    bbx_y1 = bbx_y
    bbx_y2 = bbx_y1 + bbx_h -1

    return img_ld, [bbx_x1, bbx_y1, bbx_x2, bbx_y2], img_cm

def prepare_img1(img, bbox, crop_scale_to_bbox=2.5, crop_square=True):
    img_h, img_w, _ = img.shape
    bbx_x, bbx_y, x2, y2 = bbox
    bbx_w = x2 - bbx_x + 1
    bbx_h = y2 - bbx_y + 1
    
    #print('orig_img_shape: ', img.shape)

    crp_w = int(bbx_w*crop_scale_to_bbox)
    crp_h = int(bbx_h*crop_scale_to_bbox)

    if crop_square:
        crp_w = max(crp_w, crp_h)
        crp_h = crp_w

    img_crop = np.zeros([crp_h, crp_w, 3]).astype(np.uint8)
    img_crop[:,:,1] = 255 # make empty portion green to differentiate with black

    
    crp_x1 = max(0, int(bbx_x-(crp_w-bbx_w)/2.0))
    crp_y1 = max(0, int(bbx_y-(crp_h-bbx_h)/2.0))
    crp_x2 = min(int(bbx_x-(crp_w-bbx_w)/2.0)+crp_w-1, img_w-1)
    crp_y2 = min(int(bbx_y-(crp_h-bbx_h)/2.0)+crp_h-1, img_h-1)

    delta_x1 = -min(0, int(bbx_x-(crp_w-bbx_w)/2.0))
    delta_y1 = -min(0, int(bbx_y-(crp_h-bbx_h)/2.0))

    img_crop[delta_y1:delta_y1+crp_y2-crp_y1+1, delta_x1:delta_x1+crp_x2-crp_x1+1] = img[crp_y1:crp_y2+1,crp_x1:crp_x2+1,:].copy()

    bbx_x1 = bbx_x - crp_x1 + delta_x1
    bbx_y1 = bbx_y - crp_y1 + delta_y1
    bbx_x2 = bbx_x1 + bbx_w -1
    bbx_y2 = bbx_y1 + bbx_h -1

    #img_crop=cv2.rectangle(img_crop, (bbx_x1, bbx_y1), (bbx_x2, bbx_y2), (0,255,0), 3)
    
    return img_crop, [bbx_x1, bbx_y1, bbx_x2, bbx_y2]

ap = argparse.ArgumentParser()
ap.add_argument("-mf", "--model-folder", required=True, help="model folder path")
ap.add_argument("-c", "--ckpt-num", required=False, help="checkpoint number")
ap.add_argument("-mpb", "--model-pb", required=False, default="", help="pb model")
#ap.add_argument("-p", "--prefix", required=True, help="name of model prefix")
ap.add_argument("-v", "--video-path", required=True, help="input video full path")
ap.add_argument("-t", "--threshold", required=True, type=float, help="threshold of color moment model")
ap.add_argument("-tf", "--threshold-false", required=True, type=float, help="threshold for false detection of color moment model")
ap.add_argument("-tfl", "--threshold-false-line", required=True, type=float, help="threshold for false detection of line detection model")
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

pb_path = '~/mymodels/tf_spoof_ld/clf.pb' # for line detection

SpoofVal = SpoofLineConvTf(config)

if args['model_pb'] == "":
    ckpt_num = args['ckpt_num']
    sess_cm, _, pred = SpoofVal.deploy_net(ckpt_num)
else:
    sess_cm, img_ph, pred, _, _ = SpoofVal.load_pb(args['model_pb'])


tfSpoofLd = TfSpoofLd()
sess_ld, lines_in, pred_out, _, _ = tfSpoofLd.load_pb(pb_path)

useMtcnn = args['use_mtcnn']>0
if useMtcnn:
    faceDet = tfMtcnnFaceDet(args['face_det_model']+'/tf_mtcnn/frozen_model_mtcnn_all.pb')
else:
    faceDet = tfFaceDet(args['face_det_model']+'/tf_face_detection/frozen_inference_graph_face.pb')


'''
# install flycapture2
pip install --user Cython
pip install git+https://github.com/jordens/pyflycapture2.git

# for pointgrey
import flycapture2 as fc2
c = fc2.Context()
print(c.get_num_of_cameras())
c.connect(*c.get_camera_from_index(0))
print(c.get_camera_info())
c.set_video_mode_and_frame_rate(fc2.VIDEOMODE_1280x960Y16,
            fc2.FRAMERATE_7_5)

m, f = c.get_video_mode_and_frame_rate()

print m, f
print c.get_video_mode_and_frame_rate_info(m, f)
print c.get_property_info(fc2.FRAME_RATE)
p = c.get_property(fc2.FRAME_RATE)
print p
c.set_property(**p)
c.start_capture()
im = fc2.Image()

fc_img=c.retrieve_buffer(im)
img=fc_img.convert_to(fc2.PIXEL_FORMAT_RGB8)
img=np.array(img)

print [np.array(c.retrieve_buffer(im)).sum() for i in range(80)]
a = np.array(im)
print a.shape, a.base
c.stop_capture()
c.disconnect()
'''


if args['video_path']=='pge':
    import flycapture2 as fc2
    cap = fc2.Context()
    cap.connect(*cap.get_camera_from_index(0))
    cap.start_capture()
    img_buf = fc2.Image()
else:
    cap = cv2.VideoCapture(args['video_path'])


line_count = 0

currentFrame = 0
while True:
    # Create a pipeline object. This object configures the streaming camera and owns it's handle
    if args['video_path']=='pge':
        fc_img=cap.retrieve_buffer(img_buf)
        frame=fc_img.convert_to(fc2.PIXEL_FORMAT_RGB8)
        frame=np.array(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    else:
        ret, frame = cap.read()

    if type(frame) == type(None):
        break

    [h, w] = frame.shape[:2]
    if useMtcnn:
        faces = faceDet.getModelOutput(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)        
        faces = faceDet.getModelOutput(frame)

    for face in faces:
        if useMtcnn:
            bbox = [max(face[1],0), max(face[0],0), min(face[3],w-1), min(face[2],h-1)]
            p1 = (bbox[0],bbox[1])
            p2 = (bbox[2],bbox[3])     
        else:
            bbox = [int(face[1]*w), int(face[0]*h), int(face[3]*w), int(face[2]*h)]
            p1 = (int(face[1]*w),int(face[0]*h))
            p2 = (int(face[3]*w),int(face[2]*h))                

        if (p2[0]-p1[0]) < args['min_size'] or (p2[1]-p1[1]) < args['min_size']:
            continue

        face_ld, bbox_ld, _ = prepare_img(frame, bbox)
        face_cm = SpoofVal.preprocess_img(frame, bbox, crop_scale_to_bbox=args['crop_scale'])

        rst_ld = tfSpoofLd.eval(sess_ld, lines_in, pred_out, img=face_ld, bbox=bbox_ld)

        rst_ld = np.squeeze(rst_ld[0])        

        spoof_ld_rst = rst_ld[1]>args['threshold_false_line'] # 0: no line, 1: got line

        if args['model_pb'] == "":
            rst = SpoofVal.eval(face_cm, sess_cm, pred)
        else:
            rst = SpoofVal.eval_pb(face_cm, sess_cm, img_ph, pred)
            
        rst = np.squeeze(rst[0])
        idx = np.argmax(rst)
        #print("spoof: ", idx, rst)
        #print('threshold: ', args['threshold'])        

        spoof_cm_rst = -1 # -1: ignore, 0: real, 1: fake
        if idx == 0:
            if rst[0]>args['threshold']:
                spoof_cm_rst = 0
            elif rst[0]<(1.0-args['threshold_false']):
                spoof_cm_rst = 1
        else:
            if rst[0]<(1.0-args['threshold_false']):
                spoof_cm_rst = 1
        
        if spoof_ld_rst: 
            frame=cv2.rectangle(frame, p1, p2, (255,0,0), 3)

            line_count += 1

        else:
            if spoof_cm_rst == 0:
                frame=cv2.rectangle(frame, p1, p2, (0,255,0), 3)
            elif spoof_cm_rst == 1:
                frame=cv2.rectangle(frame, p1, p2, (255,0,0), 3)
            else:
                pass

        print('line: ', rst_ld[1], 'cm: ', rst[0], 'line cnt: ', line_count)
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

#f_log.close()

# python -m spoof_test_ld_cm1 -mf /home/macul/libraries/mk_utils/tf_spoof/dgxout/train_2 -mpb /home/macul/libraries/mk_utils/tf_spoof/dgxout/train_2/train_2_7.pb -v /home/macul/Screencast_2019-05-06_10-28-48.mp4  -t 0.95 -tf 0.96 -um 1 -cs 2.0 -tfl 1.0
# python -m spoof_test_ld_cm1 -mf /home/macul/libraries/mk_utils/tf_spoof/dgxout/train_2  -v /home/macul/Screencast_2019-05-06_10-28-48.mp4  -t 0.95 -tf 0.96 -um 1 -cs 2.0 -tfl 1.0 -c 7
