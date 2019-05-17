# evaluate and compare face-align models
from __future__ import print_function
import time
import numpy as np
import tensorflow as tf
import cv2
from operator import itemgetter

class tfMtcnnFaceDet():
    def __init__(   self,
                    model_path='~/mymodels/tf_mtcnn/frozen_model_mtcnn_all.pb'):
        print('model_path', model_path)

        self.model_path = model_path
        self.PNet_threshold = 0.8
        self.RNet_threshold = 0.9
        self.ONet_threshold = 0.95
        self.input_size = 786432.0
        self.scale_filter = 0.04
        self.min_face_size = 90

        self.alpha = 0.0078125
        self.mean = -127.5*self.alpha
        self.scaleFactor = 0.709

        self.graph = tf.Graph()
        with self.graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            self.pnet_input = self.graph.get_tensor_by_name('pnet/input:0')
            # Each box represents a part of the image where a particular object was detected.
            self.pnet_output = self.graph.get_tensor_by_name('pnet/conv4-2/BiasAdd:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            self.pnet_score = self.graph.get_tensor_by_name('pnet/prob1:0')

            self.rnet_input = self.graph.get_tensor_by_name('rnet/input:0')
            self.rnet_output = self.graph.get_tensor_by_name('rnet/conv5-2/BiasAdd:0')
            self.rnet_score = self.graph.get_tensor_by_name('rnet/prob1:0')

            self.onet_input = self.graph.get_tensor_by_name('onet/input:0')
            self.onet_output = self.graph.get_tensor_by_name('onet/conv6-2/BiasAdd:0')
            self.onet_score = self.graph.get_tensor_by_name('onet/prob1:0')

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(graph=self.graph, config=config)



    # image should be in RGB order
    def getModelOutput(self, image, th=0.5): # image should be single color image        
        h, w, c = image.shape
        img_area = h*w
        maxImgShrink = (self.input_size/img_area)**0.5
        shrink_size = min(maxImgShrink, 1.0)

        im = cv2.resize(image, (int(w*shrink_size), int(h*shrink_size)))
        im = im.astype(np.float32)
        im = im*self.alpha+self.mean
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        h_new, w_new, c_new = im.shape

        win_list = self.cal_pyramid_list(h_new, w_new, self.min_face_size*shrink_size, self.scaleFactor)

        #print(h_new, w_new, self.min_face_size*shrink_size, self.scaleFactor)
        #print('win_list')
        #print(win_list)

        # P_NET
        pnet_bbox = []
        for wl in win_list:
            #print(wl)
            tmp_pnet_bbox = self.run_pnet(im, wl)
            #print('pnet_bbox')
            #print(pnet_bbox)
            pnet_bbox += tmp_pnet_bbox

        #print('pnet_bbox len: ', len(pnet_bbox))
        pnet_bbox = self.process_boxes(pnet_bbox, h_new, w_new)
        #print('pnet_bbox len: ', len(pnet_bbox))
        #print(pnet_bbox)

        if len(pnet_bbox)==0: return []

        
        # R_NET
        rnet_bbox = self.run_rnet(im, pnet_bbox)

        #print('rnet_bbox len: ', len(rnet_bbox))
        #print(rnet_bbox)

        if len(rnet_bbox)==0: return []


        # O_NET
        onet_bbox = self.run_onet(im, rnet_bbox)

        #print('onet_bbox len: ', len(onet_bbox))
        #print(onet_bbox)

        if len(onet_bbox)==0: return []

        onet_bbox = self.regress_boxes(onet_bbox)
        faces_bbox = self.nms_boxes(onet_bbox, 0.7, "min")

        faces = []
        for i in range(len(faces_bbox)):
            # face: y0, x0, y1, x1
            faces += [[int(faces_bbox[i]['y0']/shrink_size),int(faces_bbox[i]['x0']/shrink_size),int(faces_bbox[i]['y1']/shrink_size),int(faces_bbox[i]['x1']/shrink_size)]]
        
        #print('faces len: ', len(onet_bbox))
        #print(faces)

        return faces
        

    def run_onet(self, pImg, r_bbox):
        ch = 3
        h = 48
        w = 48
        onet_bbox = []
        batch_size = len(r_bbox)

        im_buf = np.zeros([batch_size, h, w, ch])

        for i in range(batch_size):
            py1 = int(r_bbox[i]['py1'])
            py0 = int(r_bbox[i]['py0'])
            px1 = int(r_bbox[i]['px1'])
            px0 = int(r_bbox[i]['px0'])
            y1 = int(r_bbox[i]['y1'])
            y0 = int(r_bbox[i]['y0'])
            x1 = int(r_bbox[i]['x1'])
            x0 = int(r_bbox[i]['x0'])
            tmp = pImg[py0:py1, px0:px1, :].copy() # np.zeros([py1-py0+1, px1-px0+1, ch])

            pad_top = int(np.abs(py0 - y0))
            pad_left = int(np.abs(px0 - x0))
            pad_bottom = int(np.abs(py1 - y1))
            pad_right = int(np.abs(px1 - x1))
            tmp = cv2.copyMakeBorder(tmp,pad_top,pad_bottom,pad_left,pad_right,cv2.BORDER_CONSTANT,None,0.0)

            im_buf[i,...] = cv2.resize(tmp, (h,w))

        with self.graph.as_default():
            (o_bbox, o_score) = self.sess.run(
                                                [self.onet_output, self.onet_score],
                                                 feed_dict={self.onet_input: im_buf})
            #print('o_bbox: ', o_bbox.shape)
            #print('o_score: ', o_score.shape)
            
            for j in range(batch_size):
                if o_score[j,1] > self.ONet_threshold:
                    out_box = r_bbox[j].copy()
                    out_box['score'] = o_score[j,1]
                    out_box['regress'][0] = o_bbox[j,0]
                    out_box['regress'][1] = o_bbox[j,1]
                    out_box['regress'][2] = o_bbox[j,2]                    
                    out_box['regress'][3] = o_bbox[j,3]

                    onet_bbox += [out_box]

            return onet_bbox

    def run_rnet(self, pImg, p_bbox):
        ch = 3
        h = 24
        w = 24
        rnet_bbox = []
        batch_size = len(p_bbox)

        im_buf = np.zeros([batch_size, h, w, ch])

        for i in range(batch_size):
            py1 = int(p_bbox[i]['py1'])
            py0 = int(p_bbox[i]['py0'])
            px1 = int(p_bbox[i]['px1'])
            px0 = int(p_bbox[i]['px0'])
            y1 = int(p_bbox[i]['y1'])
            y0 = int(p_bbox[i]['y0'])
            x1 = int(p_bbox[i]['x1'])
            x0 = int(p_bbox[i]['x0'])
            tmp = pImg[py0:py1, px0:px1, :].copy() # np.zeros([py1-py0+1, px1-px0+1, ch])

            pad_top = int(np.abs(py0 - y0))
            pad_left = int(np.abs(px0 - x0))
            pad_bottom = int(np.abs(py1 - y1))
            pad_right = int(np.abs(px1 - x1))
            tmp = cv2.copyMakeBorder(tmp,pad_top,pad_bottom,pad_left,pad_right,cv2.BORDER_CONSTANT,None,0.0)

            im_buf[i,...] = cv2.resize(tmp, (h,w))

        with self.graph.as_default():
            (r_bbox, r_score) = self.sess.run(
                                                [self.rnet_output, self.rnet_score],
                                                 feed_dict={self.rnet_input: im_buf})
            #print('r_bbox: ', r_bbox.shape)
            #print('r_score: ', r_score.shape)
            
            for j in range(batch_size):
                if r_score[j,1] > self.RNet_threshold:
                    out_box = p_bbox[j].copy()
                    out_box['score'] = r_score[j,1]
                    out_box['regress'][0] = r_bbox[j,0]
                    out_box['regress'][1] = r_bbox[j,1]
                    out_box['regress'][2] = r_bbox[j,2]                    
                    out_box['regress'][3] = r_bbox[j,3]

                    rnet_bbox += [out_box]

            return rnet_bbox
        
    def run_pnet(self, pImg, win):
        pnet_bbox = []
        im = cv2.resize(pImg, (win['w'], win['h']))
        im = np.expand_dims(im, axis=0)
        with self.graph.as_default():
            (p_bbox, p_score) = self.sess.run(
                                                [self.pnet_output, self.pnet_score],
                                                 feed_dict={self.pnet_input: im})
            #print(p_bbox.shape)
            #print('p_score: ', p_score.shape)

            candidate_boxes = self.generate_bounding_box_tf(p_score, p_bbox, win['scale'])

            box_list = self.nms_boxes(candidate_boxes)            

            return box_list
        

    def cal_pyramid_list(self, height, width, min_size, factor):
        win_list = []
        min_side = min(height, width)
        m = 12.0 / min_size

        min_side=min_side*m;
        cur_scale=1.0;

        while min_side >= 12:
            scale=m*cur_scale;
            cur_scale=cur_scale *factor;
            min_side *= factor;

            hs = int(np.ceil(height*scale))
            ws = int(np.ceil(width*scale))
            win_list += [{'h':hs, 'w':ws, 'scale':scale}]

        return win_list

    def generate_bounding_box_tf(self, score_data, bbox_data, scale, transposed=False):
        candidate_boxes = []
        stride = 2
        cellSize = 12
        _, img_h, img_w, _ = bbox_data.shape

        for y in range(img_h):
            for x in range(img_w):
                score = score_data[0,y,x,1]
                if score > self.PNet_threshold:
                    top_x = int((x*stride+1)/scale)*1.0
                    top_y = int((y*stride+1)/scale)*1.0
                    btm_x = int((x*stride + cellSize) / scale)*1.0
                    btm_y = int((y*stride + cellSize) / scale)*1.0

                    box = {'x0':top_x, 'y0': top_y, 'x1': btm_x, 'y1': btm_y, 'score': score}

                    if transposed:
                        box['regress'] = [bbox_data[0,y,x,1], bbox_data[0,y,x,0],bbox_data[0,y,x,3], bbox_data[0,y,x,2]]
                    else:
                        box['regress'] = [bbox_data[0,y,x,0], bbox_data[0,y,x,1],bbox_data[0,y,x,2], bbox_data[0,y,x,3]]

                    candidate_boxes += [box]

        return candidate_boxes

    def nms_boxes(self, candidate_bbox, threshold=0.5, nms_type="union"):
        output_box = []
        sorted_bbox = sorted(candidate_bbox, key=itemgetter('score'))
        sorted_bbox = sorted_bbox[::-1]

        #print('sorted_bbox')
        #print(sorted_bbox)


        box_num = len(sorted_bbox)
        merged = np.zeros(box_num)

        for i in range(box_num):
            if merged[i]!=0.0:
                continue

            #print('hello')
            output_box += [sorted_bbox[i]]

            h0 = 1.0*(sorted_bbox[i]['y1']-sorted_bbox[i]['y0']+1)
            w0 = 1.0*(sorted_bbox[i]['x1']-sorted_bbox[i]['x0']+1)
            area0 = h0*w0

            for j in range(i+1, box_num):
                if merged[j]!=0.0:
                    continue

                inner_x0=max(sorted_bbox[i]['x0'],sorted_bbox[j]['x0'])
                inner_y0=max(sorted_bbox[i]['y0'],sorted_bbox[j]['y0'])
                inner_x1=min(sorted_bbox[i]['x1'],sorted_bbox[j]['x1'])
                inner_y1=min(sorted_bbox[i]['y1'],sorted_bbox[j]['y1'])

                inner_h=inner_y1-inner_y0+1.0
                inner_w=inner_x1-inner_x0+1.0


                if inner_h<=0 or inner_w<=0:
                    continue

                inner_area=inner_h*inner_w*1.0

                h1=sorted_bbox[j]['y1']-sorted_bbox[j]['y0']+1
                w1=sorted_bbox[j]['x1']-sorted_bbox[j]['x0']+1

                area1=h1*w1*1.0

                if nms_type=="union":
                    score=inner_area/(area0+area1-inner_area)
                else:
                    score=inner_area/min(area0,area1)

                if score>threshold:
                    merged[j]=1.0
        return output_box

    def process_boxes(self, input_bbox, img_h, img_w):
        #print('initial:')
        #print(input_bbox)
        rects = self.nms_boxes(input_bbox,0.7)
        #print('after nms:')
        #print(rects)

        rects = self.regress_boxes(rects)
        #print('after regress:')
        #print(rects)

        rects = self.square_boxes(rects)
        #print('after square:')
        #print(rects)

        rects = self.padding(img_h,img_w,rects)
        #print('after padding:')
        #print(rects)

        return rects

    def regress_boxes(self, rects):

        for i in range(len(rects)):
            box = rects[i]
            h = box['y1'] - box['y0'] + 1.0
            w = box['x1'] - box['x0'] + 1.0

            rects[i]['x0']=box['x0']+w*box['regress'][0];
            rects[i]['y0']=box['y0']+h*box['regress'][1];
            rects[i]['x1']=box['x1']+w*box['regress'][2];
            rects[i]['y1']=box['y1']+h*box['regress'][3];

        return rects

    def square_boxes(self, rects):
        for i in range(len(rects)):
            h=rects[i]['y1']-rects[i]['y0']+1.0
            w=rects[i]['x1']-rects[i]['x0']+1.0

            l=max(h,w)

            rects[i]['x0']=rects[i]['x0']+(w-l)*0.5
            rects[i]['y0']=rects[i]['y0']+(h-l)*0.5
            rects[i]['x1']=rects[i]['x0']+l-1.0
            rects[i]['y1']=rects[i]['y0']+l-1.0

        return rects


    def padding(self, img_h, img_w, rects):
        for i in range(len(rects)):
            rects[i]['px0']=max(rects[i]['x0'],1.0)
            rects[i]['py0']=max(rects[i]['y0'],1.0)
            rects[i]['px1']=min(rects[i]['x1'],img_w*1.0)
            rects[i]['py1']=min(rects[i]['y1'],img_h*1.0)
        return rects


if __name__ == '__main__':
    mtcnnFaceDet = tfMtcnnFaceDet()

    img = cv2.imread('/home/macul/Projects/ego_prj/detection_mtcnn_tf/test.jpg')

    mtcnnFaceDet.getModelOutput(img)
