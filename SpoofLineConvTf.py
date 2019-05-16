import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1,"
import tensorflow as tf
import numpy as np
import argparse
import logging
import json
from os.path import isdir
from os import mkdir
from shutil import copyfile
import time
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.platform import gfile
from PIL import Image
import pickle
import cv2
from locallib.nn.tfnet.tfresnet import TfResNet
from locallib.nn.tfnet.tfdense import DenseNet
from locallib.nn.tfloss.tfloss import TfLosses


class SpoofLineConvTf():
    def __init__(self, config):
        self.config = config
        self.lineCfg = {
                        'enable_gaussian_blur': True,
                        'blur_kernal_size': 5,
                        'canny_low_threshold': 0,
                        'canny_high_threshold': 100,
                        'hough_rho': 1, # distance resolution in pixels of the Hough grid
                        'hough_threshold': 5, # minimum number of votes (intersections in Hough grid cell)
                        'hough_min_line_length': 25, # minimum number of pixels making up a line
                        'hough_max_line_gap': 4, # maximum gap in pixels between connectable line segments
                        'angle_limit': 15, # only lines which angle with horizontal or vertical lines are within this limit will be considered
                        'max_num_lines': 5, # the max number of lines for outside each boundary of the bounding boxes, totoal number of lines will be max_num_lines*4
                        }

        assert self.config.Channel_Number in [1, 2, 3] # 2 is for HSV of H and S

        input_img_channel = self.config.Channel_Number
        if self.config.Channel_Number == 2 and self.config.Draw_Line:
            input_img_channel = 3

        # declare placeholders for training
        self.labels1_ph = tf.placeholder(name='label1',shape=[None,], dtype=tf.int64)
        self.images1_ph = tf.placeholder(name='image1',shape=[None,self.config.Image_Height,self.config.Image_Width, input_img_channel], 
                                         dtype=tf.float32)
        self.labels2_ph = tf.placeholder(name='label2',shape=[None,], dtype=tf.int64)
        self.images2_ph = tf.placeholder(name='image2',shape=[None,self.config.Image_Height,self.config.Image_Width, input_img_channel], 
                                         dtype=tf.float32)
        # declare placeholders for validation
        self.labels_val_ph = tf.placeholder(name='label_val',shape=[None,], dtype=tf.int64)
        self.images_val_ph = tf.placeholder(name='image_val',shape=[None,self.config.Image_Height,self.config.Image_Width, input_img_channel], 
                                            dtype=tf.float32)
        # create output folder
        self.checkpointsPath = os.path.sep.join([self.config.OUT_PATH, self.config.PREFIX])
        if not isdir(self.checkpointsPath):
            mkdir(self.checkpointsPath)

    # if target_size==0, don't resize the image, otherwise, resize image to target_size*target_size
    def build_dataset(self, rec_path, batch_size, classes, target_size=0, fill_value=127, ratio_h=0.4, ratio_w=0.35, training=True):

        dataset = tf.data.TFRecordDataset(rec_path)
        dataset = dataset.map(lambda x: self.parse_function(x, classes))
        if training:
            dataset = dataset.shuffle(buffer_size=self.config.BUFFER_SIZE) # shuffle the whole dataset is better
        dataset = dataset.map(lambda *x: (self.img_fill_center(x[0], fill_value, ratio_h, ratio_w, target_size), x[1]))
        # remember to set batch size to 1 for dataset debug
        #dataset = dataset.map(lambda *x: self.feature_extraction(x[0], self.ft_extractor) + (x[1],)) # for dataset debug
        dataset = dataset.batch(batch_size)
        iterator = dataset.make_initializable_iterator()
        next_element = iterator.get_next()
        return iterator, next_element

    # build_classifer_new
    def build_classifer_new(self, input, label, training, bnEps=2e-5, bnMom=0.9, reuse=None, scope=None):
        with tf.variable_scope(scope, reuse=reuse):
            fltn = tf.keras.layers.Flatten(name='clf_fltn')(input)
            fc1 = tf.layers.dense(inputs=fltn, units=self.config.Embedding_size, kernel_initializer=self.config.Initializer, name='clf_fc1')
            out_embedding = tf.layers.batch_normalization(inputs=fc1, epsilon=bnEps, momentum=bnMom, training=training, name='out_embedding')

            if self.config.LOSS_TYPE=='arc':
                if training:
                    arc_margin = self.config.Arc_margin_angle
                else:
                    arc_margin = 0
                logit, inference_loss = TfLosses.arc_loss(embedding=out_embedding, labels=label, w_init=self.config.Initializer, 
                                                          out_num=self.config.NUM_CLASSES, s=self.config.Arc_margin_scale, 
                                                          m=arc_margin)
            else:
                logit, inference_loss = TfLosses.softmax_loss(embedding=out_embedding, labels=label, out_num=self.config.NUM_CLASSES,  
                                                                  act=self.config.Activation, reg=self.config.Regularizer, 
                                                                  init=self.config.Initializer)
            pred = tf.nn.softmax(logit, name='prediction') # output name: 'SpoofDenseNet/prediction'
            acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred, axis=1), label), dtype=tf.float32))

            return logit, inference_loss, pred, acc

    # build_classifer_old
    def build_classifer(self, input, label, training, bnEps=2e-5, bnMom=0.9, reuse=None, scope=None):
        with tf.variable_scope(scope, reuse=reuse):
            fltn = tf.keras.layers.Flatten(name='clf_fltn')(input)
            fc1 = tf.layers.dense(inputs=fltn, units=self.config.Embedding_size, kernel_initializer=self.config.Initializer, name='clf_fc1')
            out_embedding = tf.layers.batch_normalization(inputs=fc1, epsilon=bnEps, momentum=bnMom, training=training, name='out_embedding')

            out_1 = DenseNet.build_subnet(  out_embedding, self.config.Clf_layer_unit, training, self.config.Activation, self.config.Regularizer, self.config.Initializer, 
                                            args={'bn_momentum': bnMom,'drop_rate': self.config.Drop_rate}, scope=scope)

            if self.config.LOSS_TYPE=='arc':
                if training:
                    arc_margin = self.config.Arc_margin_angle
                else:
                    arc_margin = 0
                logit, inference_loss = TfLosses.arc_loss(embedding=out_1, labels=label, w_init=self.config.Initializer, 
                                                          out_num=self.config.NUM_CLASSES, s=self.config.Arc_margin_scale, 
                                                          m=arc_margin)
            else:
                logit, inference_loss = TfLosses.softmax_loss(embedding=out_1, labels=label, out_num=self.config.NUM_CLASSES,  
                                                                  act=self.config.Activation, reg=self.config.Regularizer, 
                                                                  init=self.config.Initializer)
            pred = tf.nn.softmax(logit, name='prediction') # output name: 'SpoofDenseNet/prediction'
            acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred, axis=1), label), dtype=tf.float32))

            return logit, inference_loss, pred, acc

    def build_net(self, training, bnEps=2e-5, bnMom=0.9, reuse=None):
        try:
            if self.config.Classifier == "new":
                build_classifier = self.build_classifer_new
            else:
                build_classifier = self.build_classifer
        except:
            build_classifier = self.build_classifer

        try:
            if self.config.Mask_Cmp_Loss:
                cmp_loss_mask_flag = True
            else:
                cmp_loss_mask_flag = False
        except:
            cmp_loss_mask_flag = False
            

        if training:
            image_ph = self.images1_ph
            label_ph = self.labels1_ph
        else:
            image_ph = self.images_val_ph
            label_ph = self.labels_val_ph

        with tf.variable_scope(self.config.NET_SCOPE, reuse=reuse):
            # build base network
            embedding1 = TfResNet.build_modules(    image_ph, self.config.Embedding_size, self.config.Resnet_stages, self.config.Resnet_filters, training=training, 
                                                    initializer=self.config.Initializer, reuse=None, res_ver=self.config.Resnet_residue_module, in_ver=self.config.Resnet_input_layer, 
                                                    bottle_neck=self.config.Resnet_bottle_neck, use_se=self.config.Resnet_use_se, bnEps=bnEps, bnMom=bnMom, scope='conv_stem')

            logit1, inference_loss1, pred1, acc1 = build_classifier(embedding1, label_ph, training, bnEps=bnEps, bnMom=bnMom, reuse=None, scope='clf_stem')

            if not training:
                return logit1, inference_loss1, pred1, acc1, embedding1

        with tf.variable_scope(self.config.NET_SCOPE, reuse=True):
            # build base network
            embedding2 = TfResNet.build_modules(   self.images2_ph, self.config.Embedding_size, self.config.Resnet_stages, self.config.Resnet_filters, training=training, 
                                                    initializer=self.config.Initializer, reuse=None, res_ver=self.config.Resnet_residue_module, in_ver=self.config.Resnet_input_layer, 
                                                    bottle_neck=self.config.Resnet_bottle_neck, use_se=self.config.Resnet_use_se, bnEps=bnEps, bnMom=bnMom, scope='conv_stem')

            logit2, inference_loss2, pred2, acc2 = build_classifier(embedding2, self.labels2_ph, training, bnEps=bnEps, bnMom=bnMom, reuse=None, scope='clf_stem')

        with tf.variable_scope(self.config.NET_SCOPE, reuse=reuse):
            emb_cat  = tf.concat([embedding1, embedding2], 3)
            emb_cat1 = tf.layers.conv2d( inputs=emb_cat, padding='same', kernel_size=(3, 3), strides=(2, 2), filters=32, use_bias=False, name='cmp_conv1')
            bn_cat1  = tf.layers.batch_normalization(inputs=emb_cat1, epsilon=bnEps, momentum=bnMom, training=training, name='cmp_bn1')
            act_cat1 = TfResNet.Act(data=bn_cat1, act_type="prelu", name='cmp_relu1')

            emb_cat2 = tf.layers.conv2d( inputs=act_cat1, padding='same', kernel_size=(3, 3), strides=(2, 2), filters=16, use_bias=False, name='cmp_conv2')
            bn_cat2  = tf.layers.batch_normalization(inputs=emb_cat2, epsilon=bnEps, momentum=bnMom, training=training, name='cmp_bn2')
            act_cat2 = TfResNet.Act(data=bn_cat2, act_type="prelu", name='cmp_relu2')

            emb_cat3 = tf.layers.conv2d( inputs=emb_cat2, padding='valid', kernel_size=(2, 2), strides=(1, 1), filters=1, use_bias=False, name='cmp_conv3')
            bn_cat3  = tf.layers.batch_normalization(inputs=emb_cat3, epsilon=bnEps, momentum=bnMom, training=training, name='cmp_bn3')
            act_cat3 = TfResNet.Act(data=bn_cat3, act_type="prelu", name='cmp_relu3') # the shape should be [batch,1,1,1]

            emb_cat4 = tf.squeeze(act_cat3)

            if cmp_loss_mask_flag:
                label_diff = tf.equal(self.labels1_ph, self.labels2_ph)
                mask1 = tf.math.logical_and(label_diff, tf.not_equal(self.labels1_ph,tf.zeros_like(self.labels1_ph)))
                mask2 = tf.math.logical_and(tf.math.logical_or(tf.equal(self.labels1_ph,tf.zeros_like(self.labels1_ph)), tf.equal(self.labels2_ph,tf.zeros_like(self.labels2_ph))), tf.not_equal(label_diff,True))
                label_diff = tf.cast(label_diff, tf.float32)
                mask = tf.cast(tf.math.logical_or(mask1,mask2), tf.float32)
                cmp_loss = tf.reduce_mean(tf.math.multiply(mask, emb_cat4-label_diff)**2)
            else:
                label_diff = tf.cast(tf.equal(self.labels1_ph, self.labels2_ph), tf.float32)
                cmp_loss = tf.reduce_mean((emb_cat4-label_diff)**2)

            total_loss = self.config.Loss_clf_weight*inference_loss1 + self.config.Loss_clf_weight*inference_loss2 + self.config.Loss_cmp_weight*cmp_loss

            return tf.concat([logit1, logit2],axis=0), tf.reduce_mean([inference_loss1, inference_loss2]), tf.concat([logit1, logit2],axis=0), tf.reduce_mean([acc1, acc2]), tf.concat([embedding1, embedding2],axis=0), cmp_loss, total_loss



        '''
        with tf.variable_scope(self.config.NET_SCOPE, reuse=reuse):

            emb_diff = tf.subtract(embedding1, embedding2)
            emb_diff = tf.abs(emb_diff)

            emb_diff_sub = DenseNet.build_subnet(   emb_diff, self.fc_layer_units, training=training, act=self.config.Activation, reg=self.config.Regularizer, 
                                                    init=self.config.Initializer, args={}, scope='diff_branch')

            label_diff = tf.cast(tf.equal(self.labels1_ph,self.labels2_ph), tf.int64)

            if self.config.LOSS_TYPE=='arc':
                logit, inference_loss = TfLosses.arc_loss(embedding=emb_diff_sub, labels=label_diff, w_init=self.config.Initializer, 
                                                          out_num=2, s=self.config.Arc_margin_scale, 
                                                          m=arc_margin)
            else:
                logit, inference_loss = TfLosses.softmax_loss(embedding=emb_diff_sub, labels=label_diff, out_num=2,  
                                                              act=self.config.Activation, reg=self.config.Regularizer, 
                                                              init=self.config.Initializer)

            pred = tf.nn.softmax(logit, name='prediction') # output name: 'SpoofDenseNet/prediction'
            acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred, axis=1), labels_ph), dtype=tf.float32))
        
        return embeddings, logit, inference_loss, pred, acc
        '''

    def train_net(self, start_epoch=-1):
        if not isdir(self.checkpointsPath+"/summary_{}".format(start_epoch+1)):
            mkdir(self.checkpointsPath+"/summary_{}".format(start_epoch+1))

        # build training dataset and network
        iterator_train, next_element_train = self.build_dataset(self.config.TRAIN_REC, self.config.BATCH_SIZE, self.config.NUM_CLASSES, 
                                                                target_size=self.config.Image_Height, fill_value=self.config.Fill_Value, 
                                                                ratio_h=self.config.Fill_Ratio_H, ratio_w=self.config.Fill_Ratio_W, training=True)

        logit, inference_loss, pred, acc, embeddings, cmp_loss, total_loss = self.build_net(training=True, reuse=None)

        # prepare for training
        global_step = tf.Variable(name='global_step', initial_value=0, trainable=False)
        inc_op = tf.assign_add(global_step, 1, name='increment_global_step')
        lr = tf.train.piecewise_constant(global_step, boundaries=self.config.Opt_lr_steps, values=self.config.Opt_lr, name='lr_schedule')
        # define the optimize method
        opt = tf.train.MomentumOptimizer(learning_rate=lr, momentum=self.config.Opt_momentum)
        # get train op
        #grads = opt.compute_gradients(inference_loss)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):        
            #train_op = opt.apply_gradients(grads, global_step=global_step)
            train_op = opt.minimize(total_loss, global_step=global_step)

        # build validation dataset and network
        iterator_val, next_element_val = self.build_dataset(self.config.VAL_REC, self.config.BATCH_SIZE, self.config.NUM_CLASSES, 
                                                            target_size=self.config.Image_Height, fill_value=self.config.Fill_Value, 
                                                            ratio_h=self.config.Fill_Ratio_H, ratio_w=self.config.Fill_Ratio_W, training=False)

        logit_val, inference_loss_val, pred_val, acc_val, embeddings_val = self.build_net(training=False, reuse=True)


        # create session
        cfg = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        cfg.gpu_options.allow_growth = True
        sess = tf.Session(config=cfg)
        # create model saver
        saver = tf.train.Saver(max_to_keep=50)        

        #time_stamp = time.strftime('_%Y-%m-%d-%H-%M', time.localtime(time.time()))           

        # define log file
        log_file_path = os.path.join(self.checkpointsPath, 'train_{}'.format(start_epoch+1) + '.log')
        log_file = open(log_file_path, 'w')        

        # summary writer
        summary = tf.summary.FileWriter(self.checkpointsPath+"/summary_{}".format(start_epoch+1), sess.graph)
        summaries = []
        # trainabel variable gradients
        for var in tf.trainable_variables():
            summaries.append(tf.summary.histogram(var.op.name, var))
        # add loss summary
        summaries.append(tf.summary.scalar('inference_loss', inference_loss))
        summaries.append(tf.summary.scalar('cmp_loss', cmp_loss))
        summaries.append(tf.summary.scalar('total_loss', total_loss))
        # add learning rate
        summaries.append(tf.summary.scalar('leraning_rate', lr))
        # add accuracy
        summaries.append(tf.summary.scalar('accuracy', acc))
        summary_op = tf.summary.merge(summaries)


        # start training process
        img_verify_flag = False # set to false during actual training
        count = 0
        validation_result = []
        
        if start_epoch >= 0:
            # restore checkpoint
            saver.restore(sess, self.checkpointsPath+'/'+self.config.PREFIX+'_{}.ckpt'.format(start_epoch))
        else:
            sess.run(tf.global_variables_initializer())

        for i in range(start_epoch+1,self.config.NUM_EPOCH):
            
            sess.run(iterator_train.initializer)
            while True:
                try:
                    start = time.time()
                    images1_train, labels1_train = sess.run(next_element_train)
                    images2_train, labels2_train = sess.run(next_element_train)
                    #images_train, images_show, labels_train = sess.run(next_element_train) # for dataset debug
                    if img_verify_flag: # for dataset debug
                        img = Image.fromarray(images_show[0,...], 'RGB')
                        img.show()
                        exit(1)
                    feed_dict = {self.images1_ph: images1_train, self.labels1_ph: labels1_train,
                                 self.images2_ph: images2_train, self.labels2_ph: labels2_train}          
                    logitTrain, inferLossTrain, predTrain, accTrain, embeddingsTrain, cmpLossTrain, totalLossTrain, _, _ = \
                        sess.run([logit, inference_loss, pred, acc, embeddings, cmp_loss, total_loss, train_op, inc_op],
                                  feed_dict=feed_dict,
                                  options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))
                    end = time.time()
                    pre_sec = self.config.BATCH_SIZE/(end - start)
                    
                    if count > 0 and count % self.config.Update_Interval == 0:
                        # logging
                        print('Train: epoch %d, step %d, inferLoss %.2f, '
                              'cmpLoss %.2f, totalLoss %.2f, acc %.6f, time %.3f sp/s' %
                                  (i, count, inferLossTrain, cmpLossTrain, totalLossTrain, accTrain, pre_sec))
                        log_file.write('Training: epoch %d, total_step %d, inferLoss %.2f, '
                              'cmpLoss %.2f, totalLoss %.2f, acc %.6f, time %.3f sp/s' %
                                       (i, count, inferLossTrain, cmpLossTrain, totalLossTrain, accTrain, pre_sec) + '\n')
                        log_file.flush()
                        
                        # save summary
                        summary_op_val = sess.run(summary_op, feed_dict=feed_dict)
                        summary.add_summary(summary_op_val, count)
                        
                        #print(embTrain)
                        #print(logitTrain)
                        #print(inferenceLossTrain)
                        #print(labels_train)
                        #print(images_train)
                    count += 1
                except tf.errors.OutOfRangeError:
                    print("End of epoch %d" % i)
                    break
            
            # save check points
            ckpt_filename = self.config.PREFIX+'_{:d}'.format(i) + '.ckpt'
            ckpt_filename = os.path.join(self.checkpointsPath, ckpt_filename)
            saver.save(sess, ckpt_filename)
            print('######### Save checkpoint: {}'.format(ckpt_filename))
            log_file.write('Checkpoint: {}'.format(ckpt_filename) + '\n')
            log_file.flush()
            
                    
            # do validation
            accVal = []
            predVal = []
            labelVal = np.array([])
            sess.run(iterator_val.initializer)
            while True:
                try:
                    images_val, labels_val = sess.run(next_element_val)
                    feed_dict = {self.images_val_ph: images_val, self.labels_val_ph: labels_val}
                    acc_tmp, pred_tmp = \
                        sess.run([acc_val, pred_val],
                                  feed_dict=feed_dict,
                                  options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))                        
                    accVal += [acc_tmp]
                    #print(accVal)
                    
                    if type(predVal) == type([]):
                        predVal = pred_tmp
                    else:
                        predVal = np.append(predVal, pred_tmp, axis=0)
                    labelVal = np.append(labelVal, labels_val)
                    
                    #print(predVal)
                    #print(labelVal)
                except tf.errors.OutOfRangeError:
                    break    
            accVal = np.mean(accVal)
            print('$$$$$$$$ Validation: epoch %d, accuracy is %.6f' % (i, accVal))
            log_file.write('Validation: epoch %d, accuracy %.6f' % (i, accVal) + '\n')
            log_file.flush()
            
            # save validation results
            validation_result += [{'label': labelVal, 'pred': predVal}]
            with open(self.checkpointsPath+'/'+self.config.PREFIX+'_val_result.pkl', 'wb') as f:
                pickle.dump(validation_result, f)
        log_file.close()

    def validate_net(self, checkpoint_num, output_pickle_path):
        out_dir = output_pickle_path[0:output_pickle_path.rfind('/')]
        if not isdir(out_dir):
            mkdir(out_dir)

        # build validation dataset and network
        iterator_val, next_element_val = self.build_dataset(self.config.VAL_REC, self.config.BATCH_SIZE, self.config.NUM_CLASSES, 
                                                            target_size=self.config.Image_Height, fill_value=self.config.Fill_Value, 
                                                            ratio_h=self.config.Fill_Ratio_H, ratio_w=self.config.Fill_Ratio_W, training=False)

        logit_val, inference_loss_val, pred_val, acc_val, embeddings_val = self.build_net(training=False, reuse=None)

        # create session
        cfg = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        cfg.gpu_options.allow_growth = True
        sess = tf.Session(config=cfg)
        # create model saver
        saver = tf.train.Saver()
        
        # restore checkpoint
        saver.restore(sess, self.checkpointsPath+'/'+self.config.PREFIX+'_{}.ckpt'.format(checkpoint_num))

        # do validation
        accVal = []
        predVal = []
        labelVal = np.array([])
        sess.run(iterator_val.initializer)

        while True:
            try:
                images_val, labels_val = sess.run(next_element_val)
                feed_dict = {self.images_val_ph: images_val, self.labels_val_ph: labels_val}
                acc_tmp, pred_tmp = \
                    sess.run([acc_val, pred_val],
                              feed_dict=feed_dict,
                              options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))                       
                accVal += [acc_tmp]
                #print(accVal)
                
                if type(predVal) == type([]):
                    predVal = pred_tmp
                else:
                    predVal = np.append(predVal, pred_tmp, axis=0)
                labelVal = np.append(labelVal, labels_val)
                
                #print(predVal)
                #print(labelVal)
            except tf.errors.OutOfRangeError:
                break    
        accVal = np.mean(accVal)
        print('$$$$$$$$ Validation accuracy is %.6f' % (accVal))
        
        # save validation results
        validation_result = {'label': labelVal, 'pred': predVal}
        with open(output_pickle_path, 'wb') as f:
            pickle.dump(validation_result, f)

        return validation_result

    def deploy_net(self, checkpoint_num):
        # build deploy network
        logit, _, pred, _, _ = self.build_net(training=False, reuse=None)

        # create model saver
        saver = tf.train.Saver()

        # create session
        cfg = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)#, device_count={'CPU' : 1, 'GPU' : 0})
        cfg.gpu_options.allow_growth = True
        sess = tf.Session(config=cfg)
        
        # restore checkpoint
        saver.restore(sess, self.checkpointsPath+'/'+self.config.PREFIX+'_{}.ckpt'.format(checkpoint_num))

        return sess, logit, pred

    # img should be in RGB order and preprocessed
    def eval(self, img, sess, pred, embeddings=None, logit=None):

        feed_dict = {self.images_val_ph: img[np.newaxis,...]}

        pred_eval = sess.run([pred], feed_dict=feed_dict, options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True)) 

        return pred_eval

    def eval_pb(self, img, sess, img_ph, pred, embeddings=None, logit=None):

        feed_dict = {img_ph: img[np.newaxis,...]}

        pred_eval = sess.run([pred], feed_dict=feed_dict, options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True)) 

        return pred_eval

    # bbox = [x1, y1, x2, y2]
    def preprocess_img(self, frame, bbox, crop_scale_to_bbox=2.5):
        img_crop, bbox_new = self.prepare_img(frame, bbox, crop_scale_to_bbox=crop_scale_to_bbox, crop_square=True)
        im = self.fill_c(img_crop, target_size=self.config.Image_Height, fill_value=self.config.Fill_Value, 
                    ratio_h=self.config.Fill_Ratio_H, ratio_w=self.config.Fill_Ratio_W)

        return im

    # bbox = [x1, y1, x2, y2]
    def prepare_img(self, img, bbox, crop_scale_to_bbox=2.5, crop_square=True):
        img_h, img_w, img_c = img.shape
        bbx_x, bbx_y, x2, y2 = bbox
        bbx_w = x2 - bbx_x + 1
        bbx_h = y2 - bbx_y + 1
        
        #print('orig_img_shape: ', img.shape)

        crp_w = int(bbx_w*crop_scale_to_bbox)
        crp_h = int(bbx_h*crop_scale_to_bbox)

        if crop_square:
            crp_w = max(crp_w, crp_h)
            crp_h = crp_w

        img_crop = np.zeros([crp_h, crp_w, img_c]).astype(np.uint8)
        if img_c == 3:
            img_crop[:,:,1] = 255 # make empty portion green to differentiate with black
        else: # == 1
            img_crop[:,:,0] = 127 # make empty portion grey

        
        crp_x1 = max(0, int(bbx_x-(crp_w-bbx_w)/2.0))
        crp_y1 = max(0, int(bbx_y-(crp_h-bbx_h)/2.0))
        crp_x2 = min(int(bbx_x-(crp_w-bbx_w)/2.0)+crp_w-1, img_w-1)
        crp_y2 = min(int(bbx_y-(crp_h-bbx_h)/2.0)+crp_h-1, img_h-1)

        delta_x1 = -min(0, int(bbx_x-(crp_w-bbx_w)/2.0))
        delta_y1 = -min(0, int(bbx_y-(crp_h-bbx_h)/2.0))

        img_crop[delta_y1:delta_y1+crp_y2-crp_y1+1, delta_x1:delta_x1+crp_x2-crp_x1+1,:] = img[crp_y1:crp_y2+1,crp_x1:crp_x2+1,:].copy()

        bbx_x1 = bbx_x - crp_x1 + delta_x1
        bbx_y1 = bbx_y - crp_y1 + delta_y1
        bbx_x2 = bbx_x1 + bbx_w -1
        bbx_y2 = bbx_y1 + bbx_h -1

        #img_crop=cv2.rectangle(img_crop, (bbx_x1, bbx_y1), (bbx_x2, bbx_y2), (0,255,0), 3)
        
        return img_crop, [bbx_x1, bbx_y1, bbx_x2, bbx_y2]


    def img_crop_shuffle(self, img, height, width, grid_size=4):
        crop_h = tf.cast(height/grid_size,tf.int32)
        crop_w = tf.cast(width/grid_size,tf.int32)

        idx = list(range(grid_size*grid_size))
        np.random.shuffle(idx)
        img_tmp = []
        cnt = 0
        for i, v in enumerate(idx):
            if i%grid_size == 0:
                img_crop = tf.image.crop_to_bounding_box(img, int(idx[i]/grid_size)*crop_h,int(idx[i]%grid_size)*crop_w,crop_h,crop_w)
            else:
                img_crop = tf.concat([img_crop, tf.image.crop_to_bounding_box(img, int(idx[i]/grid_size)*crop_h,int(idx[i]%grid_size)*crop_w,crop_h,crop_w)], 1)
            cnt += 1
            if cnt == grid_size:
                img_tmp += [img_crop]
                cnt = 0
        img1 = img_tmp[0]
        for i in range(1,len(img_tmp)):
            img1 = tf.concat([img1,img_tmp[i]],0)
            
        return img1


    # this function must be used for batch_size of 1 or before batch operation since the image size varies
    def parse_function(self, example_proto, classes=3):
        assert classes in [2,3], 'only classes of 2 or 3 supported!!!'        
        
        features = {'image/encoded': tf.FixedLenFeature([], tf.string),
                    'image/height': tf.FixedLenFeature([], tf.int64),
                    'image/width': tf.FixedLenFeature([], tf.int64),
                    #'image/colorspace': tf.FixedLenFeature([], tf.string),
                    #'image/channels': tf.FixedLenFeature([], tf.int64),
                    #'image/class/text': tf.FixedLenFeature([], tf.string),
                    'image/class/label': tf.FixedLenFeature([], tf.int64)}
        
        features = tf.parse_single_example(example_proto, features)
        
        img = tf.image.decode_jpeg(features['image/encoded'])
        #img = features['image/encoded']
        
        # img = tf.reshape(img, shape=(112, 112, 3))
        # r, g, b = tf.split(img, num_or_size_splits=3, axis=-1)
        # img = tf.concat([b, g, r], axis=-1)
        
        #img = tf.cast(img, dtype=tf.float32)
        #img = tf.subtract(img, 127.5)
        #img = tf.multiply(img,  0.0078125)
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)

        #img = tf.image.crop_and_resize(img, boxes, box_ind, crop_size) # scale (zoom in)        
        #img = tf.image.extract_glimpse(img, size, offsets) # translation
        if self.config.Augm_rotate>0:
            img = tf.contrib.image.rotate(img, np.random.uniform(low = -1.0*self.config.Augm_rotate, high = self.config.Augm_rotate, size=(1,))[0]) # rotate
        if self.config.Augm_brightness>0:
            img = tf.image.random_brightness(img, max_delta=self.config.Augm_brightness)
        if self.config.Augm_saturation>0:
            img = tf.image.random_saturation(img, lower=1.0-self.config.Augm_saturation, upper=1.0+self.config.Augm_saturation)
        if self.config.Augm_hue>0:
            img = tf.image.random_hue(img, max_delta=self.config.Augm_hue)
        if self.config.Augm_contrast>0:
            img = tf.image.random_contrast(img, lower=1.0-self.config.Augm_contrast, upper=1.0+self.config.Augm_contrast)
        if self.config.Augm_crop>0:
            img = tf.image.central_crop(img, np.random.uniform(low = 1-self.config.Augm_crop, high = 1, size=(1,))[0])
        
        if classes==3:
            label = tf.cast(features['image/class/label'], tf.int64) - 1
        else:
            label = tf.cast(features['image/class/label']>1, tf.int64) # two class only

        return img, label

    # require RGB image
    def fill_c(self, img, fill_value, ratio_h, ratio_w, target_size):
        im = img.copy()
        h,w,c = img.shape
        cp_h = int(h*ratio_h)
        cp_w = int(w*ratio_w)
        cp_x = int((w-cp_w)/2)
        cp_y = int((h-cp_h)/2)

        if self.config.Channel_Number == 2:            
            if self.config.Draw_Line:
                lines = self.get_lines(im)
                tmp_img = np.zeros_like(im)
                self.draw_lines(tmp_img, lines)

                im = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
                im[:,:,2] = tmp_img[:,:,1]
                
            else:
                im = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
                im = im[:,:,0:2]

        im[cp_y:cp_y+cp_h,cp_x:cp_x+cp_w,:] = fill_value

        im = (im - 127.5) * 0.0078125

        if target_size>0:
            return cv2.resize(im, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

        return im

    # require RGB image
    def fill_c1(self, img, fill_value, ratio_h, ratio_w, target_size):
        im = img.copy()

        if target_size>0:
           im = cv2.resize(im, (target_size, target_size), interpolation=cv2.INTER_LINEAR)        

        if self.config.Channel_Number == 2:            
            if self.config.Draw_Line:
                lines = self.get_lines(im)
                tmp_img = np.zeros_like(im)
                self.draw_lines(tmp_img, lines)

                im = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
                im[:,:,2] = tmp_img[:,:,1]
                
            else:
                im = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
                im = im[:,:,0:2]

        h,w,c = im.shape
        cp_h = int(h*ratio_h)
        cp_w = int(w*ratio_w)
        cp_x = int((w-cp_w)/2)
        cp_y = int((h-cp_h)/2)
        im[cp_y:cp_y+cp_h,cp_x:cp_x+cp_w,:] = fill_value

        im = (im - 127.5) * 0.0078125

        return im

    # require RGB image
    def get_lines(self, img):

        img_h, img_w, _ = img.shape

        # find lines
        gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

        if self.lineCfg['enable_gaussian_blur']:
            blur_gray = cv2.GaussianBlur(gray,(self.lineCfg['blur_kernal_size'], self.lineCfg['blur_kernal_size']),0)
        else:
            blur_gray = gray

        edges = cv2.Canny(  blur_gray, 
                            self.lineCfg['canny_low_threshold'], 
                            self.lineCfg['canny_high_threshold'])

        # Run Hough on edge detected image
        # Output "lines" is an array containing endpoints of detected line segments
        lines = cv2.HoughLinesP(edges, 
                                self.lineCfg['hough_rho'], 
                                np.pi / 180, 
                                self.lineCfg['hough_threshold'], 
                                np.array([]),
                                self.lineCfg['hough_min_line_length'], 
                                self.lineCfg['hough_max_line_gap'])
        return lines

    def draw_lines(self, img, lines):
        if type(lines) == type(None): return

        for line in lines:
            img = cv2.line(img, (line[0][0], line[0][1]), (line[0][2], line[0][3]), (0,255,0), 2)

    def img_fill_center(self, img, fill_value=127, ratio_h=0.4, ratio_w=0.35, target_size=0):
        assert target_size>=0        
        
        #imc = tf.cast(tf.py_func(fill_c, [img, fill_value, ratio_h, ratio_w, target_size], tf.uint8),tf.float32)
        #return tf.image.per_image_standardization(imc)
        return tf.cast(tf.py_func(self.fill_c, [img, fill_value, ratio_h, ratio_w, target_size], tf.double),tf.float32)
        
    def get_output_name(self):
        sess, _, pred = self.deploy_net(0)
        return [n.name for n in tf.get_default_graph().as_graph_def().node]

    def save_pb(self, checkpoint_num, output_name=['SpoofLdConvNet/clf_stem/prediction']):
        with tf.Session() as sess:
            # restore graph
            saver = tf.train.import_meta_graph(self.checkpointsPath+'/'+self.config.PREFIX+'_{}.ckpt.meta'.format(checkpoint_num))
            # load weight
            saver.restore(sess, self.checkpointsPath+'/'+self.config.PREFIX+'_{}.ckpt'.format(checkpoint_num))

            # Freeze the graph
            frozen_graph_def = tf.graph_util.convert_variables_to_constants(
                sess,
                sess.graph_def,
                output_name)

            # Save the frozen graph
            with open(self.checkpointsPath+'/'+self.config.PREFIX+'_{}.pb'.format(checkpoint_num), 'wb') as f:
                f.write(frozen_graph_def.SerializeToString())

    def load_pb(self, pb_path):           
        print("load graph")
        f = gfile.FastGFile(pb_path,'rb')
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        graph_nodes=[n for n in graph_def.node]
        names = []
        for t in graph_nodes:
            names.append(t.name)
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name='prefix')
                
            image_in = graph.get_tensor_by_name('prefix/image_val:0')
            pred_out = graph.get_tensor_by_name('prefix/SpoofLdConvNet/clf_stem/prediction:0')
        
            sess = tf.Session(graph=graph)
        return sess, image_in, pred_out, names, graph_nodes

    def test_dataset(self):
        iterator, next_element = self.build_dataset(self.config.TRAIN_REC, 1, self.config.NUM_CLASSES, 
                                                    target_size=self.config.Image_Height, fill_value=self.config.Fill_Value, 
                                                    ratio_h=self.config.Fill_Ratio_H, ratio_w=self.config.Fill_Ratio_W, 
                                                    training=True)
        with tf.Session() as sess:
            sess.run(iterator.initializer)
            images_train, labels_train = sess.run(next_element)
            print(images_train)
            print(images_train.shape, labels_train.shape)
            img = Image.fromarray(images_train[0,...].astype(np.uint8), 'RGB')
            img.show()

'''
import tensorflow as tf
from tensorflow.core.protobuf import config_pb2
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,"
from os.path import isdir
from os import mkdir
from shutil import copyfile
import time
from PIL import Image
import pickle
from spoofing_ld_conv.SpoofLineConvTf import SpoofLineConvTf
import argparse


import sys
sys.path.append('/home/macul/libraries/mk_utils/tf_spoof/output1/{}'.format(prefix))
import net_config_ld_conv_0 as config

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--ckpt-num", required=True, help="checkpoint number")
ap.add_argument("-p", "--prefix", required=True, help="name of model prefix")
ap.add_argument("-v", "--video-path", required=True, help="input video full path")
ap.add_argument("-t", "--threshold", required=True, type=float, help="threshold")
ap.add_argument("-tf", "--threshold-false", required=True, type=float, help="threshold for false detection")
ap.add_argument("-s", "--min-size", required=False, type=int, default=100, help="threshold")
args = vars(ap.parse_args())

ckpt_num = args['ckpt_num']
prefix = args['prefix']

spfLC = SpoofLineConvTf(config)

sess, _, pred = spfLC.deploy_net(ckpt_num)

frame = cv2.imread(fname)
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
faces = faceDet(frame)

img_crop = spfLC.preprocess_img(frame, faces[0])

pred_rst = spfLC.eval(img_crop, sess, pred)
'''

'''
# to get output name
import tensorflow as tf
import numpy as np
import os
from os.path import isdir
from os import mkdir
from shutil import copyfile
import time
from PIL import Image
import pickle
from spoofing_lbp.SpoofDspTf import SpoofDspTf
from tf_spoof.config import net_config as config
SpoofVal = SpoofDspTf(config)
sess, embeddings, logit, pred, acc, features_tensor = SpoofVal.deploy_net(0)
output_node_names =[n.name for n in tf.get_default_graph().as_graph_def().node]
'''
