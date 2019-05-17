# evaluate and compare face-align models
# this file use the model downloaded from https://github.com/yeephycho/tensorflow-face-detection
# need to install tensorflow-object-detection-api from https://github.com/tensorflow/models/tree/master/research/object_detection (works on tf 1.12)

from __future__ import print_function
import time
import numpy as np
import tensorflow as tf
import cv2

class tfFaceDet():
    def __init__(   self,
                    model_path='~/mymodels/tf_face_detection/frozen_inference_graph_face.pb'):
        print('model_path', model_path)

        self.model_path = model_path

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            self.boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            self.scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(graph=self.detection_graph, config=config)

    # image should be in RGB order
    def getModelOutput(self, image, th=0.5): # image should be single color image
        with self.detection_graph.as_default():
            image_np_expanded = np.expand_dims(image, axis=0)
            (boxes, scores, classes, num_detections) = self.sess.run(
                                                                [self.boxes, self.scores, self.classes, self.num_detections],
                                                                 feed_dict={self.image_tensor: image_np_expanded})
            boxes = np.squeeze(boxes) # [ymin, xmin, ymax, xmax]
            scores = np.squeeze(scores)
            classes = np.squeeze(classes)
            num_detections = np.squeeze(num_detections)

            boxes_new = []
            for i, s in enumerate(scores):
                if s>th and classes[i]==1:
                    boxes_new += [boxes[i,:]]
            # class: 1-face, 2-background
            return boxes_new
