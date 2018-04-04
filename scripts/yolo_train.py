#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import rospy
from core import YOLO, parse_annotation

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

class YoloTrain(object):
    def __init__(self):
        self.n_gpu = rospy.get_param('~n_gpu', default=1)  # # of GPUs to use to train
        
        # Either 'tiny_yolo', full_yolo, 'mobile_net, 'squeeze_net', or 'inception3':
        self.backend = rospy.get_param('~backend', default='full_yolo')
        self.backend_path = rospy.get_param('~weights_path')  # Weights directory
        self.input_size = rospy.get_param('~input_size', default=416)  # DO NOT change this. 416 is default for YOLO.
        self.max_number_detections = rospy.get_param('~max_number_detections', default=5)  # Max number of detections
        
        # The anchors to use. Use the anchor generator and copy these into the config.
        self.anchors = rospy.get_param('~anchors', default=[0.57273, 0.677385, 1.87446, 
                                                  2.06253, 3.33843, 5.47434, 7.88282, 
                                                  3.52778, 9.77052, 9.16828])
        self.weights_path = rospy.get_param('~weights_path', default='../weights/full_yolo.h5')  # Path to the weights.h5 file
        self.weight_file = rospy.get_param('~weight_file')

        self.train_annot_folder = rospy.get_param('~train_annot_folder')
        self.train_image_folder = rospy.get_param('~train_image_folder')
        self.labels = rospy.get_param('~labels')  # Eg: ['trafficcone', 'person', 'dog']
        self.train_times = rospy.get_param('~train_times', default=8)
        self.valid_times = rospy.get_param('~valid_times', default=)

        self.yolo = YOLO(
            n_gpu=self.n_gpu,
            backend = self.backend,
            backend_path=self.backend_path,
            input_size = self.input_size, 
            labels = self.labels, 
            max_box_per_image = self.max_number_detections,
            anchors = self.anchors
        )

        self.yolo.train(
            train_imgs = train_imgs,
            valid_imgs = valid_imgs,
            train_times = config['train']['train_times'],
            valid_times = config['valid']['valid_times'],
            nb_epochs = config['train']['nb_epochs'], 
            learning_rate = config['train']['learning_rate'], 
            batch_size = config['train']['batch_size'],
            warmup_epochs = config['train']['warmup_epochs'],
            object_scale = config['train']['object_scale'],
            no_object_scale = config['train']['no_object_scale'],
            coord_scale = config['train']['coord_scale'],
            class_scale = config['train']['class_scale'],
            saved_weights_name = config['train']['saved_weights_name'],
            debug = False)

if __name__ == '__main__':
    rospy.init_node('yolov2_ros_tain')

    try:
        yt = YoloTrain()
    except rospy.ROSInterruptException:
        pass