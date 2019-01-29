#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import os
import re
import sys
import tarfile
from copy import deepcopy
import numpy as np
import cv2

# ROS includes
import rospy
from cv_bridge import CvBridge, CvBridgeError
from vision_msgs.msg import Detection2DArray, ObjectHypothesis, VisionInfo
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from kolt.srv import *

class Yolov2Ros(object):
    def __init__(self):
        self.bridge = CvBridge()

        self.rgb_image_topic = rospy.get_param('~image_topic', default='/camera/rgb/image_raw')  # RGB image topic
        self.image_type = rospy.get_param('~image_type', default='rgb')  # Either 'rgb' or 'rgbd'

        rospy.loginfo('Using RGB image topic {}'.format(self.rgb_image_topic))
        rospy.loginfo('Setting image type to {}'.format(self.image_type))

        self.rgb_image_sub = rospy.Subscriber(self.rgb_image_topic, Image, self._image_cb)
        self.detect_pub = rospy.Publisher('{}/detected'.format(rospy.get_name()), Detection2DArray, queue_size=1)
        self.bounding_box_pub = rospy.Publisher('{}/bounding_box_image'.format(rospy.get_name()), Image, queue_size=1)

        if self.image_type == 'rgbd':
            self.depth_topic = rospy.get_param('~depth_image_topic', default='/camera/depth_registered/image_raw')
            rospy.loginfo('Using depth image topic {}'.format(self.depth_topic))

            self.depth_image_sub = rospy.Subscriber(self.depth_topic, Image, self._depth_cb)

        # rate = rospy.Rate(30)
        self.rgb_image = Image()
        self.depth_image = Image()
        
        last_image = Image()
        while not rospy.is_shutdown():
            cur_img = self.rgb_image
            cur_depth = self.depth_image
            if cur_img.header.stamp != last_image.header.stamp:
                rospy.wait_for_service('yolo_detect')
                try:
                    yolo_detect = rospy.ServiceProxy('yolo_detect', YoloDetect, persistent=True)
                    detected = yolo_detect(YoloDetectRequest(cur_img)).detection
                    
                    try:
                        cv_image = self.bridge.imgmsg_to_cv2(cur_img, "bgr8")
                        # cv_depth_image = self.bridge.imgmsg_to_cv2(cur_img, "16UC1")
                    except CvBridgeError as e:
                        rospy.logerr(e)
                    
                    if len(detected.detections) > 0:
                        # rospy.loginfo('Found {} bounding boxes'.format(len(detected.detection.detections)))
                        if self.image_type == 'rgbd':
                            for i in range(0,len(detected.detections)):
                                detected.detections[i].source_img = cur_depth
                        else:
                            for i in range(0,len(detected.detections)):
                                detected.detections[i].source_img = cur_img
                        self.detect_pub.publish(detected)
                    
                    image = self._draw_boxes(cv_image, detected)
                    self.bounding_box_pub.publish(self.bridge.cv2_to_imgmsg(image, "bgr8"))
                except rospy.ServiceException as e:
                    rospy.logerr(e)
            
            last_image = cur_img
            # rate.sleep()
    
    def _image_cb(self, data):
        self.rgb_image = data

    def _depth_cb(self, data):
        self.depth_image = data

    def _draw_boxes(self, image, detected):
        for detect in detected.detections:
            box = detect.bbox
            xmin = int(box.center.x - (box.size_x/2))
            xmax = int(box.center.x + (box.size_x/2))
            ymin = int(box.center.y - (box.size_y/2))
            ymax = int(box.center.y + (box.size_y/2))

            cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (0,255,0), 3)
            cv2.putText(image, 
                        str(detect.results[0].score), 
                        (xmin, ymin - 13), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1e-3 * image.shape[0], 
                        (0,255,0), 2)

        return image

if __name__ == '__main__':
    rospy.init_node('yolov2_ros')

    try:
        yr = Yolov2Ros()
    except rospy.ROSInterruptException:
        pass

