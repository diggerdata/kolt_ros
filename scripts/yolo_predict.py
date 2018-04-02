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

# ROS: includes
import rospy
from cv_bridge import CvBridge, CvBridgeError
from vision_msgs.msg import Detection2DArray, ObjectHypothesis, VisionInfo
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from yolov2_ros.srv import *

class Yolov2Ros(object):
    def __init__(self):
        self.image_msg = Image()

        self.rgb_image_topic = rospy.get_param('~image_topic', default='/camera/rgb/image_raw')  # RGB image topic
        self.image_type = rospy.get_param('~image_type', default='rgb')                          # Either 'rgb' or 'rgbd'
        self.get_object_pos = rospy.get_param('~get_object_pos', default=False)                  # Either True or False

        rospy.loginfo('Using RBG image topic {}'.format(self.rgb_image_topic))
        rospy.loginfo('Setting image type to {}'.format(self.image_type))
        rospy.loginfo('Setting get object poition to {}'.format(self.get_object_pos))
        
        # Make sure we are not useing a RGB camera and trying to get object position
        if self.get_object_pos and self.image_type == 'rgb':
            ros.logerr('Can not get object position with RGB data! Please use a RGBD camera instead.')

        self.rgb_image_sub = rospy.Subscriber(self.rgb_image_topic, Image, self._image_cb)
        self.detect_pub = rospy.Publisher('yolo/detected', Detection2DArray, queue_size=1)

        if self.image_type == 'rgbd':
            self.depth_image_topic = rospy.get_param(self.depth_image_topic, default='/camera/depth_registered/image_raw')
            rospy.loginfo('Using depth image topic {}'.format(self.depth_image_topic))

            self.depth_image_sub = rospy.Subscriber(self.depth_topic, Image, self._depth_cb)
            self.obj_location_pub = rospy.Publisher('yolo/object_location', ObjectLocation, queue_size=1)

        self.rgb_image = None  # The latest RGB image 

        rospy.spin()
    
    def _image_cb(self, data):
        rospy.wait_for_service('yolo_detect')
        try:
            yolo_detect = rospy.ServiceProxy('yolo_detect', YoloDetect)
            detected = yolo_detect(YoloDetectRequest(data))
            # rospy.loginfo('Found {} bounding boxes'.format(len(detected.detection.detections)))
            self.detect_pub.publish(detected.detection)
        except rospy.ServiceException as e:
            rospy.logerr(e)

    def _depth_cb(self, data):
        try:
            cv_depth_image = self.bridge.imgmsg_to_cv2(data, "16UC1")
        except CvBridgeError as e:
            rospy.logerr(e)

    # def _calculate_distance(self, prediction, depth_image):

if __name__ == '__main__':
    rospy.init_node('yolov2_ros')

    try:
        yr = Yolov2Ros()
    except rospy.ROSInterruptException:
        pass

