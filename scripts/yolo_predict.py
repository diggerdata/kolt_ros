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
from tqdm import tqdm
from preprocessing import parse_annotation
from utils import draw_boxes
from frontend import YOLO

# ROS: includes
import rospy
from cv_bridge import CvBridge, CvBridgeError
from vision_msgs.msg import Detection2DArray, ObjectHypothesis, VisionInfo
from std_msgs.msg import Header
from sensor_msgs.msg import Image

class Yolov2Ros(object):
    def __init__(self):
        self.image_msg = Image()

        self.rgb_image_topic = rospy.get_param('image_topic', default='/camera/rgb/image_raw')  # RGB image topic
        self.image_type = rospy.get_param('image_type', default='rgb')                          # Either 'rgb' or 'rgbd'
        self.get_object_pos = rospy.get_param('get_object_pos', default=False)                  # Either True or False
        self.backend = rospy.get_param('backend', default='full_yolo')                          # Either 'tiny_yolo', full_yolo, 'mobile_net, 'squeeze_net', or 'inception3'
        self.input_size = rospy.get_param('input_size', default=416)                            # DO NOT change this. 416 is default for YOLO.
        self.labels = rospy.get_param('labels')                                                 # Eg: ['trafficcone', 'person', 'dog']
        self.max_number_detections = rospy.get_param('max_number_detections', default=5)        # Max number of detections
        self.anchors = rospy.get_param('anchors', default=[0.57273, 0.677385, 1.87446,          # The anchors to use. Use the anchor generator and copy these into the config.
                                                  2.06253, 3.33843, 5.47434, 7.88282, 
                                                  3.52778, 9.77052, 9.16828])

        rospy.loginfo('Using RBG image topic {}'.format(rgb_image_topic))
        rospy.loginfo('Setting image type to {}'.format(image_type))
        rospy.loginfo('Setting get object poition to {}'.format(self.get_object_pos))
        
        # Make sure we are not useing a RGB camera and trying to get object position
        if self.get_object_pos and self.image_type == 'rgb':
            ros.logerr('Can not get object position with RGB data! Please use a RGBD camera instead.')

        self.rgb_image_sub = rospy.Subscriber(rgb_image_topic, Image, self._image_cb)

        self.yolo = YOLO(backend             = self.backend,
                        input_size           = self.input_size, 
                        labels               = self.labels, 
                        max_box_per_image    = self.max_number_detections,
                        anchors              = self.anchors)

        if self.image_type == 'rgbd':
            self.depth_image_topic = rospy.get_param(depth_topic, default='/camera/depth_registered/image_raw')
            rospy.loginfo('Using depth image topic {}'.format(depth_image_topic))

            self.depth_image_sub = rospy.Subscriber(self.depth_topic, Image, self._depth_cb)
            self.obj_location_pub = rospy.Publisher('yolo/object_location', ObjectLocation, queue_size=1)

        self.rgb_image = None  # The latest RGB image 

        rospy.spin()
    
    def _image_cb(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)

    def _depth_cb(self, data):
        try:
            cv_depth_image = self.bridge.imgmsg_to_cv2(data, "16UC1")
        except CvBridgeError as e:
            rospy.logerr(e)

    # def _calculate_distance(self, prediction, depth_image):

if __name__ == '__main__':
    rospy.init_node('yolov2_ros', anonymous=False)

    try:
        yr = Yolov2Ros()
    except rospy.ROSInterruptException:
        pass

