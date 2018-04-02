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
        self.bridge = CvBridge()

        self.rgb_image_topic = rospy.get_param('~image_topic', default='/camera/rgb/image_raw')  # RGB image topic
        self.image_type = rospy.get_param('~image_type', default='rgb')                          # Either 'rgb' or 'rgbd'
        self.get_object_pos = rospy.get_param('~get_object_pos', default=False)                  # Either True or False

        rospy.loginfo('Using RBG image topic {}'.format(self.rgb_image_topic))
        rospy.loginfo('Setting image type to {}'.format(self.image_type))
        rospy.loginfo('Setting get object position to {}'.format(self.get_object_pos))
        
        # Make sure we are not useing a RGB camera and trying to get object position
        if self.get_object_pos and self.image_type == 'rgb':
            ros.logerr('Can not get object position with RGB data! Please use a RGBD camera instead.')

        self.rgb_image_sub = rospy.Subscriber(self.rgb_image_topic, Image, self._image_cb)
        self.detect_pub = rospy.Publisher('{}/detected'.format(rospy.get_name()), Detection2DArray, queue_size=1)
        self.bounding_box_pub = rospy.Publisher('{}/bounding_box_image'.format(rospy.get_name()), Image, queue_size=1)

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
            yolo_detect = rospy.ServiceProxy('yolo_detect', YoloDetect, persistent=True)
            detected = yolo_detect(YoloDetectRequest(data)).detection
            try:
                cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            except CvBridgeError as e:
                rospy.logerr(e)
            # rospy.loginfo('Found {} bounding boxes'.format(len(detected.detection.detections)))

            image = self.draw_boxes(cv_image, detected)

            self.bounding_box_pub.publish(self.bridge.cv2_to_imgmsg(image, "bgr8"))
            # self.detect_pub.publish(detected)
        except rospy.ServiceException as e:
            rospy.logerr(e)

    def _depth_cb(self, data):
        try:
            cv_depth_image = self.bridge.imgmsg_to_cv2(data, "16UC1")
        except CvBridgeError as e:
            rospy.logerr(e)

    def draw_boxes(self, image, detected):
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
    
    # TODO: Implement
    def calculate_distance(self, results, image_depth):
        # Only publish if you see an object and the closest
        # Loop through all the bounding boxes and find min
        object_depth = sys.maxsize
        detected = False
        #bounding_box = None
        for i in range(len(results)):
            # check for objects pre-defined in mocap
            current_feat = results[i][0]
            if(current_feat == 'car' or
                    current_feat == 'tvmonitor' or
                    current_feat == 'dog'):
                detected = True
                location = self.get_object_2dlocation(i, results)
                #x = location[0]
                #y = location[1]
                w = location[2]
                h = location[3]
                x_center = location[4]
                y_center = location[5]

                # sanity check
                if(x_center > 640 or y_center > 480):
                    break

                center_pixel_depth = image_depth[y_center, x_center]
                distance_avg = self.depth_region(
                    image_depth, y_center, x_center, w, h)
                # convert to mm
                distance = float(center_pixel_depth) * 0.001
                # print("Distance of object {} from target : \
                #      {}".format(i, distance))

                # print("Averaged distance of object {} : "
                #      .format(distance_avg))

                # self.draw_bounding_box(results, i)
                if distance < object_depth:
                    object_depth = distance_avg
                    #bounding_box = [x, y, w, h]
                    object_name = results[i][0]

        if(detected):
            # Publish the distance and bounding box
            object_loc = [x_center, y_center]
            measurements = self.calculate_bearing(object_loc, object_depth)

            bearing = measurements[0]
            object_range = measurements[1]

            object_topic = self.construct_topic(
                            object_depth,
                            x_center,
                            y_center,
                            bearing,
                            object_range,
                            object_name)

            # rospy.loginfo(self.pub_img_pos)
            self.pub_img_pos.publish(object_topic)

    # TODO: Implement
    def depth_region(self, depth_map, y_center, x_center, w, h):
        # grab depths along a strip and take average
        # go half way
        starting_width = w/4
        end_width = w - starting_width
        x_center = x_center - starting_width
        pixie_avg = 0.0

        for i in range(starting_width, end_width):
            assert (depth_map.shape[1] > end_width)
            assert (depth_map.shape[1] > x_center)
            pixel_depth = depth_map[y_center, x_center]
            pixie_avg += pixel_depth
            x_center += 1

        pixie_avg = (pixie_avg/(end_width - starting_width)) * 0.001
        return float(pixie_avg)

    # TODO: Implement
    def calculate_bearing(self, object_loc, object_depth):
        # only consider horizontal FOV.
        # Bearing is only in 2D
        horiz_fov = 57.0  # degrees

        # Define Kinect image params
        image_width = 640  # Pixels

        # Calculate Horizontal Resolution
        horiz_res = horiz_fov/image_width

        # location of object in pixels.
        # Measured from center of image.
        # Positive x is to the left, positive y is upwards
        obj_x = image_width/2.0 - object_loc[0]

        # Calculate angle of object in relation to center of image
        bearing = obj_x*horiz_res        # degrees
        bearing = bearing*math.pi/180.0  # radians

        # Calculate true range, using measured bearing value.
        # Defined as depth divided by cosine of bearing angle
        if np.cos(bearing) != 0.0:
            object_range = object_depth/np.cos(bearing)
        else:
            object_range = object_depth

        measurements = [bearing, object_range]

        return measurements

    # TODO: Implement
    def get_object_2dlocation(self, index, results):
        x = int(results[index][1])
        y = int(results[index][2])
        w = int(results[index][3])//2
        h = int(results[index][4])//2

        x1 = x - w
        y1 = y - h
        x2 = x + w
        y2 = y + h
        x_center = (x1 + x2) // 2
        y_center = (y1 + y2) // 2

        return [x, y, w, h, x_center, y_center]

if __name__ == '__main__':
    rospy.init_node('yolov2_ros')

    try:
        yr = Yolov2Ros()
    except rospy.ROSInterruptException:
        pass

