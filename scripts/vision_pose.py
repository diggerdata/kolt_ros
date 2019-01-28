#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from copy import deepcopy
import rospy
import actionlib
import tf
from math import pi, sin, cos, radians, isinf, isnan
from cv_bridge import CvBridge, CvBridgeError
from vision_msgs.msg import Detection2DArray
from std_msgs.msg import Header
from sensor_msgs.msg import NavSatFix
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, Pose
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from nav_msgs.msg import Odometry


class VisionPose(object):
    def __init__(self):
        self.camera_frame = rospy.get_param('~camera_frame')
        self.detection_topic = rospy.get_param('~detection_topic', default='yolo_predict/detection')
        self.odom_topic = rospy.get_param('~odom_topic', default='odom')
        self.horiz_fov = rospy.get_param('~horiz_fov', default=85.2)
        self.vert_fov = rospy.get_param('~vert_fov', default=58.0)

        self.image_width = 1
        
        self.last_detection = Detection2DArray()
        self.last_odom = Odometry()
        self.bridge = CvBridge()
        
        self.detection_sub = rospy.Subscriber(self.detection_topic, Detection2DArray, self._detection_cb)
        self.odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self._odom_cb)

        rate = rospy.Rate(10)
        last_detection = Detection2DArray()
        while not rospy.is_shutdown():
            cur_detection = self.last_detection
            if cur_detection.header.stamp != last_detection.header.stamp:
                x, y, z = self.get_cone_pose(cur_detection)
                rospy.loginfo('x: {} y: {} z: {}'.format(x,y,z))
                if x and y and z != None:
                    self._handle_transform(self.camera_frame, x, y, z)
            last_detection = cur_detection
            rate.sleep()

    def _detection_cb(self, data):
        self.last_detection = data
        if self.image_width == None:
            self.image_width = data.detections[0].source_img.width

    def _odom_cb(self, data):
        self.last_odom = data

    def _handle_transform(self, camera_frame, x_cord, y_cord, z_cord):
        br = tf.TransformBroadcaster()
        br.sendTransform((x_cord, y_cord, z_cord),
                    quaternion_from_euler(0,0,0),
                    rospy.Time.now(),
                    'cone_vis_loc',
                    camera_frame)
    
    def get_cone_pose(self, detection):
        cone_poses = []

        cone_x = 0.0
        cone_y = 0.0
        cone_z = 0.0

        for i, detect in enumerate(detection.detections):
            depth_image = detect.source_img
            
            x_center = int(detect.bbox.center.x)
            y_center = int(detect.bbox.center.y)
            try:
                cv_depth_image = self.bridge.imgmsg_to_cv2(depth_image, "16UC1")
                img_height, img_width = cv_depth_image.shape
                center_pixel_depth = cv_depth_image[y_center, x_center]/1000
                # distance = self.depth_region(cv_depth_image, detect)
                distance = float(center_pixel_depth)
                if isinf(distance) or isnan(distance):
                    cone_x = None
                    cone_y = None
                    cone_z = None
                
                else:                
                    bearing_horiz, bearing_vert, object_range = self.calculate_bearing(img_width, img_height, x_center, y_center, distance)

                    cone_x = cos(radians(bearing_horiz)) * center_pixel_depth
                    cone_y = sin(radians(bearing_horiz)) * center_pixel_depth * -1
                    cone_z = sin(radians(bearing_vert)) * center_pixel_depth * -1
                    
                    rospy.loginfo('Bearing: {} Depth: {}'.format(bearing_horiz, center_pixel_depth))
            except CvBridgeError as e:
                rospy.logerr(e)

            return cone_x, cone_y, cone_z
            
    
    def depth_region(self, depth_map, detection):
        # grab depths along a strip and take average
        # go half way
        box = detection.bbox

        y_center = int(box.center.y)
        x_center = int(box.center.x)
        w = box.size_x
        h = box.size_y

        starting_width = int(w/6)
        end_width = int(w - starting_width)
        x_center = x_center - starting_width
        pixie_avg = 0.0

        for i in range(starting_width, end_width):
            assert (depth_map.shape[1] > end_width)
            assert (depth_map.shape[1] > x_center)
            pixel_depth = depth_map[y_center, x_center]
            pixie_avg += pixel_depth
            x_center += 1

        pixie_avg = (pixie_avg/(end_width - starting_width))
        return float(pixie_avg)

    def calculate_bearing(self, img_width, img_height, object_x, object_y, object_depth):
        # only consider horizontal FOV.
        # Bearing is only in 2D

        # Calculate Horizontal Resolution
        horiz_res = self.horiz_fov / img_width
        vert_res = self.vert_fov / img_height

        # location of object in pixels.
        # Measured from center of image.
        # Positive x is to the right, positive y is upwards
        obj_x = ((img_width / 2.0) - object_x) * -1
        obj_y = ((img_height / 2.0) - object_y) * -1
        rospy.loginfo('Object X: {}'.format(img_width))

        # Calculate angle of object in relation to center of image
        bearing_horiz = obj_x*horiz_res        # degrees
        bearing_vert = obj_y*vert_res        # degrees
        # bearing = bearing*pi/180.0  # radians

        # Calculate true range, using measured bearing value.
        # Defined as depth divided by cosine of bearing angle
        if np.cos(bearing_horiz) != 0.0:
            object_range = object_depth/np.cos(bearing_horiz)
        else:
            object_range = object_depth

        return bearing_horiz, bearing_vert, object_range

if __name__ == '__main__':
    rospy.init_node('vision_pose')

    try:
        cp = VisionPose()
    except rospy.ROSInterruptException:
        pass