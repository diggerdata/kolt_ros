#!/usr/bin/env python

from __future__ import absolute_import, division, print_function

from copy import deepcopy, copy
from math import cos, isinf, isnan, pi, radians, sin
from datetime import datetime

import actionlib
import numpy as np
import rospy
import tf
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import (Pose, Point, PoseArray, PoseStamped,
                               PoseWithCovarianceStamped)
from nav_msgs.msg import Odometry
from std_msgs.msg import Header
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from vision_msgs.msg import Detection2DArray
from visualization_msgs.msg import Marker, MarkerArray

from core import Tracker


class VisionPose(object):
    def __init__(self):
        self.camera_type = rospy.get_param('~camera_type', default='zed')
        self.camera_frame = rospy.get_param('~camera_frame')
        self.detection_topic = rospy.get_param('~detection_topic', default='yolo_predict/detection')
        self.odom_topic = rospy.get_param('~odom_topic', default='odom')
        self.pose_topic = rospy.get_param('~pose_topic', default='vision_poses')
        self.horiz_fov = rospy.get_param('~horiz_fov', default=85.2)
        self.vert_fov = rospy.get_param('~vert_fov', default=58.0)

        self.image_width = 1
        
        self.last_detection = Detection2DArray()
        self.last_odom = Odometry()
        self.bridge = CvBridge()
        self.tf_listener = tf.TransformListener()
        
        self.rate = 30
        self.tracker = Tracker(1, 100, 200, 255, self.rate, ra=1.0, sv=100000.0)

        self.detection_sub = rospy.Subscriber(self.detection_topic, Detection2DArray, self._detection_cb)
        self.odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self._odom_cb)
        
        self.pose_pub = rospy.Publisher(self.pose_topic, PoseArray, queue_size=1)
        self.path_pub = rospy.Publisher('paths', MarkerArray, queue_size=1)

        rate = rospy.Rate(self.rate)
        last_detection = Detection2DArray()
        while not rospy.is_shutdown():
            cur_detection = self.last_detection
            if cur_detection.header.stamp != last_detection.header.stamp:
                vision_poses = self.get_vision_pose(cur_detection)
                if vision_poses is not None:
                    tracked_poses, tracked_paths = self.tracker.update(vision_poses)
                    self._handle_pose_broadcast(tracked_poses, self.camera_frame)
                    self._handle_path_vis_broadcast(tracked_paths, self.camera_frame)

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

    def _handle_path_vis_broadcast(self, paths, camera_frame):
        if self.tf_listener.canTransform('map', camera_frame, rospy.Time()):
            m_array = MarkerArray()
            # loop through np array of poses - [[x,y,z]]
            for id, path in paths.items():
                if len(path) % 2 == 0:
                    m = Marker()
                    m.id = id
                    m.header = Header()
                    m.header.frame_id = camera_frame
                    m.header.stamp = self.tf_listener.getLatestCommonTime('map', camera_frame)
                    m.type = 5
                    m.action = 0
                    m.scale.x = 0.01
                    m.pose.position.x = 0.0
                    m.pose.position.y = 0.0
                    m.pose.position.z = 0.0
                    m.pose.orientation.x = 0.0
                    m.pose.orientation.y = 0.0
                    m.pose.orientation.z = 0.0
                    m.pose.orientation.w = 1.0
                    m.color.a = 1.0
                    m.color.b = 1.0
                    for point in path:
                        pose = PoseStamped()
                        pose.header = Header()
                        pose.header.frame_id = camera_frame
                        pose.header.stamp = self.tf_listener.getLatestCommonTime('map', camera_frame)
                        pose.pose = Pose()
                        pose.pose.position.x = point[0]
                        pose.pose.position.y = point[1]
                        pose.pose.position.z = point[2]
                        m.points.append(self.tf_listener.transformPose('map', pose).pose.position)
                    m_array.markers.append(m)
            self.path_pub.publish(m_array)
    
    def _handle_pose_broadcast(self, poses, camera_frame):
        if self.tf_listener.canTransform('map', camera_frame, rospy.Time()):
            p_array = PoseArray()
            p_array.header = Header()
            p_array.header.frame_id = 'map'
            p_array.header.stamp = rospy.get_rostime()
            # loop through np array of poses - [[x,y,z]]
            # rospy.loginfo(poses)
            for pose in poses:
                p = PoseStamped()
                p.header = Header()
                p.header.frame_id = camera_frame
                p.header.stamp = self.tf_listener.getLatestCommonTime('map', camera_frame)
                p.pose = Pose()
                p.pose.position.x = pose[0]
                p.pose.position.y = pose[1]
                p.pose.position.z = pose[2]
                # add the pose of the pose stamped to the pose array
                p_array.poses.append(self.tf_listener.transformPose('map', p).pose)
            self.pose_pub.publish(p_array)
    
    def get_vision_pose(self, detection):
        vision_poses = None

        object_x = 0.0
        object_y = 0.0
        object_z = 0.0

        for i, detect in enumerate(detection.detections):
            depth_image = detect.source_img
            
            x_center = int(detect.bbox.center.x)
            y_center = int(detect.bbox.center.y)
            try:
                cv_depth_image = None
                center_pixel_depth = None
                if self.camera_type == 'zed':
                    cv_depth_image = self.bridge.imgmsg_to_cv2(depth_image, "32FC1")
                    center_pixel_depth = cv_depth_image[y_center, x_center]
                elif self.camera_type == 'realsense':
                    cv_depth_image = self.bridge.imgmsg_to_cv2(depth_image, "16UC1")
                    center_pixel_depth = cv_depth_image[y_center, x_center]/1000
                else:
                    cv_depth_image = self.bridge.imgmsg_to_cv2(depth_image, "16UC1")
                    center_pixel_depth = cv_depth_image[y_center, x_center]/1000
                    
                img_height, img_width = cv_depth_image.shape
                distance = float(center_pixel_depth)
                if isinf(distance) or isnan(distance):
                    object_x = None
                    object_y = None
                    object_z = None
                
                else:                
                    bearing_horiz, bearing_vert, object_range = self.calculate_bearing(img_width, img_height, x_center, y_center, distance)

                    object_x = cos(radians(bearing_horiz)) * center_pixel_depth
                    object_y = sin(radians(bearing_horiz)) * center_pixel_depth * -1
                    object_z = sin(radians(bearing_vert)) * center_pixel_depth * -1

                    if vision_poses is None:
                        vision_poses = np.array([object_x, object_y, object_z])[np.newaxis]
                    else:
                        app_array = np.array([object_x, object_y, object_z])[np.newaxis]
                        vision_poses = np.append(vision_poses, app_array, axis=0)
                    # rospy.loginfo('Bearing: {} Depth: {}'.format(bearing_horiz, center_pixel_depth))
            except CvBridgeError as e:
                rospy.logerr(e)

        return vision_poses

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
