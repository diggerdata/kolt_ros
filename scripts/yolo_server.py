#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from yolov2_ros.srv import *
import rospy
from copy import deepcopy
from yolov2_ros import YOLO
from vision_msgs.msg import Detection2DArray, Detection2D, BoundingBox2D, ObjectHypothesisWithPose
from geometry_msgs.msg import PoseWithCovariance, Pose2D
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class YoloServer(object):
    def __init__(self):
        self.bridge = CvBridge()

        self.backend = rospy.get_param('backend', default='full_yolo')                          # Either 'tiny_yolo', full_yolo, 'mobile_net, 'squeeze_net', or 'inception3'
        self.input_size = rospy.get_param('input_size', default=416)                            # DO NOT change this. 416 is default for YOLO.
        self.labels = rospy.get_param('labels')                                                 # Eg: ['trafficcone', 'person', 'dog']
        self.max_number_detections = rospy.get_param('max_number_detections', default=5)        # Max number of detections
        self.anchors = rospy.get_param('anchors', default=[0.57273, 0.677385, 1.87446,          # The anchors to use. Use the anchor generator and copy these into the config.
                                                  2.06253, 3.33843, 5.47434, 7.88282, 
                                                  3.52778, 9.77052, 9.16828])
        self.weights_path = rospy.get_param('weights_path', default='../weights/full_yolo.h5')   # Path to the weights.h5 file

        self.yolo = YOLO(backend = self.backend,
                        input_size = self.input_size, 
                        labels = self.labels, 
                        max_box_per_image = self.max_number_detections,
                        anchors = self.anchors)

        s = rospy.Service('yolo_detect', YoloDetect, handle_yolo_detect)
        
        rospy.spin()

    def handle_yolo_detect(self, req):
        cv_image = None
        detection_array = Detection2DArray()
        detections = []
        
        try:
            cv_image = self.bridge.imgmsg_to_cv2(req.image, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)
        
        boxes = self.yolo.predict(cv_image)
        for box in boxes:
            detection = Detection2D()
            results = []
            bbox = BoundingBox2D()
            center = Pose2D()

            detection.header = Header()
            detection.header.stamp = rospy.get_rostime()
            detection.source_img = deepcopy(req.image)

            labels = box.get_all_labels()
            for i in range(0,len(labels)):
                object_hypothesis = ObjectHypothesisWithPose()
                object_hypothesis.id = i
                object_hypothesis.score = labels[i]
                results.append(object_hypothesis)
            
            detection.results = results

            x, y = box.get_xy_center()
            center.x = x
            center.y = y
            center.theta = 0.0
            detection.center = center

            size_x, size_y = box.get_xy_extents()
            detection.size_x = size_x
            detection.size_y = size_y

            detections.append(detection)

        detection_array.header = Header()
        detection_array.header.stamp = rospy.get_rostime()
        detection_array.detections = detections

        return YoloDetectResponse(detection_array)

if __name__ == '__main__':
    rospy.init_node('yolo_server')
    
    try:
        ys = YoloServer()
    except rospy.ROSInterruptException:
        pass