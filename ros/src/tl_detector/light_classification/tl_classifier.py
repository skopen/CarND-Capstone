#!/usr/bin/env python2
import os.path
import numpy as np
import scipy.misc
import tensorflow as tf
import cv2
import rospy
from styx_msgs.msg import TrafficLight


class TLClassifier(object):
    # In Initialization, a boolean argument "is_site" should be added so that the classifier chooses which model it should use (simulation or site model)
    def __init__(self):
        
        # Choose which model to work with
        MODEL_NAME = 'light_classification/traffic_models/site_frozen_graph'
        
        # Store the last detected color (used for debug purposes/printing messages, could be deleted later)
        self.detected_color = None

        # Path to frozen detection graph.
        PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

        self.model = None
        self.width = 0
        self.height = 0
        self.channels = 3

        # Load a frozen model into memory
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            self.sess = tf.Session(graph=self.detection_graph)
            
            # Input/output tensors for detection_graph
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def get_classification(self, image):
        # Determines the color of the traffic light in the image
        
        # Sets the minimum score (or desired probability) that the classifier's prediction should satisfy for considering the prediction as reliable.
        THRESHOLD_SCORE = 0.7
        image_np = np.asarray(image, dtype="uint8")
        image_np_expanded = np.expand_dims(image_np, axis=0)

        with self.detection_graph.as_default():
            (boxes, scores, classes, num) = self.sess.run(
                [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
                feed_dict={self.image_tensor: image_np_expanded})
        boxes = np.squeeze(boxes)
        classes = np.squeeze(classes).astype(np.int32)
        scores = np.squeeze(scores)
        num = np.squeeze(num)
        
        # Print detected classes and their score for debugging
        #rospy.loginfo(boxes)
        #rospy.loginfo(classes[:3])
        #rospy.loginfo(scores[:3])
        
        # Check the nature (green, red, yellow?) of the detected class and check its score. If the score is less than THRESHOLD_SCORE (60% in initial implementation), we consider that the detected class is not correct and the classifier should return "unknown".
        if classes[0] == 1 and scores[0] > THRESHOLD_SCORE:
            if self.detected_color != classes[0]:
                self.detected_color = classes[0]
                #rospy.loginfo('Green light')
                #cv2.imwrite('green.jpg', image_np)
            return TrafficLight.GREEN
        elif classes[0] == 2  and scores[0] > THRESHOLD_SCORE:
            if self.detected_color != classes[0]:
                self.detected_color = classes[0]
                #rospy.loginfo('Red light')
                #cv2.imwrite('red.jpg', image_np)
            return TrafficLight.RED
        elif classes[0] == 3  and scores[0] > THRESHOLD_SCORE:
            if self.detected_color != classes[0]:
                self.detected_color = classes[0]
                #rospy.loginfo('Yellow light')
                #cv2.imwrite('yellow.jpg', image_np)
            return TrafficLight.YELLOW
        else:
            if self.detected_color != classes[0]:
                self.detected_color = classes[0]
                #rospy.loginfo('Unknown')
            return TrafficLight.UNKNOWN

