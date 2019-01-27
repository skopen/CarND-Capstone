# Classifier
## classifier.py

### Simulator VS Site

In this python script, one model is used (sim_frozen_graph) when classifying light in the simulator. A second model will be added (real_frozen_graph) later for classifying light in real life site.

For this reason, a boolean argument (for example "is_site") should be passed when creating the TLClassifier in tl_detector. This way, the classifier will know if we are in simulator or on site when it initializes and, consequently, which model it should use.

### Detection

In the function get_classification(), there is this code:

    if classes[0] == 1 and scores[0] > THRESHOLD_SCORE:

The script looks at the first detected box (usually there are 3 because they are 3 lights in the simulator, but only one on site) and determines its class (1 for green, 2 for red and 3 for yellow). Furthermore, it checks the probability of this box to be the color it detected. If this probability is more than THRESHOLD_SCORE, then the program assumes that it is good enough and returns the color to tl_detector script.

## tl_detector.py

### Implementation
This code was added for printing the state of the light:

    last_detected_state = self.get_classification()
        if last_detected_state == self.last_classifier_state:
            self.counter_classifier += 1
        else:
            self.counter_classifier = 0
        
        self.last_classifier_state = last_detected_state
        
        
        rospy.loginfo(self.counter_classifier)
        #rospy.loginfo(last_detected_state)
        
        # Check how many times in the row the same color was detected. If it's more or equal to MIN_COUNTER, we update classifier_state with the actual color detected.
        if self.last_classifier_state == TrafficLight.GREEN and self.counter_classifier == MIN_COUNTER:
            rospy.loginfo('Green light')
            self.classifier_state = self.last_classifier_state
        elif self.last_classifier_state == TrafficLight.RED and self.counter_classifier == MIN_COUNTER:
            rospy.loginfo('Red light')
            self.classifier_state = self.last_classifier_state
        elif self.last_classifier_state == TrafficLight.YELLOW and self.counter_classifier == MIN_COUNTER:
            rospy.loginfo('Yellow light')
            self.classifier_state = self.last_classifier_state
        elif self.last_classifier_state == TrafficLight.UNKNOWN and self.counter_classifier == MIN_COUNTER:
            rospy.loginfo('No traffic light detected')
            self.classifier_state = self.last_classifier_state

This code is now placed at the beginning at the image_cb() function so that it gets called everytime an image is received (even if we use the traffic lights coordinates/status provided by Udacity). This code should be moved in the "get_classification()" function.

Then you should set a boolean (that was already created: self.use_classifier) to False if you want to use the traffic light status from the simulator or True if you want to use the classifier.

### Counter

A constant MIN_COUNTER (=1 in initial implementation) was added to reduce errors when classifying; if the light is green and the classifier detects it correctly but after a few seconds it detects one frame with red light (= wrong detection) before detecting green again, the status of the light should NOT be changed directly to red.

The counter was implemented to ensure that at least (MIN_COUNTER+1) = 2 images in a row are detected with the same color before actually changing the light status.