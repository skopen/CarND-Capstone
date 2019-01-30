# Classifier

## Transfer learning

For classifying the traffic light state (green, red, yellow or unknown), we used already existing models from [Tensorflow detection library](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md). Different models are available. We looked at the following ones:

- faster_rcnn_inception_v2_coco
- ssd_inception_v2_coco
- ssd_mobilenet_v1_coco

faster_rncc is very accurate but is very long to process images. ssd_mobilenet_v1_coco is a lot faster but its accuracy is pretty low. Consequently, ssd_inception_v2_coco is a good compromise between processing time and good accuracy. This model was chosen for the project. However, it doesn't detect traffic light colors. The model thus must be trained for detecting other shapes/patterns.

## Training

A dataset of ~1200 images has been gathered (mainly from the ROS bag provided by Udacity) and labeled using [labelimg.py](https://github.com/tzutalin/labelImg). 80% were used for training, 20% for testing.

Using tensorflow-gpu, the model has been trained locally on a computer equipped with a NVIDIA GPU (GTX 1050) during around 16000 steps, till the loss is almost always under 2:
![Loss](https://raw.githubusercontent.com/skopen/CarND-Capstone/master/imgs/loss_inception_model.png)

## Code

The classifier code is located in tl_classifier.py and returns its result to tl_detector.py.

When the classifier finds the class corresponding to a color,  it looks at the probability (or score) that this prediction is correct. This minimum score is 0.45. If the score is below this, the classifier returns that no traffic light state was found.

After the color is return, the detector does not actualize directly the state of the traffic light. It requires that at least 4 images in a row returns the same prediction/color before actualizing the light state. It might happen that the classifier classifies erroneously 1 or 2 images from time to time and this feature avoids that the light state changes because of wrong classifying.

## Results

![Green](https://raw.githubusercontent.com/skopen/CarND-Capstone/master/imgs/classify_green.jpg)

![Red](https://raw.githubusercontent.com/skopen/CarND-Capstone/master/imgs/classify_red.jpg)
![
](https://raw.githubusercontent.com/skopen/CarND-Capstone/master/imgs/classify_yellow.jpg)