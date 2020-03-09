# Vehicles_Detector 
This repository includes different Conv-Net architectures for detection of road-side objects including pedestrians, 
these Conv-Nets are trained only on 6 classes i.e. Bus, Van, Car, Rickshaw, Bike, and Pedestrians.Images for training are taken 
from a high angle i.e. height of a traffic light. So it performs better on high angle images as compared to low angle images.
The trained models are placed in 'inference_graph' folder by their respective names, one can use any of the given models by parsing 
its name while executing "Detect.py" file by the argument of --model (default set to Faster-RCNN Inception model)
                                               
                                                       Detect.py --model ssd_mobilenet

There are a few shortcomings in this project, since these models are trained (been through transfer learning) on a very small dataset of non-uniformly distributed objects, i.e there were very few images of buses in training dataset, so it does end up sometimes misclassifying buses with other classes.
SSD_Inception and SSD_MobileNet are comparitively faster models but with very low accuracy, whereas other two models (FasterRCNN_Inception & FasterRCNN_ResNet50) take a bit longer to infer but are much better at accuracy.
