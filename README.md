# Vehicles_Detector 
This repository includes different Conv-Net architectures for detection of road-side objects including pedestrians, 
these Conv-Nets are trained only on 6 classes i.e. Bus, Van, Car, Rickshaw, Bike, and Pedestrians.Images for training are taken 
from a high angle i.e. height of a traffic light. So it performs better on high angle images as compared to low angle images.
The trained models are placed in 'inference_graph' folder by their respective names, one can use any of the givens by parsing 
its name while executing "Detect.py" file by the argument of --model (default set to Faster-RCNN Inception model)
==> #python3 Detect.py --model mobileNet1 
