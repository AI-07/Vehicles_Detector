######## Image Object Detection Using Tensorflow-trained Classifier #########
#
# Original Author: Evan Juras (https://www.linkedin.com/in/evan-juras/)
# Date: 1/15/18
# Description:
# This program uses a TensorFlow-trained neural network to perform object detection.
# It loads the classifier and uses it to perform object detection on an image.
# It draws boxes, scores, and labels around the objects of interest in the image.


# Modified by: Hasnain Ahsan
# Date: 27/02/2020




## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## and some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

## but I changed it to make it more understandable to me.

# Import packages
import os
import argparse
import cv2
import numpy as np
import tensorflow as tf
import sys
import time

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

def arguments():

    parser = argparse.ArgumentParser(description="Tensorflow object detection module")
    parser.add_argument("--model", dest = 'model', help =
                        "model / name of the model which you want to use for inference",
<<<<<<< HEAD
                        default = "FasterRCNN_Inception_V2", type = str)
=======
                        default = "FasterRCNN__Inception_V2", type = str)
>>>>>>> 5fc64474f603749d0b57601ec373e56203a36415

    return parser.parse_args()




args = arguments()
# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'
IMAGE_LIST = os.listdir(os.path.join(os.getcwd(),"Images")) #list of all images in 'Images' folder


# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,args.model,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'annotations','label_map.pbtxt')


# Number of classes the object detector can identify
NUM_CLASSES = 6

# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Load image using OpenCV and
# expand image dimensions to have shape: [1, None, None, 3]
# i.e. a single-column array, where each item in the column has the pixel RGB value
indx = 0
for img in IMAGE_LIST:
    tic = time.time()
    image = cv2.imread(os.path.join(CWD_PATH,'Images',img))
    image_expanded = np.expand_dims(image, axis=0)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_expanded})

    # Draw the results of the detection (aka 'visulaize the results')
    threshold = 0.60
    vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=1,
        min_score_thresh=threshold)

    # All the results have been drawn on image. Now display the image.
    cv2.imwrite(('Detections/'+'Detected_'+IMAGE_LIST[indx]), image)
    toc=time.time()
    Dets = np.sum((scores > threshold))
    print('Image:',img,"took:",round(toc-tic,3),"secs for detection & Detected:",Dets,"objects") #Time taken for detection
    indx+=1
