# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications import vgg16
# from tensorflow.keras.applications.vgg16 import VGG16
# from tensorflow.keras.applications import inception_v3, mobilenet, xception
# import csv

# import argparse
# import os
# import sys
# sys.path.append('../MyScripts')
# print(sys.path.append('../MyScripts'))


# load yolov3 model and perform object detection
# based on https://github.com/experiencor/keras-yolo3
import numpy as np
from numpy import expand_dims
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

import sys
sys.path.append("..") # Adds higher directory to python modules path.

from my_utils import *
# from utils import *
from my_to_explain import *
from comp_explain import *

from get_yolo_prediction import *
 


# ============================================================================================================================================
# === Loading trained model and getting prediction
# ============================================================================================================================================
# load yolov3 model
model = load_model('model.h5')
# define the expected input shape for the model
input_w, input_h = 416, 416
# define our new photo
photo_filename = 'person_001.jpg'#'person_069.jpg' # 'zebra.jpg' 

# load and prepare image
image, image_w, image_h = load_image_pixels(photo_filename, (input_w, input_h))
print("image = ", image)
print("image.shape = ", image.shape)

# image[:, 0:100, 0:100, :] = 1
# image[:, 0:200, :, :] = 1
# image[:, :, 0:150, :] = 1
# image[:, :, 330:-1, :] = 1
plt.imshow(image.reshape(416, 416,3))
plt.grid(False)
plt.savefig('image.png')
# print("image = ", image)
print("image.shape = ", image.shape)
input("Enter")

# make prediction
yhat = model.predict(image)

# summarize the shape of the list of arrays
print([a.shape for a in yhat])
# define the anchors
anchors = [[116,90, 156,198, 373,326], [30,61, 62,45, 59,119], [10,13, 16,30, 33,23]]

# define the probability threshold for detected objects
class_threshold = 0.6
boxes = list()
for i in range(len(yhat)):
	# decode the output of the network
	boxes += decode_netout(yhat[i][0], anchors[i], class_threshold, input_h, input_w)

# correct the sizes of the bounding boxes for the shape of the image
# correct_yolo_boxes(boxes, image_h, image_w, input_h, input_w)
correct_yolo_boxes(boxes, input_h, input_w, input_h, input_w)

# suppress non-maximal boxes
print("len(boxes) = ",len(boxes))
do_nms(boxes, 0.5)

# define the labels
labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
	"boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
	"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
	"backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
	"sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
	"tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
	"apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
	"chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
	"remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
	"book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

# get the details of the detected objects
v_boxes, v_labels, v_scores, box_classes_scores = get_boxes(boxes, labels, class_threshold)
print("box_classes_scores = ", box_classes_scores)
print("v_boxes = ", v_boxes)
print("v_labels = ", v_labels)
# input("Enter1")
# summarize what we found
for i in range(len(v_boxes)):
	print(v_labels[i], v_scores[i])

# draw what we found
# draw_boxes(photo_filename, v_boxes, v_labels, v_scores)
name_prediction_file = "image_prediction.png"
draw_boxes(image, v_boxes, v_labels, v_scores, name_prediction_file)

plt.clf()
plt.imshow(image.reshape(416, 416,3))
plt.grid(False)
plt.savefig('image2.png')
# ============================================================================================================================================
# === Explaining prediction
# ============================================================================================================================================
# Get image row, cols and channels
img_rows, img_cols, img_channels = int(image.shape[1]), int(image.shape[2]), int(image.shape[3])

print("image.shape = ",image.shape)
print("img_rows = ",img_rows)
print("img_cols = ",img_cols)
print("img_channels = ",img_channels)

# Load the input data
fnames=[]
xs=[]

x  =  image
# x = load_img(photo_filename, target_size=(img_rows, img_cols))
# x = load_img(photo_filename)
# width, height = x.size
# x = load_img(photo_filename, target_size=(input_w, input_h))#(img_rows, img_cols))

# THIS BIT IS UNDER TESTING!!!
# convert to numpy array
# x = img_to_array(x)

# # scale pixel values to [0, 1]
# x = x.astype('float32')
# x /= 255.0

# #############
# x = np.expand_dims(x,0)

xs.append(x)
fnames.append(photo_filename)

xs = np.vstack(xs)
xs = xs.reshape(xs.shape[0], img_rows, img_cols, img_channels)
print ('\n[Total data loaded: {0}]'.format(len(xs)))

eobj = explain_objectt(model, xs)

output_foldername = "outs"
eobj.outputs = output_foldername # output folder name

top_classes = 1
eobj.top_classes = int(top_classes) # help="check the top-xx classifications", metavar="INT", default="1"

adv_ub = 1.
eobj.adv_ub = float(adv_ub) # help="upper bound on the adversarial percentage (0, 1]", metavar="FLOAT", default="1."

adv_lb = 0.
eobj.adv_lb = float(adv_lb) # help="lower bound on the adversarial percentage (0, 1]", metavar="FLOAT", default="0."

adv_value = 1 # This might be changed to 0
eobj.adv_value = float(adv_value) # help="masking value for input mutation", metavar="INT", default="234"

testgen_factor = 0.2
eobj.testgen_factor = float(testgen_factor) # help="test generation factor (0, 1]", metavar="FLOAT", default="0.2"

testgen_size = 2000
eobj.testgen_size = int(testgen_size) # help="testgen size ", metavar="INT", default="2000"

testgen_iter = 1 
eobj.testgen_iter = int(testgen_iter) # help="to control the testgen iteration", metavar="INT", default="1"

x_verbosity = 0
eobj.x_verbosity = int(x_verbosity) # help="the verbosity level of explanation output", metavar="INT", default="0"

eobj.fnames = fnames

measures = ['zoltar'] # help="the SFL measures (tarantula, zoltar, ochiai, wong-ii)", metavar="", default=['tarantula', 'zoltar', 'ochiai', 'wong-ii']
eobj.measures=measures

print("xs.shape = ",xs.shape)
prediction = eobj.model.predict(xs) # To get the string label look at the first stage of loading and prediciting.
# print("prediction = ",prediction)
prediction = yolo_to_deepcover(prediction, input_h, input_w, x)
if prediction is None:
	y = np.array([80]) # Here I am setting that when there aren't any detected objects to set the label the last label in the list of labels, which is a toothbrush. This should just allow deepcover to understand that the there were no persons detected in the image, since our study is on persons only.
else:
	y = np.argsort(prediction)[0][-eobj.top_classes:]

print("y = ",y)



causal  = False
if causal:
	comp_explain(eobj)
else: 
	to_explain(eobj)

# ##############################
# Continue FROM HERE TO COPY LINES FROM THE EXPLAINABILITY CODE

# ============================================================================================================================================
# === Extracting features specifications
# ============================================================================================================================================
