import os
import sys
sys.path.append('../models')
sys.path.append('../explanation_strategies')
sys.path.append('../featrues_specifications')
sys.path.append('../metrics')

from yolo.predict_yolo import YoloPredict, YoloPredict_fromPath
from generate_explanation import GenerateExplanation
from person.extract_palm import ExtractPalm
from person.extract_face import ExtractFace
from trustworthiness_score import CalculateTrustworthiness
print("Done")

from yolo.get_yolo_prediction import *

import imutils

# =================================================================================
# == Results setup
# =================================================================================
ExperimentID = 1
results_dir_name = "Results/Experiment" + str(ExperimentID) + "/"
if not os.path.exists(results_dir_name):
   os.makedirs(results_dir_name)
if not os.path.exists(results_dir_name+"00/"):
   os.makedirs(results_dir_name+"00/")
if not os.path.exists(results_dir_name+"01/"):
   os.makedirs(results_dir_name+"01/")
if not os.path.exists(results_dir_name+"02/"):
   os.makedirs(results_dir_name+"02/")
if not os.path.exists(results_dir_name+"03/"):
   os.makedirs(results_dir_name+"03/")
if not os.path.exists(results_dir_name+"04/"):
   os.makedirs(results_dir_name+"04/")
if not os.path.exists(results_dir_name+"05/"):
   os.makedirs(results_dir_name+"05/")

class image_summary():
	def __init__(self, image_name, image_id):
		self.image_name = image_name 
		self.image_id = image_id

		self.array_of_predictions = []
		self.array_of_features = []

		self.trustworthiness_score = None

class person():
	def __init__(self, person_id, left, top, right, bottom):
		self.id  = person_id

		self.left   = left
		self.top    = top
		self.right  = right
		self.bottom = bottom

		self.found_overlapping_features_flag = False
		self.list_of_overlapping_features = []
		self.prediction_trustworthiness_score = 0


# =================================================================================
# == Import dataset
# =================================================================================
dataset_dir = "../datasets/INRIAPerson/Test/pos/"

# Loop over images
image_summary_array = []
image_counter = 0
for image_name in os.listdir(dataset_dir):
	image_counter += 1
	image_name = "crop_000027.png"
	image_name = "crop001602.png"
	image_name = "crop001706.png"
	image_name = "person_272.png"
	image_path = dataset_dir + image_name

	temp_image_summary = image_summary(image_name, image_counter)

	# image_path = "../datasets/persons_selected/1_person/person_004.jpg"

	# load and prepare image
	input_w, input_h = 416, 416

	image, image_w, image_h = load_image_pixels(image_path, (input_h, input_w))
	height, width, channels = image[0].shape

	image_cv2 = cv2.imread(image_path)
	dsize = (input_w, input_h)
	image_cv2 = cv2.resize(image_cv2, dsize)

	# image_cv2 = imutils.resize(image_cv2, width=input_w, height=input_h)
	# image_cv2 = image_cv2[0:input_h,0:input_w,:]
	height_cv2, width_cv2, channels_cv2 = image_cv2.shape

	# # testing transfer
	# test_x1 = 190
	# test_x2 = 190
	# test_y1 = 75
	# test_y2 = 175

	plt.clf()
	plt.imshow(image[0])
	# plt.plot([test_x1,test_x2],[test_y1,test_y2],color="black")
	plt.grid(False)
	# plt.savefig('00_image.png')
	plt.savefig(results_dir_name+"00/"+image_name)

	print("=================================")
	print("image = ", image*255)
	# print("image_cv2_x = ", image_cv2_x*255)
	print("image_cv2 = ", image_cv2)

	print("height = ", height)
	print("width = ", width)
	print("channels = ", channels)

	print("height_cv2 = ", height_cv2)
	print("width_cv2 = ", width_cv2)
	print("channels_cv2 = ", channels_cv2)
	print("=================================")
	# cv2.line(image_cv2, (int(test_x1), int(test_y1)), (int(test_x2), int(test_y2)), (0, 0, 230), thickness=2)
	# cv2.imwrite("01_image_cv2.png", image_cv2)
	cv2.imwrite(results_dir_name+"01/"+image_name, image_cv2)


	# Extract features: Face
	annotated_image = image_cv2.copy()
	face_features_coordinates = ExtractFace(annotated_image,image_name,results_dir_name)
	face_features_coordinates=[]
	print("face_features_coordinates = ",face_features_coordinates)


	# Generate prediction
	v_boxes, v_labels, v_scores, box_classes_scores = YoloPredict(image, input_w, input_h)
	annotated_image = image_cv2.copy()
	# draw boxes
	do_plot = True
	# array_of_predictions = []
	print()
	counter = 0
	for i in range(len(v_boxes)):
		if v_labels[i] == "person":
			counter += 1
			box = v_boxes[i]
			# get coordinates
			y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
			left, top, right, bottom = x1, y1, x2, y2

			# Append it to array_of_predictions
			temp_image_summary.array_of_predictions.append(person(counter, left, top, right, bottom))

			if do_plot:
				# draw text and score in top left corner
				label = "%s (%.3f)" % (v_labels[i], v_scores[i])

				# Draw rectangle 
				cv2.rectangle(annotated_image, (int(left), int(top)), (int(right), int(bottom)), (0, 0, 230), thickness=2)
				font                   = cv2.FONT_HERSHEY_SIMPLEX
				fontScale              = 1
				fontColor              = (0,0,255)
				thickness              = 2
				lineType               = 2

				cv2.putText(annotated_image,label, 
					(x1,y1), 
					font, 
					fontScale,
					fontColor,
					thickness,
					lineType)

	# cv2.imwrite("02_prediction_cv2.png", annotated_image)#cv2.flip(annotated_image, 1))
	annotated_image_path = results_dir_name+"02/"+image_name
	cv2.imwrite(annotated_image_path, annotated_image)

	input("kill script here")
	# image_cropped = image.copy()
	# image_cropped = image_cropped.reshape(416, 416, 3)
	# image_cropped = image_cropped[int(top):int(bottom),int(left):int(right),:]
	# plt.clf()
	# plt.imshow(image_cropped)
	# plt.grid(False)
	# plt.savefig("02_prediction_tensorflow.png")
	# print("yolo_prediction_coordinates = ",prediction_coordinates)
	object_classification = False
	if object_classification == True
		# Generate explanation
		explained_image, explanation_found_falg = GenerateExplanation(image, input_w, input_h)
		# TODO: will need to convert here the explanation or set of generated explanations into an array of predictions in order to use in the Trustworhtiness calcuation.
	else:
		explained_image = []
		explanation_found_falg = False


	# Extract features: Palm
	palm_features_coordinates = []# ExtractPalm(image_path)
	print("palm_features_coordinates = ",palm_features_coordinates)

	# Assign IDs to detected features
	temp_image_summary.array_of_features = [face_features_coordinates,\
							palm_features_coordinates]
	counter = 0
	for feature in array_of_features:
		counter += 1
		feature.id = counter

	# Trustworthiness calculation
	
	CalculateTrustworthiness(image, image_cv2, temp_image_summary)
	# input(str("Trustworthiness calcuated for "+str(image_name)+". Press eneter for next image."))

	image_summary_array.append(temp_image_summary)

# Save results in a log file

