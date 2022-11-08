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

class person_log():
	def __init__(self):
		self.image_name = None 
		self.image_id = None
		self.id  = None

		self.box_left   = None
		self.box_top    = None
		self.box_right  = None
		self.box_bottom = None

		self.trustworthiness_score = None


# =================================================================================
# == Import dataset
# =================================================================================
folder_dir = "../datasets/INRIAPerson/Test/pos/"

logs = []
image_id = 0
# Loop over images
for image_name in os.listdir(folder_dir):
	image_path = folder_dir + image_name

	image_id += 1

	# image_path = "../datasets/persons_selected/1_person/person_004.jpg"
	image_path_array     = []
	trustwothiness_array = []

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


	plt.clf()
	plt.imshow(image[0])
	# plt.plot([test_x1,test_x2],[test_y1,test_y2],color="black")
	plt.grid(False)
	plt.savefig('00_image.png')

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
	cv2.imwrite("01_image_cv2.png", image_cv2)
	
	# =====================================================
	# == Check for Features Specifications
	# =====================================================
	# Extract features: Face
	face_features_coordinates = ExtractFace(image_cv2,image_path)
	print("face_features_coordinates = ",face_features_coordinates)

	# Extract features: Palm
	palm_features_coordinates = ExtractPalm(image_path)
	print("palm_features_coordinates = ",palm_features_coordinates)


	TODO: check if our scrips detects all faces in an image


	# =====================================================
	# == Generate prediction
	# =====================================================
	v_boxes, v_labels, v_scores, box_classes_scores = YoloPredict(image, input_w, input_h)
	annotated_image = image_cv2.copy()
	# draw boxes
	do_plot = True
	prediction_coordinates = []
	print()
	person_counter = 0
	for i in range(len(v_boxes)):
		person_counter += 1
		if v_labels[i] == "person":
			box = v_boxes[i]
			# get coordinates
			y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
			left, top, right, bottom = x1, y1, x2, y2

			# Append it to prediction_coordinates
			prediction_coordinates.append([left, top, right, bottom])

			temp_person = person_log()
			temp_person.image_name = image_name 
			temp_person.image_id = image_id
			temp_person.id  = person_counter

			temp_person.box_left   = left
			temp_person.box_top    = top
			temp_person.box_right  = right
			temp_person.box_bottom = bottom

			temp_person.trustworthiness_score = None
			logs.append(temp_person)

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

				cv2.imwrite("02_prediction_cv2.png", annotated_image)#cv2.flip(annotated_image, 1))

			# Check if Features specificaios overlap with predicition  and calculate trustworthiness.
			
			TODO

		

			# Trustworthiness calculation
			array_of_features = [face_features_coordinates,\
									palm_features_coordinates]
			face_beta = 1
			palm_beta = 1
			array_of_beta     = [face_beta, palm_beta]
			CalculateTrustworthiness(image, image_cv2, prediction_coordinates, explained_image, array_of_features, array_of_beta)
		

	# image_cropped = image.copy()
	# image_cropped = image_cropped.reshape(416, 416, 3)
	# image_cropped = image_cropped[int(top):int(bottom),int(left):int(right),:]
	# plt.clf()
	# plt.imshow(image_cropped)
	# plt.grid(False)
	# plt.savefig("02_prediction_tensorflow.png")
	# print("yolo_prediction_coordinates = ",prediction_coordinates)

	# # Generate explanation
	# explained_image, explanation_found_falg = GenerateExplanation(image, input_w, input_h)

	# # Extract features: Face
	# face_features_coordinates = ExtractFace(image_cv2,image_path)
	# print("face_features_coordinates = ",face_features_coordinates)

	# # Extract features: Palm
	# palm_features_coordinates = ExtractPalm(image_path)
	# print("palm_features_coordinates = ",palm_features_coordinates)

	# face_beta = 1
	# palm_beta = 1

	# # Trustworthiness calculation
	# array_of_features = [face_features_coordinates,\
	# 						palm_features_coordinates]
	# array_of_beta     = [face_beta, palm_beta]
	# CalculateTrustworthiness(image, image_cv2, prediction_coordinates, explained_image, array_of_features, array_of_beta)
	# # input(str("Trustworthiness calcuated for "+str(image_name)+". Press eneter for next image."))
