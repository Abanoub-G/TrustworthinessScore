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


class log():
	def __init__(self):
		self.ExperimentID         = []
		self.ImageName           = []
		self.ImageID              = []
		self.PredictionID         = []
		self.TCS_prediction       = []
		self.TCS_image            = []
		self.Prediction_recall    = []
		self.Predcition_accuracy  = []
	

	def append(self, ExperimentID, ImageName, ImageID, PredictionID, TCS_prediction, TCS_image, Prediction_recall, Predcition_accuracy):
		self.ExperimentID.append(ExperimentID)
		self.ImageName.append(ImageName) 
		self.ImageID.append(ImageID)  
		self.PredictionID.append(PredictionID)         
		self.TCS_prediction.append(TCS_prediction)
		self.TCS_image.append(TCS_image)
		self.Prediction_recall.append(Prediction_recall)    
		self.Predcition_accuracy.append(Predcition_accuracy) 
		

	def write_file(self, output_folder, file_name):
		# Folder "results" if not already there
		# output_folder = "tests_logs"
		if not os.path.exists(output_folder):
			os.makedirs(output_folder)

		file_path = os.path.join(output_folder, file_name)
		with open(file_path, 'w') as log_file: 
			log_file.write('ExperimentID, ImageName, ImageID, PredictionID, TCS_prediction, TCS_image, Prediction_recall, Predcition_accuracy\n')
			for i in range(len(self.testNo_array)):
				log_file.write('%d, %s, %d, %d, %3.3f, %3.3f, %3.3f, %3.3f\n' %\
					(self.ExperimentID[i],self.ImageID[i], self.PredictionID[i], self.TCS_prediction[i], self.TCS_image[i], self.Prediction_recall[i], self.Predcition_accuracy[i]))
		print('Log file SUCCESSFULLY generated!')


class image_summary_class():
	def __init__(self, image_name, image_id):
		self.image_name = image_name 
		self.image_id = image_id

		self.array_of_ground_truth_predictions = []
		self.array_of_predictions = []
		self.array_of_features = []

		self.frame_trustworthiness_score = None

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

		self.P_flag = None # flag showing this is a prediction 
		self.G_flag = None # flag showing that a ground truth agrees with prediciton if P_flag == True. If P_flag == False then G_flag indicates that this is person is generated from an annotation.

def IntersectingRectangle(left1, top1, right1, bottom1,
                          left2, top2, right2, bottom2):
 
    # gives bottom-left point of intersection rectangle
    left3 = max(left1, left2)
    bottom3 = min(bottom1, bottom2)
 
    # gives top-right point of intersection rectangle
    right3 = min(right1, right2)
    top3 = min(top1, top2)
 
    # no intersection
    if (left3 > right3 or bottom3 < top3) :
        print("No intersection")
        left3   = None
        top3    = None
        right3  = None
        bottom3 = None

    return left3, top3, right3, bottom3

def SquareAreaCalculator(left, top, right, bottom):
	return abs((right-left)*(top-bottom))

def IoU_calculator(left1, top1, right1, bottom1,
                    left2, top2, right2, bottom2):

	square1_area      = SquareAreaCalculator(left1, top1, right1, bottom1)
	square2_area      = SquareAreaCalculator(left2, top2, right2, bottom2)

	left3, top3, right3, bottom3 = IntersectingRectangle(left1, top1, right1, bottom1,
                          										  left2, top2, right2, bottom2)
	
	intersection_area = SquareAreaCalculator(left3, top3, right3, bottom3)
	
	union_area        = square1_area + square2_area - intersection_area
	 
	if left3 == None:
		IoU = 0		
	else:
		IoU = (intersection_area) / (union_area)  # We smooth our devision to avoid 0/0

	return IoU

# =================================================================================
# == Import dataset
# =================================================================================
dataset_dir = "../datasets/INRIAPerson/Test/pos/"#
dataset_ground_truth_dir = "../datasets/INRIAPerson/Test/annotations/"

# Loop over images
image_summary_array = []
image_counter = 0
for image_name in os.listdir(dataset_dir):
	image_counter += 1
	# image_name = "crop_000027.png"
	# image_name = "crop001602.png"
	# image_name = "crop001706.png"
	# image_name = "person_272.png"
	# image_name = "crop_000009.png"
	# image_name = "crop001511.png"
	image_name = "crop001514.png"
	image_path = dataset_dir + image_name

	image_summary = image_summary_class(image_name, image_counter)

	# image_path = "../datasets/persons_selected/1_person/person_004.jpg"

	# load and prepare image
	input_w, input_h = 416, 416

	image, image_w, image_h = load_image_pixels(image_path, (input_h, input_w))
	height, width, channels = image[0].shape

	image_cv2 = cv2.imread(image_path)
	dsize = (input_w, input_h)
	image_cv2 = cv2.resize(image_cv2, dsize)

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

	# =================================================================================
	# == Generate predictions
	# =================================================================================
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
			image_summary.array_of_predictions.append(person(counter, left, top, right, bottom))

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
	print("image_summary.array_of_predictions = ",image_summary.array_of_predictions)
	

	# input("kill script here")
	# image_cropped = image.copy()
	# image_cropped = image_cropped.reshape(416, 416, 3)
	# image_cropped = image_cropped[int(top):int(bottom),int(left):int(right),:]
	# plt.clf()
	# plt.imshow(image_cropped)
	# plt.grid(False)
	# plt.savefig("02_prediction_tensorflow.png")
	# print("yolo_prediction_coordinates = ",prediction_coordinates)

	# =================================================================================
	# == Generate Explanation
	# =================================================================================
	object_classification = False
	if object_classification == True:
		# Generate explanation
		explained_image, explanation_found_falg = GenerateExplanation(image, input_w, input_h)
		# TODO: will need to convert here the explanation or set of generated explanations into an array of predictions in order to use in the Trustworhtiness calcuation.
	else:
		explained_image = []
		explanation_found_falg = False
	
	# =================================================================================
	# == Ground truth annotations extraction
	# =================================================================================
	# Generate prediction vs Ground Truth assessment
	counter = 0
	annotaiton_name = image_name[:-3]+"txt"
	ground_truth_path = dataset_ground_truth_dir + annotaiton_name
	print("ground_truth_path  = ", ground_truth_path)
	with open(ground_truth_path,"r", encoding = "ISO-8859-1") as f:
		lines = f.readlines()
	print(lines)
	number_of_persons_ground_truth = lines[5][28]
	# Check no more digits are there after the first digit.

	# Get the number of grouond truth predictions.
	temp_i = 0
	while True:
		temp_i += 1
		if lines[5][28+temp_i] == ' ':
			break
		number_of_persons_ground_truth += lines[5][28+temp_i]	
	number_of_persons_ground_truth = int(number_of_persons_ground_truth)
	print("==================================")
	print("number_of_persons_ground_truth = ",number_of_persons_ground_truth)

	# Loop over the ground truth predcitions and extract boundary boxes.
	for temp_j in range(number_of_persons_ground_truth):
		flag_left_found   = False
		flag_top_found    = False
		flag_right_found  = False
		flag_bottom_found = False
		
		temp_i = 0
		left   = lines[17+temp_j*7][69]
		
		while True:
			temp_i += 1
			current_character = lines[17+temp_j*7][69+temp_i]

			if not flag_left_found and current_character != ",":
				left += current_character
			elif not flag_left_found and current_character == ",":
				flag_left_found = True
				temp_i += 2
				bottom = lines[17+temp_j*7][69+temp_i]

			elif not flag_bottom_found and current_character != ")":
				bottom += current_character
			elif not flag_bottom_found and current_character == ")":
				flag_bottom_found = True
				temp_i += 5
				right = lines[17+temp_j*7][69+temp_i]

			elif not flag_right_found and current_character != ",":
				right += current_character
			elif not flag_right_found and current_character == ",":
				flag_right_found = True
				temp_i += 2
				top = lines[17+temp_j*7][69+temp_i]

			elif not flag_top_found and current_character != ")":
				top += current_character
			elif not flag_top_found and current_character == ")":
				flag_top_found = True
				break


		image_summary.array_of_ground_truth_predictions.append(person(temp_j, left, top, right, bottom))

	# =================================================================================
	# == Ground truth VS Predictions assessment
	# =================================================================================
	# Initialise True-Positive (TP), False-Positive (FP),  False-Negative (FN)
	TP = 0
	FP = 0
	FN = 0

	IoU_threshold = 0.7

	# Loop over predictions
	for current_person_prediction in image_summary.array_of_predictions:
		
		# Extract prediction box bouandaries
		prediction_left = current_person_prediction.left
		prediction_top = current_person_prediction.top
		prediction_right = current_person_prediction.right
		prediction_bottom = current_person_prediction.bottom

		current_person_prediction.P_flag = True
		current_person_prediction.G_flag = False
		
		# Loop over ground truth annotations
		for current_person_annotation in image_summary.array_of_ground_truth_predictions:

			# Extract annotation box bouandaries
			annotation_left = current_person_annotation.left
			annotation_top = current_person_annotation.top
			annotation_right = current_person_annotation.right
			annotation_bottom = current_person_annotation.bottom
			
			# Calculate Intersection over Union (IoU)
			IoU =  IoU_calculator(prediction_left, prediction_top, prediction_right, prediction_bottom,
                    				annotation_left, annotation_top, annotation_right, annotation_bottom)

			if IoU >= IoU_threshold:
				current_person_prediction.G_flag = True
				TP += 1 

		# After looping over all ground truths
		if current_person_prediction.G_flag == False:
			FP += 1

	# Calcualte FN
	M_predictions = len(image_summary.array_of_predictions)
	N_annotations = len(image_summary.array_of_ground_truth_predictions)
	
	FN = N_annotations - TP


	Test this code

	
	# =================================================================================
	# == Detect Features Specifications
	# =================================================================================

	# Extract features: Face
	annotated_image = image_cv2.copy()
	# face_features_coordinates = ExtractFace(image_summary, annotated_image,image_name,results_dir_name)
	ExtractFace(annotated_image,image_name,results_dir_name, image_summary)
	print("image_summary.array_of_features = ",image_summary.array_of_features)
	
	# face_features_coordinates=[]
	# print("face_features_coordinates = ",face_features_coordinates)


	# Extract features: Palm
	# palm_features_coordinates = []# ExtractPalm(image_path)
	# print("palm_features_coordinates = ",palm_features_coordinates)

	# Assign IDs to detected features
	# image_summary.array_of_features = 
	                  #= [face_features_coordinates,\
							#palm_features_coordinates]

	counter = 0
	print("image_summary.array_of_features = ",image_summary.array_of_features)
	for feature in image_summary.array_of_features:
		counter += 1
		feature.id = counter


	# =================================================================================
	# == Trustworthiness Cacluation
	# =================================================================================
	CalculateTrustworthiness(image, image_cv2, image_summary)
	# input(str("Trustworthiness calcuated for "+str(image_name)+". Press eneter for next image."))

	image_summary_array.append(image_summary)



# =================================================================================
# == Log
# =================================================================================
# Save results in a log file
experiment_logs = log()
for current_image in image_summary_array:
	for current_person in current_image.array_of_predictions:
		ImageID             = current_image.image_id
		ImageName           = current_image.image_name
		PredictionID        = current_person.id
		TCS_prediction      = current_person.prediction_trustworthiness_score
		TCS_image           = current_image.frame_trustworthiness_score
		# Prediction_recall   = 
		# Predcition_accuracy = 
		experiment_logs.append(ExperimentID, ImageName, ImageID, PredictionID, TCS_prediction, TCS_image, Prediction_recall, Predcition_accuracy)

logs_file_name = "experiment_logs" + str(ExperimentID)+".txt"
experiment_logs.write_file(results_dir_name, logs_file_name)
