import numpy as np
import math
from numpy import expand_dims
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

import sys
sys.path.append("..") # Adds parent directory to python modules path.

from get_yolo_prediction import *
 


# ============================================================================================================================================
# === Loading trained model and getting prediction
# ============================================================================================================================================
# load yolov3 model
model = load_model('model.h5')
# define the expected input shape for the model
input_w, input_h = 416, 416
# define our new photo
# photo_filename = 'perons_001.jpg' #'person_001.jpg' #'person_069.jpg' #'zebra.jpg' 
photo_filename = 'person_042.jpg' #'person_069.jpg' #'zebra.jpg' 

# load and prepare image
image, image_w, image_h = load_image_pixels(photo_filename, (input_w, input_h))
print("image = ", image)
print("image.shape = ", image.shape)

# image[:, 0:100, 0:100, :] = 1
# image[:, 0:200, :, :] = 1
# image[:, :, 0:150, :] = 1
# image[:, :, 330:-1, :] = 1
plt.clf()
plt.imshow(image.reshape(416, 416,3))
plt.grid(False)
plt.savefig('image.png')
# print("image = ", image)
print("image.shape = ", image.shape)
# input("Enter")

# make prediction
yhat = model.predict(image)

# summarize the shape of the list of arrays
# print([a.shape for a in yhat])
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
initial_box_classes_score = box_classes_scores[0][0]
print("initial_box_classes_score = ", initial_box_classes_score)
print("v_boxes = ", v_boxes)
print("v_labels = ", v_labels)

# summarize what we found
for i in range(len(v_boxes)):
	print(v_labels[i], v_scores[i])

# draw what we found
name_prediction_file = "image_prediction.png"
draw_boxes(image, v_boxes, v_labels, v_scores, name_prediction_file)

# ============================================================================================================================================
# === Explaining prediction
# ============================================================================================================================================
class person_class():
	def __init__(self, image, ob_id):
		self.image  = image
		self.ob_id  = ob_id
		self.box    = None
		self.box_width = None
		self.box_height = None
		self.boundaries  = None
		self.cropped_image = None
		self.list_of_mutations = []
		self.list_of_mutations_predictions = [] # Store as 1 -> oringinal prediciton, -1 -> wrong or no prediction
		self.list_of_mutations_predictions_confidence = [] # Stores that confidence of the mutation prediction.
		self.explanation_vectors = None
		self.explanation_vectors_weighted = None

# ========= Extract objects detected that are persons
print("v_labels = ", v_labels)
print("len(v_labels) = ", len(v_labels))
if len(boxes) > 0 and 'person' in v_labels:
	print("Found person in image")
	persons_detected = []
	ob_id = 0
	for i in range(len(v_labels)):

		if v_labels[i] == 'person':
			ob_id += 1

			person = person_class(image, ob_id)
			person.cropped_image = image

			box = v_boxes[i]
			person.box = box

			# get coordinates
			y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
			person.boundaries = [y1, x1, x2, y2]

			# calculate width and height of the box
			width, height = x2 - x1, y2 - y1

			person.box_width  = width
			person.box_height = height 

			person.cropped_image[:, 0:y1, :, :]  = 1
			person.cropped_image[:, y2:-1, :, :] = 1
			person.cropped_image[:, :, 0:x1, :]  = 1
			person.cropped_image[:, :, x2:-1, :] = 1

			# print("person_boundaries = ", person_boundaries)
			plt.clf()
			plt.imshow(person.cropped_image.reshape(416, 416, 3))
			plt.grid(False)
			plt.savefig('image_croped.png')
			# input("press me")

			persons_detected.append(person)


# ========= Create mutations for test suite
def mutation_prediction(mutation):
		# make prediction
		yhat = model.predict(mutation)

		boxes = list()
		for ii in range(len(yhat)):
			# decode the output of the network
			boxes += decode_netout(yhat[ii][0], anchors[ii], class_threshold, input_h, input_w)

		# correct the sizes of the bounding boxes for the shape of the image
		# correct_yolo_boxes(boxes, image_h, image_w, input_h, input_w)
		correct_yolo_boxes(boxes, input_h, input_w, input_h, input_w)

		# suppress non-maximal boxes
		# print("len(boxes) = ",len(boxes))
		do_nms(boxes, 0.5)

		# get the details of the detected objects
		v_boxes, v_labels, v_scores, box_classes_scores = get_boxes(boxes, labels, class_threshold)

		if 'person' in v_labels:
			print("Labels's Match")
			return 1 , v_scores[0]/100
		else:
			print("Labels's do not  Match")
			return -1, 0

for person in persons_detected:
	slit_size_ratio_to_box = 5
	slit_size = int(np.floor(person.box_width/slit_size_ratio_to_box))


	# First method for generating mutations: Showing only a small slit from the object.
	for i in range(slit_size_ratio_to_box):
		for j in range(slit_size_ratio_to_box) :
			mutation  = np.copy(person.cropped_image)
			mutation[:, 0:y1+slit_size*i, :, :]  = 1
			mutation[:, y1+slit_size*(i+1):-1, :, :] = 1
			mutation[:, :, 0:x1+slit_size*j, :]  = 1
			mutation[:, :, x1+slit_size*(j+1):-1, :] = 1

			person.list_of_mutations.append(mutation)
			mutation_prediction_flag, mutation_prediction_confidence = mutation_prediction(mutation)
			person.list_of_mutations_predictions.append(mutation_prediction_flag)
			person.list_of_mutations_predictions_confidence.append(mutation_prediction_confidence)


	# Second method for generating mutations: Starting from a corner and covering gradually the rest.
	for i in range(slit_size_ratio_to_box+1):
		mutation  = np.copy(person.cropped_image)
		mutation[:, 0:y1+slit_size*i, 0:x1+slit_size*i, :]  = 1

		person.list_of_mutations.append(mutation)
		mutation_prediction_flag, mutation_prediction_confidence = mutation_prediction(mutation)
		person.list_of_mutations_predictions.append(mutation_prediction_flag)
		person.list_of_mutations_predictions_confidence.append(mutation_prediction_confidence)

	for i in range(slit_size_ratio_to_box+1):
		mutation  = np.copy(person.cropped_image)
		mutation[:, y2-slit_size*i:-1, 0:x1+slit_size*i, :]  = 1

		person.list_of_mutations.append(mutation) 
		mutation_prediction_flag, mutation_prediction_confidence = mutation_prediction(mutation)
		person.list_of_mutations_predictions.append(mutation_prediction_flag)
		person.list_of_mutations_predictions_confidence.append(mutation_prediction_confidence)

	for i in range(slit_size_ratio_to_box+1):
		mutation  = np.copy(person.cropped_image)
		mutation[:, y2-slit_size*i:-1, x2-slit_size*i:-1, :]  = 1

		person.list_of_mutations.append(mutation)
		mutation_prediction_flag, mutation_prediction_confidence = mutation_prediction(mutation)
		person.list_of_mutations_predictions.append(mutation_prediction_flag)
		person.list_of_mutations_predictions_confidence.append(mutation_prediction_confidence)

	for i in range(slit_size_ratio_to_box+1):
		mutation  = np.copy(person.cropped_image)
		mutation[:, 0:y1+slit_size*i, x2-slit_size*i:-1, :]  = 1

		person.list_of_mutations.append(mutation)
		mutation_prediction_flag, mutation_prediction_confidence = mutation_prediction(mutation)
		person.list_of_mutations_predictions.append(mutation_prediction_flag)
		person.list_of_mutations_predictions_confidence.append(mutation_prediction_confidence)

	# Third method for generating mutations: Starting from middle of object and covering gradually the rest.
	for i in range(slit_size_ratio_to_box+1):
		mutation  = np.copy(person.cropped_image)
		mutation[:, int(y1 + person.box_width/2 - slit_size*i/2):int(y1 + person.box_width/2 + slit_size*i/2), int(x1 + person.box_width/2 - slit_size*i/2):int(x1 + person.box_width/2 + slit_size*i/2), :]  = 1
		person.list_of_mutations.append(mutation)
		mutation_prediction_flag, mutation_prediction_confidence = mutation_prediction(mutation)
		person.list_of_mutations_predictions.append(mutation_prediction_flag)
		person.list_of_mutations_predictions_confidence.append(mutation_prediction_confidence)

# ========= Generate explanation
for person in persons_detected:

	# Start matrix of causality_variables_vector:[ak_ep, ak_ef, ak_np, ak_nf] for each pixel in the person 
	# [pixel_height_location, pixel_width_location, mutation_no, causality_variables_vector]
	# print("person.cropped_image.shape = ", person.cropped_image.shape)
	# input("press enter to continue ")
	mutation_shape = person.cropped_image.shape
	# print("mutation_shape = ",mutation_shape)
	person.explanation_vectors = np.zeros((mutation_shape[1],mutation_shape[2],4))
	person.explanation_vectors_weighted = np.zeros((mutation_shape[1],mutation_shape[2],4))
	# print("person.explanation_vectors.shape = ",person.explanation_vectors.shape)

	ak_ep = 0
	ak_ef = 0
	ak_np = 0
	ak_nf = 0

	# Loop over the pixels with in the box of the object 
	y1, x1, x2, y2 = person.boundaries
	# print(y1,x1,x2,y2)

	label_is_correct = True
	pixel_not_masked = True
	# print("person.box_height = ",person.box_height)
	# print("person.box_width = ",person.box_width)
	# print("len(person.list_of_mutations) = ",len(person.list_of_mutations))
	# input("Continue")
	# for i in range(person.box_height):
	# 	for j in range(person.box_width):
	for k in range(len(person.list_of_mutations)):

		mutation = person.list_of_mutations[k]
		mutation_prediction_flag = person.list_of_mutations_predictions[k]
		mutation_prediction_confidence = person.list_of_mutations_predictions_confidence[k]

		plt.clf()
		plt.imshow(mutation.reshape(416, 416, 3))
		plt.grid(False)
		plt.savefig('image_temp.png')

		print("k = ", k)
		# Find which pixels were masked
		boolArr_not_masked = (mutation[:,:,:,0] != 1) | (mutation[:,:,:,1] != 1) | (mutation[:,:,:,2] != 1)
		boolArr_masked     = (mutation[:,:,:,0] == 1) & (mutation[:,:,:,1] == 1) & (mutation[:,:,:,2] == 1)
		print("boolArr_not_masked = ", boolArr_not_masked)
		# print("boolArr_masked = ", boolArr_masked)
		
		# person.explanation_vectors[:, :, 0] = person.explanation_vectors[:, :, 0] + boolArr # ak_ep
		# person.explanation_vectors[:, :, 1] = person.explanation_vectors[:, :, 1] + boolArr # ak_ef
		# person.explanation_vectors[:, :, 2] = person.explanation_vectors[:, :, 2] + boolArr # ak_np
		# person.explanation_vectors[:, :, 3] = person.explanation_vectors[:, :, 3] + boolArr # ak_nf

		# Find label for this mutation
		if person.list_of_mutations_predictions[k] == 1:
			label_is_correct = True
		else:
			label_is_correct = False
		# print("label_is_correct = ", label_is_correct)
		# input("check if pixel is masked or not, the press enter to continue")
		
		# Do the count
		if label_is_correct == True:
			person.explanation_vectors[:, :, 0] = person.explanation_vectors[:, :, 0] + boolArr_not_masked # ak_ep
			person.explanation_vectors[:, :, 2] = person.explanation_vectors[:, :, 2] + boolArr_masked # ak_np

			person.explanation_vectors_weighted[:, :, 0] = person.explanation_vectors_weighted[:, :, 0] + boolArr_not_masked * mutation_prediction_confidence # ak_ep_weighted
			person.explanation_vectors_weighted[:, :, 2] = person.explanation_vectors_weighted[:, :, 2] + boolArr_masked * mutation_prediction_confidence # ak_np_weighted

		if label_is_correct == False:
			person.explanation_vectors[:, :, 1] = person.explanation_vectors[:, :, 1] + boolArr_not_masked # ak_ef
			person.explanation_vectors[:, :, 3] = person.explanation_vectors[:, :, 3] + boolArr_masked # ak_nf

			person.explanation_vectors_weighted[:, :, 1] = person.explanation_vectors_weighted[:, :, 1] + boolArr_not_masked # ak_ef
			person.explanation_vectors_weighted[:, :, 3] = person.explanation_vectors_weighted[:, :, 3] + boolArr_masked # ak_nf

		# print("person.explanation_vectors[:, :, 0] = ", person.explanation_vectors[:, :, 0])
		# print("person.explanation_vectors[:, :, 1] = ", person.explanation_vectors[:, :, 1])
		# print("person.explanation_vectors[:, :, 2] = ", person.explanation_vectors[:, :, 2])
		# print("person.explanation_vectors[:, :, 3] = ", person.explanation_vectors[:, :, 3])

		# input("Press enter to proceed")
		# print("person.explanation_vectors[:, :, 0] = ", person.explanation_vectors[:, :, 0])


	# === Do measures calculation
	ak_ep = person.explanation_vectors[:, :, 0]
	ak_ef = person.explanation_vectors[:, :, 1]
	ak_np = person.explanation_vectors[:, :, 2]
	ak_nf = person.explanation_vectors[:, :, 3]

	ak_ep_weighted = person.explanation_vectors_weighted[:, :, 0]
	ak_np_weighted = person.explanation_vectors_weighted[:, :, 2]


	

	# # Ochiai: ak_ef / (np.sqrt( (ak_ef+ak_nf) * (ak_ef+ak_ep) ))
	# ochiai_measure = ak_ef / (np.sqrt( (ak_ef+ak_nf) * (ak_ef+ak_ep) ))

	# # Tarantula: ( ak_ef/(ak_ef+ak_nf) ) / ( (ak_ef/(ak_ef+ak_nf) + (ak_ep/(ak_ep+ak_np) ) 
	# tarantula_measure = ( ak_ef/(ak_ef+ak_nf) ) / ( (ak_ef/(ak_ef+ak_nf)) + (ak_ep/(ak_ep+ak_np)) ) 

	# # Zoltar: ak_ef / (ak_ef + ak_nf + ak_ep + (1000 * ak_nf * ak_ep) / ak_ef)
	# zoltar_measure = ak_ef / (ak_ef + ak_nf + ak_ep + (1000 * ak_nf * ak_ep) / ak_ef)

	# Wong II: ak_ef - ak_ep
	wong_measure = ak_ef - ak_ep

	# Abanoub: ak_ep
	ag_measure = ak_ep - ak_ef #+ ak_nf - ak_np

	ag_weight_measure = ak_ef#ak_ep_weighted

	# # Create table for sorting out the scoring of the pixels |i, j, score|
	# ochiai_sorted = np.zeros((mutation_shape[1]*mutation_shape[2],3))
	# tarantula_sorted = np.zeros((mutation_shape[1]*mutation_shape[2],3))
	# zoltar_sorted = np.zeros((mutation_shape[1]*mutation_shape[2],3))
	wong_sorted = np.zeros((mutation_shape[1]*mutation_shape[2],3))
	ag_sorted = np.zeros((mutation_shape[1]*mutation_shape[2],3)) 
	ag_weight_sorted = np.zeros((mutation_shape[1]*mutation_shape[2],3)) 



	# Extract elements from matrix and arrange them in decending order
	# Started arranging elements in decending order
	counter = 0
	for i in range(ag_measure.shape[0]):
		for j in range(ag_measure.shape[1]):
			# ochiai_sorted[counter,:]    = i, j,  ochiai_measure[i,j]
			# tarantula_sorted[counter,:] = i, j,  tarantula_measure[i,j]
			# zoltar_sorted[counter,:]    = i, j,  zoltar_measure[i,j]
			wong_sorted[counter,:]      = i, j,  wong_measure[i,j]
			ag_sorted[counter,:]         = i, j,  ag_measure[i,j]
			ag_weight_sorted[counter,:]         = i, j,  ag_weight_measure[i,j] 
			counter += 1

	# print("zoltar_sorted before = ", zoltar_sorted)
	# ochiai_sorted    = ochiai_sorted[(-ochiai_sorted[:, 2]).argsort()]
	# tarantula_sorted = tarantula_sorted[(-tarantula_sorted[:, 2]).argsort()]
	# zoltar_sorted    = zoltar_sorted[(-zoltar_sorted[:, 2]).argsort()]
	wong_sorted      = wong_sorted[(-wong_sorted[:, 2]).argsort()]
	ag_sorted        = ag_sorted[(-ag_sorted[:, 2]).argsort()]
	ag_weight_sorted = ag_weight_sorted[(-ag_weight_sorted[:, 2]).argsort()]
	# print("ag_sorted = ",ag_sorted)
	# print("zoltar_sorted after = ", zoltar_sorted)

	# explained_image_ochiai    = np.zeros((1,mutation_shape[1],mutation_shape[2],mutation_shape[3]))
	# explained_image_tarantula = np.zeros((1,mutation_shape[1],mutation_shape[2],mutation_shape[3]))
	# explained_image_zoltar    = np.zeros((1,mutation_shape[1],mutation_shape[2],mutation_shape[3]))
	explained_image_wong      = np.zeros((1,mutation_shape[1],mutation_shape[2],mutation_shape[3]))
	explained_image_ag        = np.zeros((1,mutation_shape[1],mutation_shape[2],mutation_shape[3]))
	explained_image_ag_weight = np.zeros((1,mutation_shape[1],mutation_shape[2],mutation_shape[3]))
	
	# ochiai_explantation_found_flag = False
	# tarantula_explantation_found_flag = False
	# zoltar_explantation_found_flag = False
	wong_explantation_found_flag = False
	ag_explantation_found_flag = False
	ag_weight_explantation_found_flag = False


	step = 1000
	counter = 0
	while True:
		# Get pixels of highest scoring pixels and work your way down the list
		for counter_2 in range(counter*step, (counter+1)*step):
			i, j, score = ag_sorted[counter_2,:]
			i = int(i)
			j = int(j)
			
	
			# explained_image_ochiai[0,i,j,:]    = image[0,i,j,:]
			# explained_image_tarantula[0,i,j,:] = image[0,i,j,:]
			# explained_image_zoltar[0,i,j,:]    = image[0,i,j,:]
			explained_image_wong[0,i,j,:]      = image[0,i,j,:]
			explained_image_ag[0,i,j,:]        = image[0,i,j,:]
			explained_image_ag_weight[0,i,j,:]        = image[0,i,j,:]

		explained_images = [\
							# [explained_image_ochiai, "explained_image_ochiai.png"],\
							# [explained_image_tarantula, "explained_image_tarantula.png"],\
							# [explained_image_zoltar, "explained_image_zoltar.png"],\
							[explained_image_wong, "explained_image_wong.png"],\
							[explained_image_ag, "explained_image_ag.png"],\
							[explained_image_ag_weight, "explained_image_ag_weight.png"]]
		
		# For the different measures.
		for explained_image, explained_image_name in explained_images:
			# Predict using the revealed pixels
			yhat = model.predict(explained_image)
		
			boxes = list()
			for ii in range(len(yhat)):
				# decode the output of the network
				boxes += decode_netout(yhat[ii][0], anchors[ii], class_threshold, input_h, input_w)

			# correct the sizes of the bounding boxes for the shape of the image
			# correct_yolo_boxes(boxes, image_h, image_w, input_h, input_w)
			correct_yolo_boxes(boxes, input_h, input_w, input_h, input_w)

			# suppress non-maximal boxes
			# print("len(boxes) = ",len(boxes))
			do_nms(boxes, 0.5)

			# get the details of the detected objects
			v_boxes, v_labels, v_scores, box_classes_scores = get_boxes(boxes, labels, class_threshold)
			print("box_classes_scores = ", box_classes_scores)
			try:
				latest_box_classes_score = box_classes_scores[0][0]
			except:
				latest_box_classes_score = 0


			print("latest_box_classes_score = ",latest_box_classes_score)

		
			# if explained_image_name == "explained_image_ochiai.png":
			# 	if ochiai_explantation_found_flag == False:
			# 		if 'person' in v_labels:
			# 			plt.clf()
			# 			plt.imshow(explained_image.reshape(416, 416, 3))
			# 			plt.grid(False)
			# 			plt.savefig(explained_image_name)
			# 			ochiai_explantation_found_flag = True

			# if explained_image_name == "explained_image_tarantula.png":
			# 	if tarantula_explantation_found_flag == False:
			# 		if 'person' in v_labels:
			# 			plt.clf()
			# 			plt.imshow(explained_image.reshape(416, 416, 3))
			# 			plt.grid(False)
			# 			plt.savefig(explained_image_name)
			# 			tarantula_explantation_found_flag = True

			# if explained_image_name == "explained_image_zoltar.png":
			# 	if zoltar_explantation_found_flag == False:
			# 		if 'person' in v_labels:
			# 			plt.clf()
			# 			plt.imshow(explained_image.reshape(416, 416, 3))
			# 			plt.grid(False)
			# 			plt.savefig(explained_image_name)
			# 			zoltar_explantation_found_flag = True

			# Set threshold for classification difference between explained image and original image classification certainity
			difference_in_classification_certainity_threshold = 0.05 

			if explained_image_name == "explained_image_wong.png":
				if wong_explantation_found_flag == False:
					if ('person' in v_labels) and (initial_box_classes_score - latest_box_classes_score < difference_in_classification_certainity_threshold):
						plt.clf()
						plt.imshow(explained_image.reshape(416, 416, 3))
						plt.grid(False)
						plt.savefig(explained_image_name)
						wong_explantation_found_flag = True

			if explained_image_name == "explained_image_ag.png":
				if ag_explantation_found_flag == False:
					if ('person' in v_labels) and (initial_box_classes_score - latest_box_classes_score < difference_in_classification_certainity_threshold):
						plt.clf()
						plt.imshow(explained_image.reshape(416, 416, 3))
						plt.grid(False)
						plt.savefig(explained_image_name)
						ag_explantation_found_flag = True

			if explained_image_name == "explained_image_ag_weight.png":
				if ag_weight_explantation_found_flag == False:
					if ('person' in v_labels) and (initial_box_classes_score - latest_box_classes_score < difference_in_classification_certainity_threshold):
						plt.clf()
						plt.imshow(explained_image.reshape(416, 416, 3))
						plt.grid(False)
						plt.savefig(explained_image_name)
						ag_weight_explantation_found_flag = True



			

		# Break if prediction is correct
		if  ag_explantation_found_flag and ag_weight_explantation_found_flag and wong_explantation_found_flag:
			break 

		# if  ag_explantation_found_flag and \
		    # ochiai_explantation_found_flag and \
		    # tarantula_explantation_found_flag and \
		    # zoltar_explantation_found_flag and \
		    # wong_explantation_found_flag:
			# break
		# if 'person' in v_labels:
		# 	break

		counter += 1
		# input("Press enter for next explanation")






	# for i in range(a.shape[0]):
	#     for j in range(a.shape[1]):
	#         p[counter,:] = i, j, a[i,j]
	#         counter += 1
	# p[0,2] = float('nan')
	# print(p)
	# print("=====================================================")
	# p = p[p[:, 2].argsort()]
	# print(p)

		# print("mutation[:,i,j,0] = ", mutation[:,i,j,0])
		# print("mutation[:,i,j,1] = ", mutation[:,i,j,1])
		# print("mutation[:,i,j,2] = ", mutation[:,i,j,2])
		# input("Continue from print")

		

		################ TODO check the count done below is correct against by diagram of deepcover. 
		# # Do the count
		# if label_is_correct == True and pixel_masked == False:
		# 	person.explanation_vectors[j,i,0] += 1 # ak_ep 
		# 	ak_ep += 1

		# if label_is_correct == False and pixel_masked == False:
		# 	person.explanation_vectors[j,i,1] += 1 # ak_ef
		# 	ak_ef += 1
		
		# if label_is_correct == True and pixel_masked == True:
		# 	person.explanation_vectors[j,i,2] += 1 # ak_np
		# 	ak_np += 1 
		
		# if label_is_correct == False and pixel_masked == True:
		# 	person.explanation_vectors[j,i,3] += 1 # ak_nf
		# 	ak_nf += 1

	# 	print("label_is_correct = ", label_is_correct)
	# 	print("pixel_masked = ", pixel_masked)
	# 	print("ak_ep = ",ak_ep)
	# 	print("ak_ef = ",ak_ef)
	# 	print("ak_np = ",ak_np)
	# 	print("ak_nf = ",ak_nf)
	# input("Press enter for next mutation ...")


	# Do explainability calculation

	# Identify which pixels are of importance.






#  I need to do the HEREEE above 
#  then once I have the matrix of vectors calcuatle one of the measure for each pixel
#  then arrange the order of pixels in decending order
#  then include the pixels gradually and ask for a predcition until you get the prediction.



# print("person.list_of_mutations_predictions = ",person.list_of_mutations_predictions)

		# person.list_of_mutations.append(mutation)

		# plt.clf()
		# plt.imshow(mutation.reshape(416, 416, 3))
		# plt.grid(False)
		# plt.savefig('image_temp.png')
		# input("Press enter for next mutation ...")
		
		# # Loop to predict each of the mutations
		# for mutation in person.list_of_mutations:

		






			# print("v_labels 2 = ",v_labels)
			# input("Press enter for next mutation ...")















#################################################################

# 		if v_labels[i] == "person":   # if label for that box is a person proceed. TODO: Check if v_labels is the right one
# 			ob_id += 1
# 			person_boundaries = 
# 			persons.append(person(ob_id, image, person_boundaries)) # TODO: Create a class called person to take the details of each object detected

# else:
# 	print("COMMENT: No persons detected to try and explain. Try another image.")

# # ========= Generate mutations and test suite for each object detected 
# for person in persons:
# 	# Create mutations preprocess
	
# 	# Evaluate each mutation in yolo and give it a lable 1 or -1


