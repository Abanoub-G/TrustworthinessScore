from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2

def ExtractFace(image, image_path):
	shape_predictor_file = "../featrues_specifications/person/shape_predictor_68_face_landmarks.dat"
	# initialize dlib's face detector (HOG-based) and then create
	# the facial landmark predictor
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(shape_predictor_file)

	# load the input image, resize it, and convert it to grayscale
	# image = cv2.imread(image_path)
	# print(image)
	# image = imutils.resize(image, width=416, height=416)
	# image = imutils.resize(image, width=500)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# detect faces in the grayscale image
	rects = detector(gray, 1)

	# loop over the face detections
	annotated_image = image.copy()
	features_coordinates = []
	for (i, rect) in enumerate(rects):
		# determine the facial landmarks for the face region, then
		# convert the landmark (x, y)-coordinates to a NumPy array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
		trigger = True

		# loop over the face parts individually
		for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
			# clone the original image so we can draw on it, then
			# display the name of the face part on the image
			clone = image.copy()
			cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
				0.7, (0, 0, 255), 2)

			# loop over the subset of facial landmarks, drawing the
			# specific face part
			for (x, y) in shape[i:j]:
				cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)

			# extract the ROI of the face region as a separate image
			(x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))

			# Work out the left, top, right, bottom parts here.
			if trigger:
				left = x
				top  = y
				right = x + w
				bottom = y + h
				trigger = False
			left = np.min([x, left])
			top = np.min([y, top])
			right = np.max([x + w, right])
			bottom = np.max([y + h, bottom])
			

		features_coordinates.append([left, top, right, bottom])
		cv2.rectangle(annotated_image, (int(left), int(top)), (int(right), int(bottom)), (0, 0, 230), thickness=2)
	print("===============================================")
	print("Original readings")
	print("left = ", left)
	print("top = ", top)
	print("right = ", right)
	print("bottom = ", bottom)
	print("===============================================")
	
	str_end = image_path[-4:]
	annotated_image_path = image_path[:-4] + "_face_annotation" + str_end
	# cv2.imwrite('test.jpg', cv2.flip(annotated_image2, 1))
	annotated_image_path = "04_face_annotation_cv2.png"
	cv2.imwrite(annotated_image_path, annotated_image)#cv2.flip(annotated_image, 1))
	return features_coordinates
			
# Test this fucntion --> Working
