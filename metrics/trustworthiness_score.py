import matplotlib.pyplot as plt
import imutils
import cv2
from yolo.get_yolo_prediction import *

def CalculateTrustworthiness(image, image_cv2, prediction_coordinates, explained_image, array_of_features, array_of_beta, left, right, top, bottom):
	# print("prediction_image = ", prediction_coordinates)
	# print("explained_image = ", explained_image)
	# print("array_of_features = ", array_of_features)
	# print("array_of_beta = ", array_of_beta)

	# image_cv2 = cv2.imread(image_path)
	# image_cv2 = imutils.resize(image_cv2, width=500)

	# load and prepare image
	# input_w, input_h = 416, 416
	# image, _, _ = load_image_pixels(image_path, (416, 416))


	for feature in array_of_features:
		if len(feature) != 0 :

			# testing transfer
			left   = feature[0][0]
			right  = feature[0][2]
			top    = feature[0][1]
			bottom = feature[0][3]

			feature_explained_image = explained_image.reshape(416, 416, 3)
			feature_explained_image = feature_explained_image[top:bottom, left:right,:]
			# print(explained_image.shape)
			plt.imshow(feature_explained_image)
			plt.grid(False)
			plt.savefig("06_feature_explained_cropped.png")
			print("feature_explained_image = ",feature_explained_image)

			# boolArr_not_masked = (explained_image[:,feature[0][0]:feature[0][2],feature[0][1]:feature[0][3],0] != 1) | \
			# 					 (explained_image[:,feature[0][0]:feature[0][2],feature[0][1]:feature[0][3],1] != 1) | \
			# 					 (explained_image[:,feature[0][0]:feature[0][2],feature[0][1]:feature[0][3],2] != 1)

			# Find which pixels of the explanation are masked
			boolArr_not_masked = (feature_explained_image[:,:,0] != 0) | (feature_explained_image[:,:,1] != 0) | (feature_explained_image[:,:,2] != 0)
			boolArr_masked     = (feature_explained_image[:,:,0] == 0) & (feature_explained_image[:,:,1] == 0) & (feature_explained_image[:,:,2] == 0)

			print("boolArr_not_masked = ",boolArr_not_masked)
			print("boolArr_masked = ",boolArr_masked)

			print("boolArr_not_masked.sum() = ",boolArr_not_masked.sum())
			print("boolArr_masked.sum() = ",boolArr_masked.sum())
			# print("np.array(boolArr_not_masked).sum() =",np.array(boolArr_not_masked).sum())



			# print(boolarr.sum())


			# plt.clf()
			# face_extraction_image = image[0]
			# plt.imshow(face_extraction_image[test_y1:test_y2, test_x1:test_x2,:])
			# plt.plot([test_x1,test_x2],[test_y1,test_y2],color="black")
			# plt.grid(False)
			# plt.savefig('test_00_image.png')

			# cv2.line(image_cv2, (int(test_x1), int(test_y1)), (int(test_x2), int(test_y2)), (0, 0, 230), thickness=2)
			# cv2.imwrite("test_01_image_cv2.png", image_cv2)

			

			# boolArr_not_masked = (explained_image[:,feature[0][0]:feature[0][2],feature[0][1]:feature[0][3],0] != 1) | \
			# 					 (explained_image[:,feature[0][0]:feature[0][2],feature[0][1]:feature[0][3],1] != 1) | \
			# 					 (explained_image[:,feature[0][0]:feature[0][2],feature[0][1]:feature[0][3],2] != 1)

			# print("feature = ", feature)
			# print("feature[0][0] = ", feature[0][0])
			# print("feature[0][1] = ", feature[0][1])
			# print("feature[0][2] = ", feature[0][2])
			# print("feature[0][3] = ", feature[0][3])

			# print("===============================================")
			# print("Transferred readings")
			# print("left = ", feature[0][0])
			# print("top = ", feature[0][1])
			# print("right = ", feature[0][2])
			# print("bottom = ", feature[0][3])
			# print("===============================================")

			

			# temp = image
			# temp = temp.reshape(416, 416, 3)
			# plt.clf()
			# plt.imshow(temp)
			# plt.grid(False)
			# plt.savefig("image_temp.png")

			# annotated_image = image_cv2.copy()
			# feature1 = feature[0][1]
			# feature2 = feature[0][3]
			# print("feature1 = ",feature1)
			# print("feature2 = ",feature2)


			# cv2.rectangle(annotated_image, (int(feature[0][0]), int(feature[0][1])), (int(feature[0][2]), int(feature[0][3])), (0, 0, 230), thickness=2)
			# cv2.imwrite("06_face_cv2.png", annotated_image)

			# image_cropped = image.copy()
			# image_cropped = image_cropped.reshape(416, 416, 3)
			# image_cropped = image_cropped[int(top):int(bottom),int(left):int(right),:]
			# plt.clf()
			# plt.imshow(image_cropped)
			# plt.grid(False)
			# plt.savefig("06_prediction_tensorflow.png")

	
			
			
			# plt.clf()
			# temp = explained_image
			# explained_image = explained_image.reshape(416, 416, 3)
			# explained_image = explained_image[feature[0][0]:feature[0][2],feature[0][1]:feature[0][2],:]
			# # print(explained_image.shape)
			# plt.imshow(explained_image)
			# plt.grid(False)
			# plt.savefig("07_feature_explained_cropped.png")
			# cv2.rectangle(explained_image, (int(feature[0][0]), int(feature[0][1])), (int(feature[0][2]), int(feature[0][3])), (0, 0, 230), thickness=2)
			# cv2.imwrite("2.png", explained_image)

		else:
			print("Feature not detected")
		

	# Find which pixels of the explanation are masked
	# boolArr_not_masked = (explained_image[:,:,:,0] != 1) | (explained_image[:,:,:,1] != 1) | (explained_image[:,:,:,2] != 1)
	# boolArr_masked     = (explained_image[:,:,:,0] == 1) & (explained_image[:,:,:,1] == 1) & (explained_image[:,:,:,2] == 1)

	# Extract which pixels overlap between not masked and explanation boundaries
	# print("explained_image.shape() =",explained_image.shape)
	# plt.clf()
	# plt.imshow(explained_image.reshape(416, 416, 3))
	# plt.grid(False)
	# plt.savefig("explained_image.png")

