def CalculateTrust(image, prediction_coordinates, explained_image, face_features_coordinates, palm_features_coordinates):
	# Find which pixels of the explanation are masekd
	boolArr_not_masked = (explained_image[:,:,:,0] != 1) | (explained_image[:,:,:,1] != 1) | (explained_image[:,:,:,2] != 1)
	boolArr_masked     = (explained_image[:,:,:,0] == 1) & (explained_image[:,:,:,1] == 1) & (explained_image[:,:,:,2] == 1)

	# Extract which pixels overlap between not masked and explanation boundaries

