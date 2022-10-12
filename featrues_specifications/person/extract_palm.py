import cv2
import mediapipe as mp
import numpy as np
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
import person.drawing_utils as mp_drawing
import person.drawing_styles as mp_drawing_styles
mp_hands = mp.solutions.hands

def ExtractPalm(image_path):
	# e.g. image_path = "images/LegsData312.png"
	# For static images:
	IMAGE_FILES = [image_path]
	with mp_hands.Hands(
		static_image_mode=True,
		max_num_hands=2,
		min_detection_confidence=0.5) as hands:
		for idx, file in enumerate(IMAGE_FILES):
			features_coordinates = []
			# Read an image, flip it around y-axis for correct handedness output (see
			# above).
			image = cv2.imread(file)#cv2.flip(cv2.imread(file), 1)
			# Convert the BGR image to RGB before processing.
			results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

			# Print handedness and draw hand landmarks on the image.
			print('Handedness:', results.multi_handedness)
			if not results.multi_hand_landmarks:
				continue
			image_height, image_width, _ = image.shape
			annotated_image = image.copy()
			annotated_image2 = image.copy()
			for hand_landmarks in results.multi_hand_landmarks:
				# print('hand_landmarks:', hand_landmarks)
				# print(
				# 	f'Index finger tip coordinates: (',
				# 	f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
				# 	f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
				# )
				
				idx_to_coordinates = mp_drawing.draw_landmarks(
					annotated_image,
					hand_landmarks,
					mp_hands.HAND_CONNECTIONS,
					mp_drawing_styles.get_default_hand_landmarks_style(),
					mp_drawing_styles.get_default_hand_connections_style())
					
				trigger = True	
				for idx, landmark_px in idx_to_coordinates.items():
					# print(landmark_px)

					if trigger:
						left = landmark_px[0]
						top  = landmark_px[1]
						right = landmark_px[0]
						bottom = landmark_px[1]
						trigger = False
					left = np.min([landmark_px[0], left])
					top = np.min([landmark_px[1], top])
					right = np.max([landmark_px[0], right])
					bottom = np.max([landmark_px[1], bottom])
			
				features_coordinates.append([left, top, right, bottom])
				cv2.rectangle(annotated_image2, (int(left), int(top)), (int(right), int(bottom)), (0, 0, 230), thickness=2)
			str_end = image_path[-4:]
			annotated_image_path = image_path[:-4] + "_palm_annotation" + str_end
			# cv2.imwrite('test.jpg', cv2.flip(annotated_image2, 1))
			annotated_image_path = "05_palm_annotation_cv2.png"
			cv2.imwrite(annotated_image_path, annotated_image2)#cv2.flip(annotated_image2, 1))
	return features_coordinates

# Test this fucntion --> Working