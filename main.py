import os
import sys
sys.path.append('Predict')
sys.path.append('ExplanationGeneration')
sys.path.append('FeatruesSpecifications')


from predict_yolo import YoloPredict, YoloPredict_fromPath
from generate_explanation import GenerateExplanation
from extract_palm import ExtractPalm
from extract_face import ExtractFace

from get_yolo_prediction import *


image_path = "FeatruesSpecifications/person_042.jpg"

# load and prepare image
input_w, input_h = 416, 416
image, image_w, image_h = load_image_pixels(image_path, (input_w, input_h))

# Generate preiction
yolo_prediction_coordinates, _, _, _, _ = YoloPredict_fromPath(image_path, input_w, input_h)
print("yolo_prediction_coordinates = ",yolo_prediction_coordinates)

# Generate explanation
explained_image = GenerateExplanation(image_path, input_w, input_h)

# Extract features: Palm
palm_features_coordinates = ExtractPalm(image_path)
print(palm_features_coordinates)

# Extract features: Face
face_features_coordinates = ExtractFace(image_path)
print(face_features_coordinates)

# Trust calculation
CalculateTrust(image, yolo_prediction_coordinates, explained_image, face_features_coordinates, palm_features_coordinates)

Return the trsut metric that we'll use in quantification.