import torch
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)


import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog


import os
import numpy as np
import json
from detectron2.structures import BoxMode

from detectron2.data import DatasetCatalog, MetadataCatalog

import random





print("======================")
print("Check GPU is info")
print("======================")
print("How many GPUs are there? Answer:",torch.cuda.device_count())
print("The Current GPU:",torch.cuda.current_device())
print("The Name Of The Current GPU",torch.cuda.get_device_name(torch.cuda.current_device()))
# Is PyTorch using a GPU?
print("Does Pytorch have access to GPU? Answer:",torch.cuda.is_available())


# switch to False to use CPU
use_cuda = True

use_cuda = use_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu");


print("Is Pytorch currently using GPU (cuda) or CPU? Answer:", device)

print("======================")


# ==========================================================
# == Register dataset 
# ==========================================================


def get_microcontroller_dicts(directory):
	classes = ['Raspberry_Pi_3', 'Arduino_Nano', 'ESP8266', 'Heltec_ESP32_Lora']
	dataset_dicts = []
	for idx, filename in enumerate([file for file in os.listdir(directory) if file.endswith('.json')]):
		json_file = os.path.join(directory, filename)
		with open(json_file) as f:
			img_anns = json.load(f)

		record = {}
		
		filename = os.path.join(directory, img_anns["imagePath"])
		
		record["file_name"] = filename
		record["image_id"] = idx
		record["height"] = 600
		record["width"] = 800

		annos = img_anns["shapes"]
		objs = []
		for anno in annos:
			px = [a[0] for a in anno['points']]
			py = [a[1] for a in anno['points']]
			poly = [(x, y) for x, y in zip(px, py)]
			poly = [p for x in poly for p in x]

			obj = {
				"bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
				"bbox_mode": BoxMode.XYXY_ABS,
				"segmentation": [poly],
				"category_id": classes.index(anno['label']),
				"iscrowd": 0
			}
			objs.append(obj)
		record["annotations"] = objs
		dataset_dicts.append(record)
	return dataset_dicts

for d in ["train", "test"]:
	DatasetCatalog.register("microcontroller_" + d, lambda d=d: get_microcontroller_dicts('Microcontroller_Segmentation/' + d))
	MetadataCatalog.get("microcontroller_" + d).set(thing_classes=['Raspberry_Pi_3', 'Arduino_Nano', 'ESP8266', 'Heltec_ESP32_Lora'])
microcontroller_metadata = MetadataCatalog.get("microcontroller_train")


dataset_dicts = get_microcontroller_dicts("Microcontroller_Segmentation/train")
for d in random.sample(dataset_dicts, 3):
	img = cv2.imread(d["file_name"])
	v = Visualizer(img[:, :, ::-1], metadata=microcontroller_metadata, scale=0.5)
	v = v.draw_dataset_dict(d)
	plt.figure(figsize = (14, 10))
	plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
	plt.savefig("temp.png")
	input("Press enter to proceed")

input("Press enter to proceed to Model Training")


# ==========================================================
# == Train Model
# ==========================================================

from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg

cfg = get_cfg()
# print(cfg)

cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

cfg.DATASETS.TRAIN = ("microcontroller_train",)

cfg.DATASETS.TEST = ()

cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 1000
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

trainer = DefaultTrainer(cfg) 

trainer.resume_or_load(resume=False)
input("Press eneter 9 ======================================================================================================================================")
trainer.train()

input("Press enter to proceed to Inference")

# ==========================================================
# == Use model for inference
# ==========================================================

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 
cfg.DATASETS.TEST = ("microcontroller_test", )
predictor = DefaultPredictor(cfg)

from detectron2.utils.visualizer import ColorMode
dataset_dicts = get_microcontroller_dicts('Microcontroller_Segmentation/test')
for d in random.sample(dataset_dicts, 3):    
	im = cv2.imread(d["file_name"])
	outputs = predictor(im)
	v = Visualizer(im[:, :, ::-1],
				metadata=microcontroller_metadata, 
				scale=0.8, 
				instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
	)
	v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
	plt.figure(figsize = (14, 10))
	plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
	plt.savefig("temp.png")
	input("Press enter to proceed")