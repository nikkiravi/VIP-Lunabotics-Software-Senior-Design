# Import Libraries
from detectron2.structures import BoxMode, PolygonMasks
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
import cv2
import sys
import os
import distutils.core
import numpy as np
import json
import random
import pickle
import matplotlib.pyplot as plt
import argparse
import warnings

warnings.filterwarnings("ignore")

# GLOBAL VARIABLES
# dist = distutils.core.run_setup("./detectron2/setup.py")
# sys.path.insert(0, os.path.abspath('./detectron2'))
classes = ["Pebbles", "Large Rock"]
cfg_path = r"cfg_model.pickle"

def parser():
	parser = argparse.ArgumentParser(description = "Semantic Segmentation")
	parser.add_argument("--train", action="store_true", help="Train Model")
	parser.add_argument("--max_iter", type=int, help="Maximum number of iterations to train")
	parser.add_argument("--thresh", type=float, help="Threshold value")
	parser.add_argument("--test", action="store_true", help="Test Model")
	parser.add_argument("--bounding_box", action="store_true", help="Display bounding boxes")
	parser.add_argument("--predict", type=str, help="Provide filename of image to predict semantic segmentation")
	parser.add_argument("--image_path", type=str, help="Path to image")
	args = parser.parse_args()

	return args

def get_data_dicts(path, classes):
	dataset_dicts = []
	files = [file for file in os.listdir(path) if file.endswith(".json")]

	for idx, json_filename in enumerate(files):
		json_file = os.path.join(path, json_filename)
		with open(json_file) as fptr:
			img_annotations = json.load(fptr)

		record = {}
		image_filename = os.path.join(path, img_annotations["imagePath"])
		record["file_name"] = image_filename
		record["image_id"] = idx
		
		height, width = cv2.imread(image_filename).shape[:2]
		record["width"], record["height"] = width, height #720, 480

		annotations = img_annotations["shapes"]
		objs = []
		for ann in annotations:
			px = [a[0] for a in ann["points"]] # x-coordinate (top-left and bottom-right corners)
			py = [a[1] for a in ann["points"]] # y-coordinate (top-left and bottom-right corners)
			
			px = [px[0],px[1],px[0],px[1]]
			py = [py[0],py[0],py[1],py[1]]
			
			poly = [(x, y) for x, y in zip(px, py)]
			poly = [p for x in poly for p in x]

			obj = {"bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
					"bbox_mode": BoxMode.XYXY_ABS,
					"segmentation": [poly],
					"category_id": classes.index(ann['label']),
					"iscrowd": 0}

			objs.append(obj)

		record["annotations"] = objs
		dataset_dicts.append(record)
	return dataset_dicts

def set_up_model(max_iter=500, save=None):
	cfg = get_cfg()
	cfg.MODEL.DEVICE = 'cpu'
	cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
	cfg.DATASETS.TRAIN = ("category_train",)
	cfg.DATASETS.TEST = ()
	cfg.DATALOADER.NUM_WORKERS = 2
	cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
	cfg.SOLVER.IMS_PER_BATCH = 2
	cfg.SOLVER.BASE_LR = 0.00025
	cfg.SOLVER.MAX_ITER = max_iter
	cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
	
	if(save):
		with open(save, 'wb') as fptr:
			pickle.dump(cfg, fptr, protocol=pickle.HIGHEST_PROTOCOL)

	return cfg

def train_model(cfg):
	os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
	trainer = DefaultTrainer(cfg) 
	trainer.resume_or_load(resume=False)
	trainer.train()

def semantic_segmentation(image, labels, masks):
	blank = np.zeros(image.shape)
	blank[:,:] = (139, 10, 80)
	color = None
	for label, mask in zip(labels, masks):
		if(not label):
			color = (255,0,0)
		else:
			color = (0,0,255)
				
		for row in range(len(mask)):
			for col in range(len(mask[row])):
				if(mask[row][col]):
					blank[row][col] = color
	return blank

def visualizer(image, outputs, metadata, filename=None, bounding_box=False):
	mask = outputs["instances"].get("pred_masks")
	label = outputs["instances"].get("pred_classes")
	sem_seg_image = semantic_segmentation(image, label, mask)
	plt.figure(figsize = (14, 10))
	
	if(bounding_box):
		vis = Visualizer(image[:, :, ::-1], metadata=metadata, scale=0.8, instance_mode=ColorMode.SEGMENTATION)
		vis = vis.draw_instance_predictions(outputs["instances"].to("cpu"))
		plt.imshow(cv2.cvtColor(vis.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
	else:
		plt.imshow(sem_seg_image)
	
	if(filename):
		cv2.imwrite(os.path.join("./Results/", filename), sem_seg_image)
	
	plt.show()

def make_inferences(cfg_path, path, metadata, thresh=0.2, bounding_box=True):
	with open(cfg_path, 'rb') as fptr:
		cfg = pickle.load(fptr)
	
	cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
	cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thresh 
	cfg.DATASETS.TEST = ()
	predictor = DefaultPredictor(cfg)

	test_dataset_dicts = get_data_dicts(path+'test', classes)

	for data in random.sample(test_dataset_dicts, len(test_dataset_dicts) - 1):    
		image = cv2.imread(data["file_name"])
		outputs = predictor(image)
		visualizer(image, outputs, metadata, bounding_box=bounding_box)
		
	return predictor

def evaluate(cfg_path, predictor):
	with open(cfg_path, 'rb') as fptr:
		cfg = pickle.load(fptr)
		
	evaluator = COCOEvaluator("category_test", output_dir="./output")
	val_loader = build_detection_test_loader(cfg, "category_test")
	print(inference_on_dataset(predictor.model, val_loader, evaluator))

def prediction(cfg_path, image_path, metadata, filename=None, bounding_box=False, thresh=0.3):
	with open(cfg_path, 'rb') as fptr:
		cfg = pickle.load(fptr)
	
	cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
	cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thresh 
	
	predictor = DefaultPredictor(cfg)
	image = cv2.imread(image_path)
	outputs = predictor(image)
	visualizer(image, outputs, metadata, filename=filename, bounding_box=bounding_box)


def main():
	args = parser()
	path = r"./Images/"
	for d in ["train", "test"]:
		DatasetCatalog.register("category_" + d, lambda d=d: get_data_dicts(path + d, classes))
		MetadataCatalog.get("category_" + d).set(thing_classes=classes)

	microcontroller_metadata = MetadataCatalog.get("category_train")

	if(args.train):
		max_iter = args.max_iter
		cfg = set_up_model(max_iter=max_iter, save=cfg_path)
		train_model(cfg)
	
	if(args.test):
		thresh = args.thresh
		bounding_box = args.bounding_box
		predictor = make_inferences(cfg_path, path, microcontroller_metadata, thresh=thresh, bounding_box=bounding_box)
		evaluate(cfg_path, predictor)

	if(args.predict):
		thresh = args.thresh
		bounding_box = args.bounding_box
		image_path = args.image_path
		prediction(cfg_path, image_path, microcontroller_metadata, filename=args.predict, bounding_box=bounding_box, thresh=thresh)


if __name__ == "__main__":
	main()

