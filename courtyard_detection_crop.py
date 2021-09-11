#!/usr/bin/env python
# coding: utf-8

# import the requered libraries

from PIL import Image
import os
import glob
import random
import csv
random.seed(4)

import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import torchvision.transforms as transforms

import tensorflow as tf


Image.MAX_IMAGE_PIXELS = None



# Mapping between Class name and Index
cat_to_index = {'Yard'         : 1}



device      = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
num_classes = 2

# download the model architecture
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)


in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

model.to(device)

# load the saved checkpiont file
checkpoint_path = 'checkpoint_100_100_final.pth'
model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))


# courtyard detection and cropping the in the defenition

def detection_crop(city_jpg):
    path =  city_jpg + "/Subdivided/"
    save_path = city_jpg + "/courtyards/"
    save_path_offset = city_jpg + "/courtyards_offset/"
    file_name = os.listdir(path)

    os.mkdir(save_path)
    os.mkdir(save_path_offset)

    model.eval()

    for fn in file_name:
        test_image = Image.open(path + fn)
        w, h = test_image.size

        folder_name = os.path.splitext(fn)[0]

        transform = transforms.Compose([transforms.ToTensor()])

        img_tensor = transform(test_image)
        images = img_tensor.unsqueeze_(0)

        images = list(image.to(device) for image in images)
        predictions = model(images)

        boxes = predictions[0]['boxes'].data.cpu().numpy()

        for i in range(len(boxes)):
            xmin = int(boxes[i][0] - 15) if int(boxes[i][0] - 15) >=0 else 0
            ymin = int(boxes[i][1] - 15) if int(boxes[i][1] - 15) >=0 else 0
            xmax = int(boxes[i][2] + 15) if int(boxes[i][2] + 15) <=w else w
            ymax = int(boxes[i][3] + 15) if int(boxes[i][3] + 15) <=h else h
            detected_courtyard = test_image.crop((xmin, ymin, xmax, ymax))
            filename_append = str(i)
            detected_courtyard.save(save_path + '{}_{}.jpg'.format(folder_name, filename_append))

            xmin_o = int(boxes[i][0]-(boxes[i][2]-boxes[i][0])) if int(boxes[i][0]-(boxes[i][2]-boxes[i][0])) >=0 else 0
            ymin_o = int(boxes[i][1]-(boxes[i][3]-boxes[i][1])) if int(boxes[i][1]-(boxes[i][3]-boxes[i][1])) >=0 else 0
            xmax_o = int(boxes[i][2]+(boxes[i][2]-boxes[i][0])) if int(boxes[i][2]+(boxes[i][2]-boxes[i][0])) <=w else w
            ymax_o = int(boxes[i][3]+(boxes[i][3]-boxes[i][1])) if int(boxes[i][3]+(boxes[i][3]-boxes[i][1])) <=h else h
            detected_courtyard_o = test_image.crop((xmin_o, ymin_o, xmax_o, ymax_o))
            filename_append = str(i)
            detected_courtyard_o.save(save_path_offset + '{}_{}_{}.jpg'.format(folder_name, filename_append, 'o'))


# run the detection_crop def for the list of cities

cities = ['abarghu', 'birjand', 'bushehr', 'esfehan', 'kong', 'meybod', 'semnan', 'shiraz', 'yazd']

for city in cities:
    detection_crop(city + '_jpg')






