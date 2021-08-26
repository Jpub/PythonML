#! /usr/bin/env python

import os, sys
import requests, json
import cv2
from utils.utils import get_yolo_box_tfs, makedirs
from utils.bbox import draw_boxes
import numpy as np

anchors=[]
with open('anchors.json') as anchors_str:    
    anchors = json.load(anchors_str)


net_h, net_w = 416, 416 # a multiple of 32, the smaller the faster
obj_thresh, nms_thresh = 0.4, 0.45

TFS_URL="http://localhost:8501/v1/models/yolo3:predict"
img_path = sys.argv[1]
img_data = cv2.imread(img_path)

boxes = get_yolo_box_tfs(TFS_URL, img_data, net_h, net_w, anchors, obj_thresh, nms_thresh)

draw_boxes(img_data, boxes, ["raccoon"], 0) 
cv2.imwrite('./output/' + img_path.split('/')[-1], np.uint8(img_data))  
