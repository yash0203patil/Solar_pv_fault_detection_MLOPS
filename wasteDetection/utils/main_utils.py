import cv2
import numpy as np
import pandas as pd
import math
import torch
from datetime import datetime
import sys
import yaml
import os
import base64

from wasteDetection.exception import SignException
from wasteDetection.logger import logging

def box_label(image, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
    lw = max(round(sum(image.shape) / 2 * 0.003), 2)
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
    if label:
        tf = max(lw - 1, 1)  # font thickness
        w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
        outside = p1[1] - h >= 3
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        p1n = (p1[0]+25, p1[1])
        p2n = (p2[0]+25, p2[1])
        cv2.rectangle(image, p1n, p2n, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(image, label, (p1[0]+25,  p1[1] - 2 if outside else p1[1] + h + 2), 0, lw / 3, txt_color, thickness=tf, lineType=cv2.LINE_AA)


def plot_bboxes(image, boxes, labels=[], colors=[], score=False, conf=None):
    if labels == []:
        labels = {0: u'__background__', 1: u'Grid', 2: u'Defect'}
    if colors == []:
        colors = [(0, 255, 0),(255, 0, 0),(89, 161, 197)]
    for box in boxes:
        if score:
            label = labels[int(box[-1])+1] + " " + str(round(100 * float(box[-2]),1)) + "%"
        else:
            label = labels[int(box[-1])+1]
        if conf:
            if box[-2] > conf:
                color = colors[int(box[-1])]
                box_label(image, box, label, color)
        else:
            color = colors[int(box[-1])]
            box_label(image, box, label, color)


def calculate_nrows(data):
    distances_from_origin = []
    for point in data:
        x, y ,w, h, conf, c = point
        distance = math.sqrt(x**2 + y**2)
        distances_from_origin.append([x, y ,w, h, conf, c, distance])

    df = pd.DataFrame(distances_from_origin).sort_values(6,axis=0).reset_index().drop("index", axis=1)
    
    total_height = math.dist((df.iloc[0,0], df.iloc[0,1]), (df.iloc[0,0], df.iloc[-1, 3]))
    height_rect = math.dist((df.iloc[0,0], df.iloc[0,1]), (df.iloc[0,0], df.iloc[0,3]))
    nrows = int(math.ceil(total_height / height_rect))
    panel_height = int(df.iloc[-1, 3])

    return nrows, panel_height


def sequence(image, data, nrows, panel_height):
    centers = np.array(data)
    d = panel_height / nrows
    for i in range(nrows):
        f = centers[:, 1] - d * i
        a = centers[(f < d) & (f > 0)]
        rows = a[a.argsort(0)[:, 0]]
        yield rows


def sorted_bbox(image, data, nrows, panel_height):
    sorted_array = []
    count = 0
    for row in sequence(image, data, nrows, panel_height):
        for x1, y1, x2, y2, conf, clas  in row:
            count +=1
            sorted_array.append([x1, y1, x2, y2, conf, clas,count])
    return sorted_array, image


def total_defected_area(img, det_data, seg_data):
    distance = []
    for points in det_data:
        x,y,w,h,conf,c = points
        dist = math.sqrt(x**2 + y**2)
        distance.append([ x,y,w,h,conf,c,dist])
    df = pd.DataFrame(distance).sort_values(6,axis=0).reset_index().drop("index",axis=1)

    cord1 = (df.iloc[0,0] , df.iloc[0,1])
    cordh = (df.iloc[0,0] , df.iloc[-1,3])
    cordw = (df.iloc[-1,2] , df.iloc[0,1])
    total_area = math.dist(cord1,cordh)*math.dist(cord1,cordw)
    if len(seg_data) != 0:
        binary_mask = torch.any(seg_data,dim=0).int()*255
        binary_mask= binary_mask.cpu().numpy()
        binary_count = np.unique(binary_mask,return_counts=True)
        blackspot_pixel_count  = binary_count[1][1]
        blackspot_per = round((blackspot_pixel_count / total_area)*100 , 2)
        return blackspot_per, binary_mask,total_area 
    else: 
        blackspot_per = 0
        binary_mask = []
        return blackspot_per, binary_mask ,total_area



def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            logging.info("Read yaml file successfully")
            return yaml.safe_load(yaml_file)

    except Exception as e:
        raise SignException(e, sys) from e
    

def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "w") as file:
            yaml.dump(content, file)
            logging.info("Successfully write_yaml_file")

    except Exception as e:
        raise SignException(e, sys)
    



def decodeImage(imgstring, fileName):
    imgdata = base64.b64decode(imgstring)
    with open("./data/" + fileName, 'wb') as f:
        f.write(imgdata)
        f.close()


def encodeImageIntoBase64(croppedImagePath):
    with open(croppedImagePath, "rb") as f:
        return base64.b64encode(f.read())