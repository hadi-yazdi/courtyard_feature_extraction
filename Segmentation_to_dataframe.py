#Segmentation to CSV is a code to extract the features in courtyard and make a Dataset

from PIL import Image, ImageDraw, ImageShow
from PIL import ImagePath
import os
import glob
import random
import csv
random.seed(600)

import pandas as pd
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

import tensorflow as tf

from imantics import Polygons, Mask

# Detectron 2 files
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

import pickle

from shapely.geometry import Polygon
from shapely.ops import cascaded_union

from scipy.spatial import ConvexHull
from math import sqrt
from math import atan2, cos, sin, pi
from collections import namedtuple

import time



def splitxy(A): 
    x = [] 
    y = [] 
    for i in range(len(A)): 
        if (i % 2) == 0: 
            x.append(A[i]) 
        else: 
            y.append(A[i]) 
    return x, y




# Source= https://bitbucket.org/william_rusnack/minimumboundingbox/src/master/


def unit_vector(pt0, pt1):
    # returns an unit vector that points in the direction of pt0 to pt1
    dis_0_to_1 = sqrt((pt0[0] - pt1[0])**2 + (pt0[1] - pt1[1])**2)
    return (pt1[0] - pt0[0]) / dis_0_to_1,            (pt1[1] - pt0[1]) / dis_0_to_1


def orthogonal_vector(vector):
    # from vector returns a orthogonal/perpendicular vector of equal length
    return -1 * vector[1], vector[0]


def bounding_area(index, hull):
    unit_vector_p = unit_vector(hull[index], hull[index+1])
    unit_vector_o = orthogonal_vector(unit_vector_p)

    dis_p = tuple(np.dot(unit_vector_p, pt) for pt in hull)
    dis_o = tuple(np.dot(unit_vector_o, pt) for pt in hull)

    min_p = min(dis_p)
    min_o = min(dis_o)
    len_p = max(dis_p) - min_p
    len_o = max(dis_o) - min_o

    return {'area': len_p * len_o,
            'length_parallel': len_p,
            'length_orthogonal': len_o,
            'rectangle_center': (min_p + len_p / 2, min_o + len_o / 2),
            'unit_vector': unit_vector_p,
            }


def to_xy_coordinates(unit_vector_angle, point):
    # returns converted unit vector coordinates in x, y coordinates
    angle_orthogonal = unit_vector_angle + pi / 2
    return point[0] * cos(unit_vector_angle) + point[1] * cos(angle_orthogonal),            point[0] * sin(unit_vector_angle) + point[1] * sin(angle_orthogonal)


def rotate_points(center_of_rotation, angle, points):
    # Requires: center_of_rotation to be a 2d vector. ex: (1.56, -23.4)
    #           angle to be in radians
    #           points to be a list or tuple of points. ex: ((1.56, -23.4), (1.56, -23.4))
    # Effects: rotates a point cloud around the center_of_rotation point by angle
    rot_points = []
    ang = []
    for pt in points:
        diff = tuple([pt[d] - center_of_rotation[d] for d in range(2)])
        diff_angle = atan2(diff[1], diff[0]) + angle
        ang.append(diff_angle)
        diff_length = sqrt(sum([d**2 for d in diff]))
        rot_points.append((center_of_rotation[0] + diff_length * cos(diff_angle),
                           center_of_rotation[1] + diff_length * sin(diff_angle)))

    return rot_points


def rectangle_corners(rectangle):
    # Requires: the output of mon_bounding_rectangle
    # Effects: returns the corner locations of the bounding rectangle
    corner_points = []
    for i1 in (.5, -.5):
        for i2 in (i1, -1 * i1):
            corner_points.append((rectangle['rectangle_center'][0] + i1 * rectangle['length_parallel'],
                            rectangle['rectangle_center'][1] + i2 * rectangle['length_orthogonal']))

    return rotate_points(rectangle['rectangle_center'], rectangle['unit_vector_angle'], corner_points)


BoundingBox = namedtuple('BoundingBox', ('area',
                                         'length_parallel',
                                         'length_orthogonal',
                                         'rectangle_center',
                                         'unit_vector',
                                         'unit_vector_angle',
                                         'corner_points'
                                        )
)

def MinimumBoundingBox(points):
    # Requires: points to be a list or tuple of 2D points. ex: ((5, 2), (3, 4), (6, 8))
    #           needs to be more than 2 points
    # Effects:  returns a namedtuple that contains:
    #               area: area of the rectangle
    #               length_parallel: length of the side that is parallel to unit_vector
    #               length_orthogonal: length of the side that is orthogonal to unit_vector
    #               rectangle_center: coordinates of the rectangle center
    #                   (use rectangle_corners to get the corner points of the rectangle)
    #               unit_vector: direction of the length_parallel side. RADIANS
    #                   (it's orthogonal vector can be found with the orthogonal_vector function
    #               unit_vector_angle: angle of the unit vector
    #               corner_points: set that contains the corners of the rectangle

    if len(points) <= 2: raise ValueError('More than two points required.')

    hull_ordered = [points[index] for index in ConvexHull(points).vertices]
    hull_ordered.append(hull_ordered[0])
    hull_ordered = tuple(hull_ordered)

    min_rectangle = bounding_area(0, hull_ordered)
    for i in range(1, len(hull_ordered)-1):
        rectangle = bounding_area(i, hull_ordered)
        if rectangle['area'] < min_rectangle['area']:
            min_rectangle = rectangle

    min_rectangle['unit_vector_angle'] = atan2(min_rectangle['unit_vector'][1], min_rectangle['unit_vector'][0])
    min_rectangle['rectangle_center'] = to_xy_coordinates(min_rectangle['unit_vector_angle'], min_rectangle['rectangle_center'])

    # this is ugly but a quick hack and is being changed in the speedup branch
    return BoundingBox(
        area = min_rectangle['area'],
        length_parallel = min_rectangle['length_parallel'],
        length_orthogonal = min_rectangle['length_orthogonal'],
        rectangle_center = min_rectangle['rectangle_center'],
        unit_vector = min_rectangle['unit_vector'],
        unit_vector_angle = min_rectangle['unit_vector_angle'],
        corner_points = set(rectangle_corners(min_rectangle))
    )

# Source is until here


# Importing the image segmentation models
filename = 'config.pkl'
with open(filename, 'rb') as f:
     cfg = pickle.load(f)
predictor = DefaultPredictor(cfg)

filename_offset = 'config_offset.pkl'
with open(filename_offset, 'rb') as g:
     cfg_offset = pickle.load(g)
predictor_offset = DefaultPredictor(cfg_offset)


# Importing the weather dataset
df_weather = pd.read_csv('weather.csv', index_col=0)
df_weather



# The main segmentation to CSV def
def to_csv(city, Scale, image_angle, df_weather):
    path = city + "_jpg" + "/courtyards/"
    file_name = os.listdir(path)


    Features = ["Courtyard_area(m2)","Shadow_area(m2)","Water_place_area(m2)","Green_area(m2)"]  # Features[pred_classes[i]]
    Columns =  ['Courtyard_ID', "Courtyard_area(m2)","Water_place_area(m2)","Green_area(m2)",'Courtyard_length(m)',
                'Courtyard_width(m)','Courtyard_direction(deg)',
               'Water_place_length(m)','Water_place_width(m)','Water_place_direction(deg)','House_area(m2)',
              'House_length(m)','House_width(m)','House_direction(deg)','City']
    Index = range(len(file_name))
    df = pd.DataFrame(index=Index, columns=Columns)
    value = [ list(df_weather.loc[city, :]) for i in Index]
    df2 = pd.DataFrame(value, index=Index, columns=list(df_weather.columns.values))
    df = pd.concat([df, df2], axis=1)

    for d in file_name:
        
        im = np.array(Image.open(path + d))
        outputs = predictor(im)
        pred_classes = outputs['instances'].get('pred_classes').cpu().numpy()
        masks = outputs['instances'].get('pred_masks').cpu().permute(1, 2, 0).numpy()

        df.loc[file_name.index(d), 'Courtyard_ID'] =  os.path.splitext(d)[0]
        df.loc[file_name.index(d), 'City'] = city
        
        poly = [[],[],[],[]]

        # convert the masks to the polygons
        for i in range(masks.shape[2]):
            if pred_classes[i] != 1:
                polygons = Mask(masks[:, :, i]).polygons()
                if len(polygons.segmentation[0]) > 6:
                    x, y = splitxy(polygons.segmentation[0])
                    xy = list(zip(x, y))
                    polygon_xy = Polygon(xy)
                    poly[pred_classes[i]].append(polygon_xy)

        # merging the polygons to produce one polygon for each class and finding the area of each object
        for i in range(len(poly)):
            if i != 1:
                poly[i] = cascaded_union(poly[i])
                object_area = poly[i].area*(Scale**2)
                df.loc[file_name.index(d), Features[i]] = object_area

        #fitting a rectangle to the objects to extracting the features of them
        if poly[0].geom_type == 'Polygon':
            poly[0] = cascaded_union(poly[0])
            x, y = poly[0].exterior.coords.xy
            points_xy = tuple(zip(x, y))
            rect = MinimumBoundingBox(points_xy)
            df.loc[file_name.index(d), 'Courtyard_width(m)'] = min(rect.length_parallel, rect.length_orthogonal)*Scale
            df.loc[file_name.index(d), 'Courtyard_length(m)'] = max(rect.length_parallel, rect.length_orthogonal)*Scale
            D = (180*rect.unit_vector_angle)/np.pi
            if rect.length_parallel >= rect.length_orthogonal :
                if 0 >= D >= -180 :
                    D += 270
                else:
                    D += 90
            else:
                if -90 >= D >= -180:
                    D += 360
                elif -90 < D < 90:
                    D += 180
                else:
                    continue
            if (D - image_angle) > 270:
                df.loc[file_name.index(d), 'Courtyard_direction(deg)'] = (D - image_angle) - 180
            elif (D - image_angle) < 90:
                df.loc[file_name.index(d), 'Courtyard_direction(deg)'] = (D - image_angle) + 180
            else:
                df.loc[file_name.index(d), 'Courtyard_direction(deg)'] = (D - image_angle)

        if poly[2].geom_type == 'Polygon':
            poly[2] = cascaded_union(poly[2])
            x, y = poly[2].exterior.coords.xy
            points_xy = tuple(zip(x, y))
            rect = MinimumBoundingBox(points_xy)
            df.loc[file_name.index(d), 'Water_place_width(m)'] = min(rect.length_parallel, rect.length_orthogonal)*Scale
            df.loc[file_name.index(d), 'Water_place_length(m)'] = max(rect.length_parallel, rect.length_orthogonal)*Scale
            D = (180*rect.unit_vector_angle)/np.pi
            if rect.length_parallel >= rect.length_orthogonal :
                if 0 >= D >= -180 :
                    D += 270
                else:
                    D += 90
            else:
                if -90 >= D >= -180:
                    D += 360
                elif -90 < D < 90:
                    D += 180
                else:
                    continue 
            if (D - image_angle) > 270:
                df.loc[file_name.index(d), 'Water_place_direction(deg)'] = (D - image_angle) - 180
            elif (D - image_angle) < 90:
                df.loc[file_name.index(d), 'Water_place_direction(deg)'] = (D - image_angle) + 180
            else:
                df.loc[file_name.index(d), 'Water_place_direction(deg)'] = (D - image_angle)

    path = city + "_jpg" + "/courtyards_offset/"    
    file_name = os.listdir(path)

    Features = ["House_area(m2)"]  # Features[pred_classes[i]]

    for d in file_name:    
        im = np.array(Image.open(path + d))
        outputs = predictor_offset(im)
        pred_classes = outputs['instances'].get('pred_classes').cpu().numpy()
        masks = outputs['instances'].get('pred_masks').cpu().permute(1, 2, 0).numpy()

        poly = [[]]

        for i in range(masks.shape[2]):
            if pred_classes[i] == 0:

                polygons = Mask(masks[:, :, i]).polygons()
                if len(polygons.segmentation[0]) > 6:
                    x, y = splitxy(polygons.segmentation[0])
                    xy = list(zip(x, y))
                    polygon_xy = Polygon(xy)
                    poly[pred_classes[i]].append(polygon_xy)

        for i in range(len(poly)):
            poly[i] = cascaded_union(poly[i])
            object_area = poly[i].area*(Scale**2)
            df.loc[file_name.index(d), Features[i]] = object_area

        if poly[0].geom_type == 'Polygon':
            poly[0] = cascaded_union(poly[0])
            x, y = poly[0].exterior.coords.xy
            points_xy = tuple(zip(x, y))
            rect = MinimumBoundingBox(points_xy)
            df.loc[file_name.index(d), 'House_width(m)'] = min(rect.length_parallel, rect.length_orthogonal)*Scale
            df.loc[file_name.index(d), 'House_length(m)'] = max(rect.length_parallel, rect.length_orthogonal)*Scale
            D = (180*rect.unit_vector_angle)/np.pi
            if rect.length_parallel >= rect.length_orthogonal :
                if 0 >= D >= -180 :
                    D += 270
                else:
                    D += 90
            else:
                if -90 >= D >= -180:
                    D += 360
                elif -90 < D < 90:
                    D += 180
                else:
                    continue  
            if (D - image_angle) > 270:
                df.loc[file_name.index(d), 'House_direction(deg)'] = (D - image_angle) - 180
            elif (D - image_angle) < 90:
                df.loc[file_name.index(d), 'House_direction(deg)'] = (D - image_angle) + 180
            else:
                df.loc[file_name.index(d), 'House_direction(deg)'] = (D - image_angle)

    for i in df.index:
        if df.loc[i, 'Courtyard_area(m2)'] == 'nan' or df.loc[i, 'Courtyard_area(m2)'] == 0:
            continue
        else:
            df.loc[i, 'Water_place/Courtyard_area_Ratio'] = df.loc[i, 'Water_place_area(m2)']/df.loc[i, 'Courtyard_area(m2)']
            df.loc[i, 'Green/Courtyard_area_Ratio'] = df.loc[i, 'Green_area(m2)']/df.loc[i, 'Courtyard_area(m2)']
        if df.loc[i, 'House_area(m2)'] == 'nan' or df.loc[i, 'House_area(m2)'] == 0:
            continue
        else:
            df.loc[i, 'Courtyard/House_area_Ratio'] = df.loc[i, 'Courtyard_area(m2)']/df.loc[i, 'House_area(m2)']
            df.loc[i, 'Water_place/House_area_Ratio'] = df.loc[i, 'Water_place_area(m2)']/df.loc[i, 'House_area(m2)']
            df.loc[i, 'Green/House_area_Ratio'] = df.loc[i, 'Green_area(m2)']/df.loc[i, 'House_area(m2)']
        if df.loc[i, 'Courtyard_width(m)'] == 'nan' or df.loc[i, 'Courtyard_width(m)'] == 0:
            continue
        else:
            df.loc[i, 'Courtyard_Ratio'] = df.loc[i, 'Courtyard_length(m)']/df.loc[i, 'Courtyard_width(m)']
        if df.loc[i, 'House_width(m)'] == 'nan' or df.loc[i, 'House_width(m)'] == 0:   
            continue
        else:
            df.loc[i, 'House_Ratio'] = df.loc[i, 'House_length(m)']/df.loc[i, 'House_width(m)']
        if df.loc[i, 'Water_place_width(m)'] == 'nan' or df.loc[i, 'Water_place_width(m)'] == 0:  
            continue
        else:
            df.loc[i, 'Water_place_Ratio'] = df.loc[i, 'Water_place_length(m)']/df.loc[i, 'Water_place_width(m)']
            
    for i in range(1, len(df.columns)):
        if df.columns[i] == "City" or df.columns[i] == "climate_divisions":
            continue
        else:
            df.loc[:, df.columns[i]] = pd.to_numeric(df.loc[:, df.columns[i]], downcast="float")
    
    df = df.replace(0, np.nan)
    
    return df


start_time = time.time() # Strat time calculation


# Runing the model on different cities
Scale = [0.075, 0.083, 0.045]
Angle = [80, -80, +35.23]
cities = ['Abarkooh', 'Birjand', 'Kong']

#Scale = [0.075, 0.083, 0.099, 0.058, 0.045, 0.074, 0.051, 0.087, 0.071]
#Angle = [-4.76, -1.35, +30.39, -31.40, +35.23, -68.90, +26.03, -36.45, -34.60]
# cities = ['Abarkooh', 'Birjand', 'Bushehr', 'Esfehan', 'Kong', 'Meybod', 'Semnan', 'Shiraz', 'Yazd']


df_list = []
for i in range(len(cities)):
    df = to_csv(cities[i], Scale[i], Angle[i], df_weather)
    df.to_csv('Datasets/'+ cities[i] +'.csv', index=False)
    df_list.append(df)
    print(cities[i], '--------->' , 'is finished')


print("--- %s seconds ---" % (time.time() - start_time)) # Print the time consumption



# Merging the dataset of different cities
df = pd.concat(df_list, axis=0)


# Save the dataset to CSV in same directory
df.to_csv('dataset.csv', index=False)






