# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 16:33:24 2022

@author: v-zhiqili

This file is the adapter for depth segment
"""
import cv2,os,copy
from argparse import ArgumentParser
import numpy as np
import Geometry
import CameraArray

def segmentFromDepth(target_colors,target_depths,target_depth_threshold=None,camera_array_setting=None):
    
    if target_depth_threshold is None:
        camera_array= CameraArray.CameraArray()
        camera_array.readCameraYaml(camera_array_setting.camera_calib_path, camera_array_setting.mask_camera_list, camera_array_setting.validate_camera_list)
        depth_threshold=camera_array.getSegmentHandDis()
    else:
        depth_threshold=copy.deepcopy(target_depth_threshold)
    segment_masks=[]
    for color,depth,dis in zip(target_colors,target_depths,depth_threshold):
        if dis is None:
            segment_masks.append(copy.deepcopy(color))
        else:
            fdepth=depth.astype(np.float64)/1000
            mask=np.ones((color.shape[0],color.shape[1]),dtype= np.uint8)*255
            mask[fdepth>=dis]=0
            mask[fdepth==0]=0
            segment_masks.append(mask)
    return segment_masks