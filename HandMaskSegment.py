# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 11:34:40 2022

@author: v-zhiqili
"""
import Constant
import cv2
import mediapipe as mp
import numpy as np  
from skimage.draw import line
import math
import copy,os

from getGrid import *
import AuxGrid 
import ReadImage
import SegmentFromDepth,CameraArray
class SegmentMode:
    ColorBasedSegment=1
    DepthBasedSegment=2
    MixedColorDepthSegment=3
    def isColorSegment(mode):
        if(not mode&SegmentMode.ColorBasedSegment==0):
            return True
        return False
    def isDepthSegment(mode):
        if(not mode&SegmentMode.DepthBasedSegment==0):
            return True
        return False
    def parseName(name):
        if name=="color":
            return SegmentMode.ColorBasedSegment
        elif name=="depth":
            return SegmentMode.DepthBasedSegment
        elif name=="mixed":
            return SegmentMode.MixedColorDepthSegment
        else:
            raise Exception("Invalid name for segment mode")
def segmentImage(color_img_to_mask,   
                 depth_img_to_mask,   
                 validate_img = None,depth_threshold=None,mode=SegmentMode.ColorBasedSegment ):           
    color_masks,depth_masks=[],[]
    if SegmentMode.isColorSegment(mode):
        
        for color,depth in zip(color_img_to_mask,depth_img_to_mask):
            results,_=AuxGrid.getSkeletonResults(color)
            maskImage,result_image=None,copy.deepcopy(color)
            div,divx,divy,maskImage=None,None,None,None
            if results.multi_hand_landmarks:    
                for hand_landmarks,handness in zip(results.multi_hand_landmarks,results.multi_handedness):
                    if(not AuxGrid.checkIntegrity(color,hand_landmarks)):
                        continue
                    willGrid,_=AuxGrid.checkGrid(handness.classification[0].label,handness.classification[0].score,color,validate_img)
                    if(willGrid):
                        result_image,div,divx,divy,zero_images_simpleGraph,zero_images,maskImage=console(color,hand_landmarks,False,maskImage,div,divx,divy,nodebug=True)
                        result_image,div,divx,divy,zero_images_simpleGraph,zero_images,maskImage=console(result_image,hand_landmarks,True,maskImage,div,divx,divy,nodebug=True)
            if maskImage is None:
                maskImage = np.zeros((result_image.shape[0],result_image.shape[1], 3), np.uint8)
            color_masks.append(copy.deepcopy(maskImage[:,:,0]))
    if SegmentMode.isDepthSegment(mode):
        if depth_threshold is None:
            raise Exception("You must set depth_threshhold for depth mode")
        depth_masks=SegmentFromDepth.segmentFromDepth(color_img_to_mask, depth_img_to_mask,depth_threshold)
    ######################################################################
    #######################  Get the Final Result   ######################
    ######################################################################
    if mode==SegmentMode.ColorBasedSegment:
        masks=color_masks
    elif mode==SegmentMode.DepthBasedSegment:
        masks=depth_masks
    elif mode==SegmentMode.MixedColorDepthSegment:
        masks=[]
        for color_mask,depth_mask in zip(color_masks,depth_masks):
            mask= copy.deepcopy(color_mask)
            mask[depth_mask==255]=255
            masks.append(mask)
    else:
        raise Exception("You have to select a correct mode")
    return masks


def HandMaskSegment(dataset_path,
                    clip_ids,
                    target_camera,
                    validate_camera,
                    prompt=True,mode=SegmentMode.ColorBasedSegment):
    readImage=ReadImage.ReadImage();
    depth_threshold=None
    if SegmentMode.isDepthSegment(mode):
        # here the threashold need to be parse
        camera_array= CameraArray.CameraArray()
        camera_array_setting=CameraArray.CameraArraySetting(dataset_path,target_camera,validate_camera)
        camera_array.readCameraYaml(camera_array_setting)
        depth_threshold=camera_array.getSegmentHandDis()
    for clip_id in clip_ids:
        readImage.set_read_args(dataset_path,clip_id,target_camera,validate_camera)
        readImage.get_path_of_frame()
        for index in range(readImage.get_frame_number()):
            colors,depths,validate=readImage.read_image(index,False,prompt)
            hand_masks=segmentImage(colors,depths,validate,depth_threshold,mode)
            hand_mask_path=readImage.get_handmask_path(index)
            for path,hand_mask in zip(hand_mask_path,hand_masks):
                if(prompt):
                    print("Target:writing validate view color of cursor %d to "%(index),path)
                cv2.imwrite(path,copy.deepcopy(hand_mask))
        




    
