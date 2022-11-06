# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 15:27:58 2022

@author: v-zhiqili

This file is for video generate

"""
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 11:34:40 2022

@author: v-zhiqili
"""
import Constant
import cv2,os
import mediapipe as mp
import numpy as np  
from skimage.draw import line
import math
import copy

from getGrid import *
import AuxGrid 
import ReadImage

display_size=[(1,1),(1,2),(1,3),(2,3),(2,3),(2,3),(3,3),(3,3),(3,3)]
targetLocation=[(0,0),(0,0),(0,1),(0,1),(0,1),(0,1),(1,1),(1,1),(1,1)]

def get_image_location(validate_number,validate_cur):
    cur=0
    for i in range(display_size[validate_number][0]):
        for j in range(display_size[validate_number][1]):
            if not (i,j)==targetLocation[validate_number]:
                if cur==validate_cur:
                    return (i,j)
                cur+=1

def setSubImage(image,subImage,subImageLocation):
    image[subImageLocation[0]*subImage.shape[0]:(subImageLocation[0]+1)*subImage.shape[0],
          subImageLocation[1]*subImage.shape[1]:(subImageLocation[1]+1)*subImage.shape[1],
          :]=subImage

def setSubImageIndex(image,subImage,index,validate_number):
    location=get_image_location(validate_number, index)
    setSubImage(image,subImage,location)

def setMaskImage(image,targetImage,targetMask,validate_number):
    maskImage = np.zeros((targetImage.shape[0],targetImage.shape[1], 3), np.uint8)
    maskImage[:,:,0]=targetImage[:,:,0]*(1-targetMask[:,:]/255)
    maskImage[:,:,1]=targetImage[:,:,1]*(1-targetMask[:,:]/255)
    maskImage[:,:,2]=targetImage[:,:,2]*(1-targetMask[:,:]/255)
    location=targetLocation[validate_number]
    setSubImage(image,maskImage,location)

def VideoGenerate(dataset_path,video_root,clip_ids,target_camera,validate_camera,fps=15,prompt=True):
    readImage=ReadImage.ReadImage();
    dataset_name=os.path.split(dataset_path)[1]
    
    for clip_id in clip_ids:
        readImage.set_read_args(dataset_path,clip_id,target_camera,validate_camera)
        readImage.get_path_of_frame()
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        videoWriterList,video_paths,video_image_shapes=[None for cam in target_camera],[],[None for cam in target_camera]
        
        for cam in target_camera:
            video_dir=os.path.join(video_root,dataset_name,str(cam))
            if(not os.path.exists(video_dir)):
                os.makedirs(video_dir)
            video_path=os.path.join(video_dir,str(clip_id)+".avi")
            video_paths.append(video_path)
        
        for index in range(readImage.get_frame_number()):
            colors,depths,handmasks,validate_imgs=readImage.read_image(index,True,prompt)
            validate_imgs=[(AuxGrid.getSkeletonResults(validate_img))[1] for validate_img in validate_imgs]
            colors=[(AuxGrid.getSkeletonResults(color))[1] for color in colors]
            for i in range(len(target_camera)):
                if(videoWriterList[i] is None):
                    video_raw_shape=(colors[i].shape[0]*display_size[len(validate_camera)][0],
                                     colors[i].shape[1]*display_size[len(validate_camera)][1])
                    
                    video_image_shapes[i]=video_raw_shape
                    videoWriterList[i]=cv2.VideoWriter(video_paths[i],fourcc,fps,
                                                       (video_raw_shape[1],
                                                        video_raw_shape[0]),
                                                       True)
                total_image=np.zeros((video_image_shapes[i][0],video_image_shapes[i][1], 3), np.uint8)
                setMaskImage(total_image,colors[i],handmasks[i],len(validate_camera))
                for k in range(len(validate_imgs)):
                    setSubImageIndex(total_image,validate_imgs[k],k,len(validate_camera))
                print("shape:",total_image.shape)
                videoWriterList[i].write(copy.deepcopy(total_image))
        for video_writer in videoWriterList:
            video_writer.release() 
