# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 17:58:23 2022

@author: v-zhiqili

check whether the hand will be grid of 
"""
import cv2
import mediapipe as mp
import numpy as np  
from skimage.draw import line
import math
import copy
from getGrid import *
import getGrid
def checkIntegrity(image,handmarks):
    image_handmarks=calc_landmark_list(image,handmarks)
    if len(image_handmarks)<21:
        return False
    for point in image_handmarks:
        if point[0]<0 or point[1]<0 or point[0]>=image.shape[0] or point[1]>=image.shape[1]:
            return False
    return True

def getSkeletonResults(image):
    validate_image=copy.deepcopy(image)
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        validate_image.flags.writeable = False
        validate_image = cv2.cvtColor(validate_image, cv2.COLOR_BGR2RGB)
        results = hands.process(validate_image)
    # Draw the hand annotations on the image.
        validate_image.flags.writeable = True
        validate_image = cv2.cvtColor(validate_image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    validate_image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
    return results,validate_image 

def checkExist(validate_image,handness,hand_score,image):
    results,image=getSkeletonResults(validate_image)
    if results.multi_hand_landmarks:
        handnesses={}
        for handness_tem,hand_landmarks in zip(results.multi_handedness,results.multi_hand_landmarks):
            if handness_tem.classification[0].label in handnesses:
                if handness_tem.classification[0].score<handnesses[handness_tem.classification[0].label][0]:
                    handnesses[handness_tem.classification[0].label]=(handness_tem.classification[0].score,hand_landmarks)
            else:
                handnesses[handness_tem.classification[0].label]=(handness_tem.classification[0].score,hand_landmarks)        
        if handness is not None and handness in handnesses and checkIntegrity(validate_image,handnesses[handness][1]) and handnesses[handness][0]>0.9 and hand_score>0.9:
            return False,image
        else:
            return True,image
    else:
        return True,image
    
def checkGrid(handness,hand_score,image,validate_images):
    will_grid,checkImage=True,[]
    for validate_image in validate_images:
        check_grid,check_image=checkExist(validate_image,handness,hand_score,image)
        checkImage.append(checkImage)
        if not check_grid:
            will_grid=False
    return will_grid,checkImage