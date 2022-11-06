# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 18:39:03 2022

@author: v-zhiqili
"""

import cv2,os,copy
from argparse import ArgumentParser
import numpy as np

def normlise(x):
    xx=copy.deepcopy(x)
    xx=xx/np.linalg.norm(xx)
    return xx
def crossMul(x1,x2):
    return np.array(
        [x1[1]*x2[2]-x1[2]*x2[1],
         -x1[0]*x2[2]+x1[2]*x2[0],
         x1[0]*x2[1]-x1[1]*x2[0]
            ])
def rayPlaneCross(p,ray,plane,prompt=False):
    #plane=[point,vec1,vec2]
    A=np.zeros((3,3))
    A[0:3,0]=plane[1]
    A[0:3,1]=plane[2]
    A[0:3,2]=ray
    b=p-plane[0]
    x=np.dot(np.linalg.inv(A),b)
    corss_point=x[0]*plane[1]+x[1]*plane[2]+plane[0]
    if prompt:
        print("Compare:",corss_point-p,ray)    
    have_cross=True
    vec1,vec2,vec3=plane[1],plane[2],corss_point-plane[0]
    #print("test",np.dot(vec1,crossMul(vec2, vec3)))
    direction1=crossMul(vec1,vec3)
    direction2=crossMul(vec3,vec2)
    if(np.dot(direction1,direction2)<0):
        have_cross=False
    if x[2]>0:
        have_cross=False
    return corss_point,np.linalg.norm(corss_point-p),have_cross
def rayPyramidCross(p,ray,p0,corners):
    cp,dis=[],[]
    ray=normlise(ray)
    for i in range(0,4):
        plane=[p0,corners[i]-p0,corners[(i+1)%4]-p0]
        corss_point,d,have_cross=rayPlaneCross(p,ray,plane)
        if have_cross:
            cp.append(corss_point),dis.append(d)
    return cp,dis