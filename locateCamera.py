# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 13:33:17 2022

@author: v-zhiqili
"""

import cv2,os
from argparse import ArgumentParser
import numpy as np
import glWindow,copy
from glWindow import *
import Geometry
import CameraArray
########################################################################################
################          Set the OpenGL Window          ###########################
########################################################################################

def draw():
    #print("begin draw")
    global leap2Raw,leap1Raw
    #print(glWindow.camera_on_off)
    #print(len(cameraArray.camera_points))
    glWindow.drawSetDynamics()
    #glWindow.drawAxis()
    visualiseCamera2(cameraArray)
    glutSwapBuffers()                    # 切换缓冲区，以显示绘制内容
    
    #print("end draw")
def idel():
    #print("begin idel")
    #print(glWindow.camera_on_off)
    #if(len(cameraArray.camera_points)>0):
    #    print(cameraArray.camera_points)
    #print(len(cameraArray.camera_points))
    if(not (glWindow.clip_Index == cameraArray.clip_id and glWindow.frame_Index == cameraArray.frame_id)):
        cameraArray.clip_id,cameraArray.frame_id=glWindow.clip_Index,glWindow.frame_Index
       # print("cameraArray.clip_id,cameraArray.frame_id",cameraArray.clip_id,cameraArray.frame_id)
        glWindow.clip_Index_max=200#cameraArray.getClipNumber()
        glWindow.frame_Index_max=200#cameraArray.getFrameNumber(glWindow.clip_Index)
    if (glWindow.loadPicture):
        glWindow.loadPicture=False
        cameraArray.camera_points,cameraArray.camera_colors=[],[]
        for camera in cameraArray.camera:
                color_path=cameraArray.getColorPath(camera.camera_id, cameraArray.clip_id, cameraArray.frame_id)
                depth_path=cameraArray.getDepthPath(camera.camera_id, cameraArray.clip_id, cameraArray.frame_id)
                if(color_path == None or depth_path==None):
                    raise Exception("color path %s or depth path %s is None"%(color_path,depth_path))
                camera.loadImage(color_path,depth_path)
                camera.getPoints(10)
        cameraArray.segment()
                # here will filter by the depth
                # TODO
    if(not cameraArray.camera_on_off==glWindow.camera_on_off):
        cameraArray.camera_on_off=copy.deepcopy(glWindow.camera_on_off)
        cameraArray.camera_points,cameraArray.camera_colors=[],[]
        for camera in cameraArray.camera:
            print(camera.camera_id)
            if(camera.camera_id<len(cameraArray.camera_on_off) and cameraArray.camera_on_off[camera.camera_id]):
                cameraArray.camera_points+=camera.points
                cameraArray.camera_colors+=camera.colors
    displayColor()
    drawInformation()
    glutPostRedisplay()
    #print("end idel")
def initGLWindow():
    glutInit()
    displayMode = GLUT_DOUBLE | GLUT_ALPHA | GLUT_DEPTH
    glutInitDisplayMode(displayMode)
    glutInitWindowSize(WIN_W, WIN_H)
    glutInitWindowPosition(300, 200)
    glutCreateWindow('Quidam Of OpenGL')
    init()                              # 初始化画布
    glutIdleFunc(idel)
    glutDisplayFunc(draw)               # 注册回调函数draw()
    glutReshapeFunc(glWindow.reshape)            # 注册响应窗口改变的函数reshape()
    glutMouseFunc(glWindow.mouseclick)           # 注册响应鼠标点击的函数mouseclick()
    glutMotionFunc(glWindow.mousemotion)         # 注册响应鼠标拖拽的函数mousemotion()
    glutKeyboardFunc(glWindow.keydown)           # 注册键盘输入的函数keydown()
    #print("begin Yaml")
    glutMainLoop()                      # 进入glut主循环      

########################################################################################
################                   Visual                    ###########################
########################################################################################
def visualiseCamera2(cameraArray):
    for camera in cameraArray.camera: 
        o=camera.findLocation()
        x,y,z=camera.findAxis()
        drawAxis(x,y,z,o,0.1)
        corners=camera.findPyramid()
        for corner in corners:
            if(camera.camera_id<len(glWindow.camera_on_off) and glWindow.camera_on_off[camera.camera_id]):
                drawLine(o, corner, 5, (1.0,1.0,1.0,0.3))
            else:
                drawLine(o, corner, 0.2, (1.0,1.0,1.0,0.3))
    # then draw the cross section

    mask_index=glWindow.GLIndex%len(cameraArray.mask_camera)

    ps,diss=cameraArray.getCrossPoint(mask_index)
    p=cameraArray.mask_camera[mask_index].findLocation()
    for final_point,validate_camera in zip(ps,cameraArray.validate_camera):
        p0=validate_camera.findLocation()
        if(final_point is not None):
            drawSegment(p, final_point,(1.0,0.0,1.0))
            drawSegment(p0, final_point,(1.0,0.0,1.0))
            
    # then draw the points
    glPointSize(10)
    glBegin(GL_POINTS)
    for p,c in zip(cameraArray.camera_points,cameraArray.camera_colors):
        glColor3f(c[0]/255,c[1]/255,c[2]/255)
        #print(c[0]/255,c[1]/255,c[2]/255)
        #glColor4f(1.0,1.0,1.0,1.0)
        #print(p[0],p[1],p[2])
        glVertex3f(p[0],p[1],p[2])  
    glEnd() 
def displayColor():
    # in this file the display the color
    for camera in cameraArray.camera:
        if (camera in cameraArray.mask_camera and 
            camera.camera_id<len(glWindow.camera_on_off) and
            glWindow.camera_on_off[camera.camera_id]):
            resized = cv2.resize
            if glWindow.begin_hand_segment:
                cv2.imshow("camera_"+str(camera.camera_id), cv2.resize(camera.segmentColor,(320,240)))
            else:
                cv2.imshow("camera_"+str(camera.camera_id), cv2.resize(camera.color,(320,240)))
        elif(cv2.getWindowProperty("camera_"+str(camera.camera_id),cv2.WND_PROP_VISIBLE )>0.5):
            cv2.destroyWindow("camera_"+str(camera.camera_id))
def drawInformation():
    img = np.ones((100, 250, 3), np.uint8) * 255

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "Clip %d"%(glWindow.clip_Index), (0, 25), font, 1, (0, 0, 255), 3)
    cv2.putText(img, "Frame %d"%(glWindow.frame_Index), (0, 75), font,1, (0, 0, 255), 3)
    cv2.imshow("information", img)
########################################################################################
################                 Helper                      ###########################
########################################################################################
def parse_list(list_str):
    if(list_str is None):
        return None
    pre_list,ans=list_str.split(","),[]
    for item in pre_list:
        ans.append(int(item))
    return ans


if __name__ == "__main__":

    arg_parse=ArgumentParser()
    arg_parse.add_argument("--path", help="path of the dataset",dest="path",default=None)
    arg_parse.add_argument("--mask_camera", help="mask_camera",dest="mask_camera",default="0,6,8")
    arg_parse.add_argument("--validate_camera", help="validate_camera",dest="validate_camera",default="1,5")
    args=arg_parse.parse_args()
    cameraArray=CameraArray.CameraArray()
    cameraArray.readCameraYaml(CameraArray.CameraArraySetting(args.path,
                               parse_list(args.mask_camera),
                               parse_list(args.validate_camera)))

    initGLWindow()
    #if(args.visual=="True"):
    #    visualiseCamera(cameras)
        