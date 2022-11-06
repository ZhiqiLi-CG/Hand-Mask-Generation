# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 17:09:11 2022

@author: v-zhiqili
"""
import cv2,copy
import numpy as np
import Constant
from Constant import STATUS,vectorInfo,SubGraph,Graph
###########################################################
#       Show the image
##########################################################
def showImage(imageAndName={},waitNum=5):
    for name in imageAndName:
        image=imageAndName[name]
        if image is None or len(image.shape)<2 or image.shape[0]==0 or image.shape[1]==0:
            continue
        cv2.imshow(name, image)
    key=cv2.waitKey(waitNum)
    if  key & 0xFF == 27:
        return STATUS.CLOSE
    if(key==97):
        Constant.index-=1
        if(Constant.index<0):
            Constant.index=0
    if(key==100):
        Constant.index+=1 
    if(key>=0x30 and key <=0x39):
        if(key-0x30!=Constant.mode):
            Constant.index=0
        Constant.mode=key-0x30
    if key == 98:
        Constant.oneStep=1
    if key == 99:
        Constant.oneStep=-1
###########################################################
#       Here is the draw function
##########################################################
def swap_pos(pos):
    return (pos[1],pos[0])
def debug_console(image,hand_landmarks,graph,connections,finger_levels,zero_images,raw_zero_images,paths,simpleGraph,zero_images_simpleGraph,raw_zero_images_simpleGraph,mask_img,index,mode):
    print(mode)
    if mode==0:
        image=drawJoint(index,hand_landmarks,image)
    elif mode==1:
        image=drawSkeleton(index,hand_landmarks,image)
    elif mode==2 or mode==3 or mode==4:
        image=drawSubGraph(index,graph,connections,simpleGraph,image,mode-2)
    elif mode==5:
        image=drawGraphConnections(index,graph,connections,image)
    elif mode==6:
        #image=drawLevel(index,graph,finger_levels,image)
        image=drawMask(index,mask_img,image)
    elif mode==7:
        image=drawGray(index,zero_images,zero_images_simpleGraph,image)
    elif mode==8:
        image=drawRawGray(index,raw_zero_images,raw_zero_images_simpleGraph,image)
    elif mode==9:
        #print(len(paths))
        #print(paths)
        image=drawPath(index,paths,image)
    return image
def drawMask(index,mask_img,image):
    tem=copy.deepcopy(mask_img[:,:,0])
    tem=tem/255
    if(index%2==0):
        image[:,:,0]=image[:,:,0]*tem
        image[:,:,1]=image[:,:,1]*tem
        image[:,:,2]=image[:,:,2]*tem
    else:
        image[:,:,0]=image[:,:,0]*(1-tem)
        image[:,:,1]=image[:,:,1]*(1-tem)
        image[:,:,2]=image[:,:,2]*(1-tem)
    return image 
def drawPath(index,paths,image):
    for index in range(len(paths)):
        path=paths[index]
        for i in range(len(path)):
            if(i!=0):
                v1=Constant.graph.vertex_position[path[i-1]]
                v2=Constant.graph.vertex_position[path[i]]
                image=cv2.line(image, swap_pos(v1), swap_pos(v2), 0, 2)
    return image
def drawRawGray(index,raw_zero_images,raw_zero_images_simpleGraph,image):
    
    tem=copy.deepcopy(raw_zero_images[Constant.index])
    tem=tem/255
    image[:,:,0]=image[:,:,0]*tem
    image[:,:,1]=image[:,:,1]*tem
    image[:,:,2]=image[:,:,2]*tem
    return image 

def drawGray(index,zero_images,zero_images_simpleGraph,image):
    for i in range(len(zero_images_simpleGraph)):
        tem=np.ones(zero_images_simpleGraph[i].shape)
        tem[zero_images_simpleGraph[i]<1]=0
        image[:,:,0]=image[:,:,0]*tem
        image[:,:,1]=image[:,:,1]*tem
        image[:,:,2]=image[:,:,2]*tem
    tem=np.ones(zero_images[index].shape)
    tem[zero_images[index]<1]=0
    #np.savetxt("D:\\ML\\test.txt", zero_images[index],fmt ='%f')
    #exit(1)
    #print("zero_images",zero_images[index])
    #print("tem",tem)
    image[:,:,0]=image[:,:,0]*tem
    image[:,:,1]=image[:,:,1]*tem
    image[:,:,2]=image[:,:,2]*tem
    return image 
def drawLevel(index,graph,finger_levels,image):
    level=finger_levels[0][index][0]
    v1,v2=level[0],level[len(level)-1]
    v1,v2=Constant.graph.vertex_position[v1],Constant.graph.vertex_position[v2]
    image=cv2.line(image, swap_pos(v1), swap_pos(v2), 0, 2)
    return image
def drawGraphConnections(index,graph,connections,image):
    for k in range(0,4):
        sub_graph1,sub_graph2=graph[int(index/4)][k]
        for e in sub_graph1.getEdge():
            v1,v2=e[0],e[1]
            v1,v2=Constant.graph.vertex_position[v1],Constant.graph.vertex_position[v2]
            image=cv2.line(image, swap_pos(v1), swap_pos(v2), 0, 2)
        for e in sub_graph2.getEdge():
            v1,v2=e[0],e[1]
            v1,v2=Constant.graph.vertex_position[v1],Constant.graph.vertex_position[v2]
            image=cv2.line(image, swap_pos(v1), swap_pos(v2), 0, 2)
    for k in range(len(connections)):
        for e in connections[k].getEdge():
            v1,v2=e[0],e[1]
            v1,v2=Constant.graph.vertex_position[v1],Constant.graph.vertex_position[v2]
            image=cv2.line(image, swap_pos(v1), swap_pos(v2), 0, 2)
    return image
def drawSubGraph(index,graph,connections,simpleGraph,image,mode):
    sub_graph1,sub_graph2=graph[int(index/4)][index%4]
    if mode==0:
        for v in sub_graph1.getVertex():
            vertex=Constant.graph.vertex_position[v]    
            image=cv2.circle(image,swap_pos(vertex), 1,0,1)
        for v in sub_graph2.getVertex():
            vertex=Constant.graph.vertex_position[v]    
            image=cv2.circle(image,swap_pos(vertex), 1,0,1)
        for i in range(len(simpleGraph)):
            sub_graph1,sub_graph2=simpleGraph[i]
            for v in sub_graph1.getVertex():
                vertex=Constant.graph.vertex_position[v]    
                image=cv2.circle(image,swap_pos(vertex), 1,0,1)
            for v in sub_graph2.getVertex():
                vertex=Constant.graph.vertex_position[v]    
                image=cv2.circle(image,swap_pos(vertex), 1,0,1)
    elif mode==1:
        for e in sub_graph1.getEdge():
            v1,v2=e[0],e[1]
            v1,v2=Constant.graph.vertex_position[v1],Constant.graph.vertex_position[v2]
            image=cv2.line(image, swap_pos(v1), swap_pos(v2), 0, 1)
        for e in sub_graph2.getEdge():
            v1,v2=e[0],e[1]
            v1,v2=Constant.graph.vertex_position[v1],Constant.graph.vertex_position[v2]
            image=cv2.line(image, swap_pos(v1), swap_pos(v2), 0, 1)
        
        for i in range(len(simpleGraph)):
            sub_graph1,sub_graph2=simpleGraph[i]
            
            for e in sub_graph1.getEdge():
                v1,v2=e[0],e[1]
                v1,v2=Constant.graph.vertex_position[v1],Constant.graph.vertex_position[v2]
                image=cv2.line(image, swap_pos(v1), swap_pos(v2), 0, 1)
            for e in sub_graph2.getEdge():
                v1,v2=e[0],e[1]
                v1,v2=Constant.graph.vertex_position[v1],Constant.graph.vertex_position[v2]
                image=cv2.line(image, swap_pos(v1), swap_pos(v2), 0, 1)
    elif mode==2:
        for level in sub_graph1.getLevels():
            v1,v2=level[0],level[len(level)-1]
            v1,v2=Constant.graph.vertex_position[v1],Constant.graph.vertex_position[v2]
            image=cv2.line(image, swap_pos(v1), swap_pos(v2), 0, 1)
        for level in sub_graph2.getLevels():
            v1,v2=level[0],level[len(level)-1]
            v1,v2=Constant.graph.vertex_position[v1],Constant.graph.vertex_position[v2]
            image=cv2.line(image, swap_pos(v1), swap_pos(v2), 0, 1)
        for connection in connections:
            if connection.getLevels() is not None:
                for level in connection.getLevels():
                    v1,v2=level[0],level[len(level)-1]
                    v1,v2=Constant.graph.vertex_position[v1],Constant.graph.vertex_position[v2]
                    image=cv2.line(image, swap_pos(v1), swap_pos(v2), 0, 1)
    return image
def drawJoint(index,hand_landmarks,image):
    image=cv2.circle(image, swap_pos(hand_landmarks[index]), 5,0,5)
    return image
def drawSkeleton(index,hand_landmarks,image):
    vertex1=Constant.bones[index][0]
    vertex2=Constant.bones[index][1]
    image=cv2.line(image, swap_pos(hand_landmarks[vertex1]), swap_pos(hand_landmarks[vertex2]), 0, 5)
    return image