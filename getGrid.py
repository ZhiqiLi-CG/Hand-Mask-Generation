# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 16:17:49 2022

@author: v-zhiqili
"""
import cv2
import mediapipe as mp
import numpy as np  
from skimage.draw import line
import math
import copy
from Draw import *
import Constant
from Constant import STATUS,vectorInfo, SubGraph,Graph
###########################################################
#       Here is the auc function
##########################################################

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    for _, landmark in enumerate(landmarks.landmark):
        #landmark_x = min(int(landmark.x * image_width), image_width - 1)
        #landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_x = int(landmark.x * image_width)
        landmark_y = int(landmark.y * image_height)
        landmark_point.append([landmark_y, landmark_x,])
    return landmark_point
def calc_landmark_list_all(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    for _, landmark in enumerate(landmarks.landmark):
        #landmark_x = min(int(landmark.x * image_width), image_width - 1)
        #landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_x = int(landmark.x * image_width)
        landmark_y = int(landmark.y * image_height)
        landmark_xf = landmark.x * image_width
        landmark_yf = landmark.y * image_height
        landmark_zf = landmark.z * image_width
        landmark_point.append([landmark_y, landmark_x,landmark_xf,landmark_yf,landmark_zf])
    return landmark_point
def getDiv(image):
    gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    x = cv2.Sobel(gray,cv2.CV_16S,1,0)
    y = cv2.Sobel(gray,cv2.CV_16S,0,1)
    return x,y,((x)**2+(y)**2)**0.5
###########################################################
#       Here is the function to get Cost
##########################################################
def getEdgeCostPixel(x,y,divx,divy,edge,bone,w,considerWidth,zero_image):
    edge= edge/np.linalg.norm(x=edge)
    item1=abs(-divx[x,y]*edge[1]+divy[x,y]*edge[0])
    item2=1
    if considerWidth==0:
        w=w*4
    item2=math.exp(-zero_image[x,y]**2/(w/4)**2)
    item3=1
    if bone is not None:
        bone=bone/np.linalg.norm(x=bone)
        item3=abs(edge[0]*bone[0]+edge[1]*bone[1])
    item=item1*item2*item3
    return item
def getEdgeCost(divx,divy,vertice1,vertice2,bone,w,considerWidth,zero_image):
    vertice1,vertice2=Constant.graph.vertex_position[vertice1],Constant.graph.vertex_position[vertice2]
    discrete_line = list(zip(*line(*vertice1, *vertice2)))
    edge=np.array(vertice1)-np.array(vertice2)
    sum,n=0,0
    for point in discrete_line:
        sum+=getEdgeCostPixel(point[0],point[1],divx,divy,edge,bone,w,considerWidth,zero_image)
        n+=1
    return sum/n
###########################################################
#       Here is the function to get Width and Step
##########################################################
def getFingerWidth(landmarks):
    # here the landmark is that after converting
    ring=Constant.fingers[3]
    length=0
    skip=True
    for o in ring:
        if skip:
            skip=False
            continue
        vertex1=landmarks[Constant.bones[o][0]]
        vertex2=landmarks[Constant.bones[o][1]]
        length+=((vertex1[2]-vertex2[2])**2+(vertex1[3]-vertex2[3])**2+(vertex1[4]-vertex2[4])**2)**0.5
    ratio=length/Constant.ring_length
    pixel_width=[0,0,0,0,0]
    for i in range(len(pixel_width)):
        pixel_width[i]=int(Constant.finger_width[i]*ratio)
    return pixel_width
def getStep(width,vertex1,vertex2):
    length=np.linalg.norm(x=np.array([vertex1[0]-vertex2[0],vertex1[1]-vertex2[1]]))
    step1,step2=int(width/20),int(length/10)
    if(step1<1):
        step1=1
    if(step2<1):
        step2=1
    return step1,step2
###########################################################
#       Here is the auc function
##########################################################
def cross(v1,v2):
    return v1[0]*v2[1]-v1[1]*v2[0]
def dot(v1,v2):
    return v1[0]*v2[0]+ v1[1]*v2[1]
def chech_rotate(start1,end1,start2,end2,vertical1,vertical2):
    # -1 means cross 
    v1,v2=start1-end1,end2-start2
    sign=cross(v1,v2)*cross(v1,vertical1)
    if sign>0:
        return None
    else:
        return dot(vertical1,vertical2)
###########################################################
#       Here is the connect function
##########################################################
def simpleConnectGraph(subgraph1,subgraph2):
    subGraph=SubGraph()
    edge=[]
    levels1,levels2=subgraph1.getLevels(),subgraph2.getLevels()
    upper_level=levels1[len(levels1)-1]
    lower_level=levels2[0]
    for i in range(len(lower_level)):
        Constant.graph.connect_upper_level(i,lower_level[i],upper_level,edge)
    subGraph.setAll([None,edge,None,None,None,None])
    return subGraph
def rotate_theta(theta,vec):
    new_vec=copy.deepcopy(vec)
    new_vec[0]=math.cos(theta)*vec[0]-math.sin(theta)*vec[1]
    new_vec[1]=math.sin(theta)*vec[0]+math.cos(theta)*vec[1]
    return new_vec
def simpleConnectRound(subgraph1,subgraph2,height,width):
    connect_graph=SubGraph()
    levels1,levels2=subgraph1.getLevels(),subgraph2.getLevels()
    upper_level=levels1[len(levels1)-1]
    lower_level=levels2[0]
    # generate a new graph
    #print(Constant.graph.vertex_position)
    #print(len(Constant.graph.vertex_position))
    #print(levels1,levels2)
    #print(upper_level)
    #print(len(upper_level))
    level_vec1=np.array(Constant.graph.vertex_position[upper_level[len(upper_level)-1]])\
                -np.array(Constant.graph.vertex_position[upper_level[0]])
    level_vec2=np.array(subgraph1.getVertex2())-np.array(subgraph1.getVertex1())
    direction=cross(level_vec1,level_vec2)
    sign=1
    test=rotate_theta(Constant.theta_step,level_vec1)
    if(cross(level_vec1,test)*direction<0):
        sign=-1
    levels=[copy.deepcopy(upper_level)]
    step=subgraph1.getStep()
    for i in range(1,9):
        theta=Constant.theta_step*sign*i
        vec_new=np.array(rotate_theta(theta,level_vec1))
        vec_new_unit=vec_new/np.linalg.norm(vec_new)
        level=[]
        for j in range(len(levels1[0])):
            #rint(vec_new_unit,np.array(subgraph1[4]),j,step)
            vertex=vec_new_unit*j*step+np.array(subgraph1.getVertex2())
            if(Graph.valid_vertex(vertex,height,width)):
                index= Constant.graph.add_vertex(vertex)
                level.append(index)
                connect_graph.getVertex().append(index)
                Constant.graph.connect_upper_level(j,index,levels[i-1],connect_graph.getEdge())
            else:
                level.append(-1)
        levels.append(level)
    levels.append(copy.deepcopy(lower_level))
    for j in range(len(lower_level)):
        Constant.graph.connect_upper_level(j,lower_level[j],levels[len(levels)-2],connect_graph.getEdge())
    connect_graph.setLevels(levels)
    connect_graph.setStep(step)
    return connect_graph

def getParallelBone(vertices1,vertices2,w,shape):
    zero_image=np.ones((shape[0],shape[1]),dtype=np.uint8)*255
    for vertice1,vertice2 in zip(vertices1,vertices2):
        edge=np.array(vertice1)-np.array(vertice2)
        edge_verticle=np.array([-edge[1],edge[0]])
        edge_verticle=edge_verticle/np.linalg.norm(x=edge_verticle)
        vertice11,vertice12=(vertice1+w/2*edge_verticle).astype(np.int16),(vertice2+w/2*edge_verticle).astype(np.int16)
        vertice21,vertice22=(vertice1-w/2*edge_verticle).astype(np.int16),(vertice2-w/2*edge_verticle).astype(np.int16)
        zero_image=cv2.line(zero_image, swap_pos(vertice11), swap_pos(vertice12), 0, 1)
        zero_image=cv2.line(zero_image, swap_pos(vertice21), swap_pos(vertice22), 0, 1)
    raw_zero_image=copy.deepcopy(zero_image)
    zero_image = cv2.distanceTransform(src=zero_image, distanceType=cv2.DIST_L2, maskSize=0)
    return zero_image,raw_zero_image

def getGraph(image, landmarks,hand_landmarks_all,rotatePalm=True):
    # here the landmark is that after converting
    Constant.graph.reset_graph()
    pixel_width=getFingerWidth(hand_landmarks_all)
    graphs=[[None,None,None,None],[None,None,None,None],[None,None,None,None],[None,None,None,None],[None,None,None,None]]
    zero_images,raw_zero_images=[],[]
    zero_images_simpleGraph,raw_zero_images_simpleGraph=[],[]
    for i in range(len(Constant.fingers)):
        bone_width=pixel_width[i];
        #print("bone_width")
        #print(i)
        #print(bone_width)
        vertices1,vertices2=[],[]
        for j in range(len(Constant.fingers[i])):
            vertex1=landmarks[Constant.bones[Constant.fingers[i][j]][0]]
            vertex2=landmarks[Constant.bones[Constant.fingers[i][j]][1]]    
            vertices1.append(vertex1)
            vertices2.append(vertex2)
            step1,step2=getStep(bone_width,vertex1,vertex2)
            sub_graph1,sub_graph2=Constant.graph.generate_graph(vertex1,vertex2,bone_width,step1,step2,image.shape[0],image.shape[1],i,j)
            graphs[i][j]=[sub_graph1,sub_graph2]
        zero_image,raw_zero_image=getParallelBone(vertices1,vertices2,bone_width,image.shape)
        zero_images.append(zero_image)
        raw_zero_images.append(raw_zero_image)
    # here Find the graph to connect the finger     
    simpleGraph=[]
   
    bone_width=pixel_width[0];
    joint=[2,1,1,1,1]
    #print("bone_width",bone_width)
    vertices1,vertices2=[],[]
    
    for i in range(2):
        vertex1=landmarks[Constant.bones[Constant.fingers[0][i]][0]]
        vertex2=landmarks[Constant.bones[Constant.fingers[0][i+1]][0]]    
        vertices1.append(vertex1)
        vertices2.append(vertex2)
        step1,step2=getStep(bone_width,vertex1,vertex2)
        sub_graph1,sub_graph2=Constant.graph.generate_graph(vertex1,vertex2,bone_width,step1,step2,image.shape[0],image.shape[1],i,j)
        simpleGraph.append([sub_graph1,sub_graph2])
    for i in range(len(Constant.fingers)-1):
        vertex1=landmarks[Constant.bones[Constant.fingers[i][joint[i]]][0]]
        vertex2=landmarks[Constant.bones[Constant.fingers[i+1][joint[i+1]]][0]]    
        vertices1.append(vertex1)
        vertices2.append(vertex2)
        step1,step2=getStep(bone_width,vertex1,vertex2)
        sub_graph1,sub_graph2=Constant.graph.generate_graph(vertex1,vertex2,bone_width,step1,step2,image.shape[0],image.shape[1],i,j)
        simpleGraph.append([sub_graph1,sub_graph2])
    for i in range(1):
        vertex1=landmarks[Constant.bones[Constant.fingers[4][1]][0]]
        vertex2=landmarks[Constant.bones[Constant.fingers[4][0]][0]]    
        vertices1.append(vertex1)
        vertices2.append(vertex2)
        step1,step2=getStep(bone_width*4,vertex1,vertex2)
        sub_graph1,sub_graph2=Constant.graph.generate_graph(vertex1,vertex2,bone_width,step1,step2,image.shape[0],image.shape[1],i,j)
        simpleGraph.append([sub_graph1,sub_graph2])
    zero_image,raw_zero_image=getParallelBone(vertices1,vertices2,bone_width,image.shape)
    zero_images_simpleGraph.append(zero_image)
    raw_zero_images_simpleGraph.append(raw_zero_image)
        
    
    # then connect the graph for each fiinger
    # Tofix: the connection should be modifyed
    connections,finger_levels=[],[[],[],[],[],[],[]]
    for i in range(len(graphs)):
        finger_levels[i].append((graphs[i][1][0].getLevels(),graphs[i][1][0].getVertex1(),graphs[i][1][0].getVertex2(),1))
        for j in range(2,len(graphs[i])):
            connections.append(simpleConnectGraph(graphs[i][j-1][0],graphs[i][j][0]))
            finger_levels[i].append((graphs[i][j][0].getLevels(),graphs[i][j][0].getVertex1(),graphs[i][j][0].getVertex2(),1))
        #print("note here!!!:",i,j)
        round_connection=simpleConnectRound(graphs[i][len(graphs[i])-1][0],graphs[i][len(graphs[i])-1][1],image.shape[0],image.shape[1])
        connections.append(round_connection)
        finger_levels[i].append((round_connection.getLevels(),None,None,1))
        for j in range(len(graphs[i])-1,1,-1):
            finger_levels[i].append((graphs[i][j][1].getLevels(),graphs[i][j][1].getVertex1(),graphs[i][j][1].getVertex2(),1))
            connections.append(simpleConnectGraph(graphs[i][j][1],graphs[i][j-1][1]))
        finger_levels[i].append((graphs[i][1][1].getLevels(),graphs[i][1][1].getVertex1(),graphs[i][1][1].getVertex2(),1))
    # then for the palm\
    simpleLength=len(simpleGraph)
    circle_index=1
    begin_index=simpleLength-1
    circle_range=range(simpleLength-2,-1,-1)
    sign_circle=1
    if rotatePalm:
        sign_circle=-1
        circle_index=0
        begin_index=0
        circle_range=range(1,simpleLength)
    finger_levels[5].append((simpleGraph[begin_index][circle_index].getLevels(),simpleGraph[begin_index][circle_index].getVertex1(),simpleGraph[begin_index][circle_index].getVertex2(),0))
    for j in circle_range:
        connections.append(simpleConnectGraph(simpleGraph[j+sign_circle][circle_index],simpleGraph[j][circle_index]))
        finger_levels[5].append((simpleGraph[j][circle_index].getLevels(),simpleGraph[j][circle_index].getVertex1(),simpleGraph[j][circle_index].getVertex2(),1))
        
    for i in range(len(finger_levels)):
        tem=[]
        for j in range(len(finger_levels[i])):
            for k in range(len(finger_levels[i][j][0])):
                tem=tem+[(finger_levels[i][j][0][k],finger_levels[i][j][1],finger_levels[i][j][2],finger_levels[i][j][3])]
        finger_levels[i]=tem
    # then connect the graph for each fiinger
    
    # then add the simple graph to levels
    #for i in range(len(simpleGraph)):
    #    tem,levels=[],simpleGraph[i][1].getLevels()
    #    for j in range(len(levels)):
    #        tem=tem+[(levels[j],None,None)]
    #        #tem=tem+[(levels[j],simpleGraph[i][1].getVertex1(),simpleGraph[i][1].getVertex2())]
    #    finger_levels.append(tem)
    return graphs,connections,finger_levels,zero_images,pixel_width,raw_zero_images,simpleGraph,zero_images_simpleGraph,raw_zero_images_simpleGraph
def findPath(levels,divx,divy,width,zero_image):
    for i in range(len(levels)):
        level=levels[i][0]
        bone_v1,bone_v2=levels[i][1],levels[i][2]
        considerWidth=levels[i][3]
        bone=None
        if(bone_v1 is not None and bone_v2 is not None):
            bone=np.array(bone_v1)-np.array(bone_v2)    
        for v in level:
            if v!=-1:
                for adj in Constant.graph.vertex_edge[v]:
                    new_value=Constant.graph.vertex_cost[v][0]+getEdgeCost(divx,divy,v,adj,bone,width,considerWidth,zero_image)
                    if new_value>Constant.graph.vertex_cost[adj][0]:
                        Constant.graph.vertex_cost[adj]=(new_value,v)
        # then get the path
    level=levels[len(levels)-1][0]
    maxValue,path_v=0,None
    for v in level:
        #print(Constant.graph.vertex_cost[v][0])
        if Constant.graph.vertex_cost[v][0]>maxValue:
            maxValue,path_v=Constant.graph.vertex_cost[v][0],v
    if(maxValue==0):
        raise Exception("Wrong Terminal in finding the path")
        return []
    path=[path_v,]
    i=0
    while(True):
        i+=1
        if(i>100):
            break
        #print(vertex_edge[0],vertex_edge[1],vertex_edge[2],vertex_edge[3])
        #print(path_v)
        if(Constant.graph.vertex_cost[path_v][1]==-1):
            break
        path_v=Constant.graph.vertex_cost[path_v][1]
        path.append(path_v)
    return path
def console(image,hand_landmarks,rotate_circle,before_mask=None,div=None,divx=None,divy=None,nodebug=False):
    #print(Constant.index,Constant.mode)
    if div is None:
        divx,divy,div=getDiv(image)
    hand_landmarks_all=calc_landmark_list_all(image,hand_landmarks)
    hand_landmarks=calc_landmark_list(image,hand_landmarks)

    graph,connections,finger_levels,zero_images,pixel_width,raw_zero_images,simpleGraph,zero_images_simpleGraph,raw_zero_images_simpleGraph=getGraph(image,hand_landmarks,hand_landmarks_all,rotate_circle)
    # here is for path finding 
    paths=[]
    zero_images+=zero_images_simpleGraph
    pixel_width+=[pixel_width[0]]
    for i,levels,zero_image in zip(range(len(finger_levels)),finger_levels,zero_images):
        path=findPath(levels,divx,divy,pixel_width[i],zero_image)
        paths.append(path)
        #print(i,path)    
    paths[5].append(paths[5][0])
    # then connect all the path
    mask_img = np.zeros((image.shape[0],image.shape[1], 3), np.uint8)
    real_paths=[]
    for i in range(len(paths)):
        real_path=[]
        for v in paths[i]:
            real_path.append([Constant.graph.vertex_position[v][1],Constant.graph.vertex_position[v][0]])    
        real_paths.append(np.array(real_path))
        cv2.fillPoly(mask_img,[real_paths[i]],[255,255,255])
    if before_mask is not None:
        mask_img[:,:,0][before_mask[:,:,0]==255]=255
        mask_img[:,:,1][before_mask[:,:,1]==255]=255
        mask_img[:,:,2][before_mask[:,:,2]==255]=255
    if not nodebug:
        image=debug_console(image,hand_landmarks,graph,connections,finger_levels,zero_images,raw_zero_images,paths,simpleGraph,zero_images_simpleGraph,raw_zero_images_simpleGraph,mask_img,Constant.index,Constant.mode)
    return image,div,divx,divy,zero_images_simpleGraph,zero_images,mask_img