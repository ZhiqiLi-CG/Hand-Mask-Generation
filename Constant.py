# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 16:34:34 2022

@author: v-zhiqili
"""
import cv2
import mediapipe as mp
import numpy as np  
from skimage.draw import line
import math
import copy
###########################################################
#       Here is the constant
##########################################################
bones=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20)]
fingers=[(0,1,2,3),(4,5,6,7),(8,9,10,11),(12,13,14,15),(16,17,18,19)]
ring_length=74.5 # mm A new means of measuring index/ring finger (2D:4D) ratio and its association with gender and hand preference
ring_width=15*1.2
finger_width=[20*1.2,15*1.2,15*1.2,15*1.2,12*1.2]
theta_step=math.pi/9
index=0
mode=0
oneStep=1
Image5HandHistory=[]
Image1HandHistory=[]
def vec3dist(vec1,vec2):
    return ((vec1[0]-vec2[0])**2+(vec1[1]-vec2[1])**2+(vec1[2]-vec2[2])**2)**0.5
def HandDist(landmark1,landmark2):
    dist=0
    for i in range(len(landmark1)):
        dist+=vec3dist(landmark1[i], landmark2[i])
    return dist
def JudgeHand(HandHistory,landmark,handness):
    if(handness.classification[0].score>0.95):
        return handness.classification[0].label
    else:
        if(len(HandHistory)==0):
            return  handness.classification[0].label
        result_hand=None
        for i in range(len(HandHistory)):
            for hand in HandHistory[i]:
                if result_hand is None:
                    result_hand=hand
                elif(HandDist(result_hand[0],landmark)<HandDist(landmark,hand[0])):
                    result_hand=hand
        return result_hand[1].classification[0].label
def AddHand(HandHistory,landmark,handness):
    if(len(HandHistory)==5):
        HandHistory.pop(0)
    HandHistory.append([landmark,handness])
class STATUS():
    SUCCESS  = 0
    FAIL  = 1
    INTERRUPT=2
    CLOSE=3
def vectorInfo(vec):
    length=np.linalg.norm(x=vec)
    edge= vec/length
    edge_verticle=np.array([-edge[1],edge[0]])
    return edge,edge_verticle,length        
###########################################################
#       Description of data structure
###########################################################
#   Graph=[V,E]
class SubGraph:
    """
    The structure of the SubGraph:
        [A,B,C,D,E,F]
        A list: index of vertex
        B list: edge represented by tuple
        C list: levels
        D: vertex1
        E: vertex2
        F: step of length
    """
    def __init__(self):
        self.graph=[[],[],[],None,None,None]
    def getVertex(self):
        return self.graph[0]
    def getEdge(self):
        return self.graph[1]
    def getLevels(self):
        return self.graph[2]
    def getVertex1(self):
        return self.graph[3]
    def getVertex2(self):
        return self.graph[4]
    def getStep(self):
        return self.graph[5]
    def setVertex(self,vertex):
        self.graph[0]=vertex
    def setEdge(self,edge):
        self.graph[1]=edge
    def setLevels(self,levels):
        self.graph[2]=levels
    def setVertex1(self,vertex1):
        self.graph[3]=vertex1
    def setVertex2(self,vertex2):
        self.graph[4]=vertex2
    def setStep(self,step):
        self.graph[5]=step
    def setAll(self,setList):
        for i in range(len(setList)):
            if(setList[i] is not None):
                self.graph[i]=setList[i]    
class Graph:
    """
    Data Member:
        self.vertex_index: the index for the point now
        self.vertex_position: position list for points
        self.vertex_edge: edges adjcent to the vertex
        self.vertex_cost: cost for edges
    Method Member:
        self.reset_graph: reset the graph
        
    """
    ##################################################################
    #######         fundamental function                ##############
    ##################################################################
    def __init__(self):
        self.vertex_index=-1
        self.vertex_position=[]
        self.vertex_edge=[]
        self.vertex_cost=[]
    def reset_graph(self):
        self.vertex_index=-1
        self.vertex_position=[]
        self.vertex_edge=[]
        self.vertex_cost=[]    
    def add_vertex(self,position):
        self.vertex_index+=1;
        self.vertex_position.append((int(position[0]),int(position[1])))
        self.vertex_edge.append([])
        self.vertex_cost.append((0,-1))
        return self.vertex_index 
    def add_edge(self,v1,v2,edge=None):
        if edge is not None:
            edge.append([v1,v2])
        self.vertex_edge[v1].append(v2)
    ##################################################################
    #######         advanced    function                ##############
    ##################################################################
    def connect_upper_level(self,index,vertex_index,upperlevel,edge):
        cadidate_index=[]
        if(index-1>=0):
            cadidate_index.append(index-1)
        cadidate_index.append(index)
        if(index+1<len(upperlevel)):
            cadidate_index.append(index+1)
        for i in range(len(cadidate_index)):
            v=None
            if(cadidate_index[i]<len(upperlevel)):
                v=upperlevel[cadidate_index[i]]
            else:
                v=-1
            if(v!=-1):
                self.add_edge(v,vertex_index,edge)
    def generate_graph(self,vertex1,vertex2,graph_width,step1,step2,height,width,finger_index,bone_index):
        sub_graph1,sub_graph2=SubGraph(),SubGraph()
        edge=np.array([vertex2[0]-vertex1[0],vertex2[1]-vertex1[1]])
        edge,edge_verticle,length=vectorInfo(edge)
        level_number,point_number=int(length/step2)+1,int(graph_width/step1)+1
        levels=[]
        for i in range(level_number):
            vertex_new1=vertex1+edge*i*step2
            level=[]
            for j in range(1,point_number):
                vertex=j*step1*edge_verticle+vertex_new1
                if(Graph.valid_vertex(vertex,height,width)):
                    index= self.add_vertex(vertex)
                    level.append(index)
                    sub_graph1.getVertex().append(index)
                    if(i!=0):
                        self.connect_upper_level(j-1,index,levels[i-1],sub_graph1.getEdge())                
                else:
                    level.append(-1)
            levels.append(level)
        sub_graph1.setAll([None,None,levels,vertex1,vertex2,step1])
        # add the right
        levels=[]
        for i in range(level_number-1,-1,-1):
            vertex_new1=vertex1+edge*i*step2
            level=[]
            for j in range(1,point_number):
                vertex=-j*step1*edge_verticle+vertex_new1
                if(Graph.valid_vertex(vertex,height,width)):
                    index= self.add_vertex(vertex)
                    level.append(index)
                    sub_graph2.getVertex().append(index)
                    if(i!=level_number-1):
                        self.connect_upper_level(j-1,index,levels[level_number-1-i-1],sub_graph2.getEdge())                
                else:
                    level.append(-1)
            levels.append(level)    
        sub_graph2.setAll([None,None,levels,vertex2,vertex1,step1])
        return sub_graph1,sub_graph2
    ##################################################################
    ############         Aux    function                ##############
    ##################################################################
    def valid_vertex(position,height,width):
        if(position[0]>=0 and position[0]<height and position[1]>=0 and position[1]<width):
            return True
        return False
graph=Graph()