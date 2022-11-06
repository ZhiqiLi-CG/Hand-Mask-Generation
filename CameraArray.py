# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 18:40:35 2022

@author: v-zhiqili
"""
import cv2,os,copy
from argparse import ArgumentParser
import numpy as np
import Geometry

class CameraArraySetting:
    def __init__(self,dataset_path,mask_camera_list, validate_camera_list):
        self.dataset_path=dataset_path
        self.mask_camera_list=mask_camera_list 
        self.validate_camera_list=validate_camera_list
        
class CameraArray:
    def __init__(self):
        self.mask_camera=None
        self.validate_camera=None
        self.camera=None
        self.dataset_path=None
        self.calib_path=None
        self.mask_camera_dis=None
        
        # the following is for visual
        self.frame_id=-1
        self.clip_id=-1
        self.camera_on_off=[False for i in range(0,10)]
        self.camera_points=[]
        self.camera_colors=[]
        
    def getSegmentHandDis(self):
        self.mask_camera_dis=[]
        for mask_index in range(len(self.mask_camera)):
            ps,diss=self.getCrossPoint(mask_index)
            minValue=None
            for dis in diss:
                if dis is not None:
                    if minValue is None or dis<minValue:
                        minValue=dis
            if minValue is None:
                print("Warning: segment hand for mask_camera %d fails"%(mask_index))
            self.mask_camera_dis.append(minValue)
        return copy.deepcopy(self.mask_camera_dis)
    def segment(self):
        if self.mask_camera_dis is None:
            self.getSegmentHandDis()
        for dis,mask_camera in zip(self.mask_camera_dis,self.mask_camera):
            mask_camera.segment(dis)
    def getCrossPoint(self,mask_camera_id):
        mask_camera=self.mask_camera[mask_camera_id]
        p=mask_camera.findLocation()
        _,_,ray=mask_camera.findAxis()
        cross_points=[]
        cross_dis=[]
        for validate_camera in self.validate_camera:
            p0=validate_camera.findLocation()
            corners=validate_camera.findPyramid()
            cps,diss=Geometry.rayPyramidCross(p,ray,p0,corners)
            final_point,final_dis=None,None
            for cp,dis in zip(cps,diss):
                if final_dis is None or dis<final_dis:
                    final_point,final_dis=cp,dis
            
            cross_points.append(final_point)
            cross_dis.append(final_dis)
        return cross_points,cross_dis
    def getClipNumber(self):
        camera_id=self.camera[0].camera_id
        clip_id=0
        while(True):
            path = os.path.join(self.dataset_path, 
                                "%02d_rgbd"%(camera_id),
                                "clip%d"%(clip_id))
            if(not os.path.exists(path)):
                return clip_id
            clip_id+=1
            if(clip_id>100):
                return -1
    def getFrameNumber(self,clip_id):
        if(clip_id<0):
            return -1
        frame_id=0
        while(True):
            if ( self.getColorPath(self.camera[0].camera_id,0,frame_id)==0):
                return frame_id
            frame_id+=1
        
    def getColorPath(self,camera_id,clip_id,frame_id):
        path1 = os.path.join(self.dataset_path, 
                            "%02d_rgbd"%(camera_id),
                            "clip%d"%(clip_id),
                            "%06d_color.png"%(frame_id)
                            )
        path2 = os.path.join(self.dataset_path, 
                            "%02d_rgbd"%(camera_id),
                            "clip%d"%(clip_id),
                            "%06d_color.jpeg"%(frame_id)
                            )
        if(os.path.exists(path1)):
            return path1
        elif(os.path.exists(path2)):
            return path2
        return None
    def getDepthPath(self,camera_id,clip_id,frame_id):
        path1 = os.path.join(self.dataset_path, 
                            "%02d_rgbd"%(camera_id),
                            "clip%d"%(clip_id),
                            "%06d_depth.png"%(frame_id)
                            )
        if(os.path.exists(path1)):
            return path1
        return None
    def readCameraYaml(self,cameraArraySetting):
        print("begin Yaml")
        self.dataset_path,self.calib_path=cameraArraySetting.dataset_path,os.path.join(cameraArraySetting.dataset_path,"calib.yaml")
        fs = cv2.FileStorage(self.calib_path, cv2.FILE_STORAGE_READ)
        fn = fs.getNode("camera")
        self.mask_camera,self.validate_camera=[],[]
        
        for i in range(fn.size()):
            if(i in cameraArraySetting.mask_camera_list or i in cameraArraySetting.validate_camera_list):
                camera=Camera()
                camera.camera_id=i
                camera.extrinsic=fn.at(i).getNode("extrinsic").mat()
                camera.intrinsic=fn.at(i).getNode("intrinsic").mat()
                camera.setSize(fn.at(i).getNode("height").real()
                               ,fn.at(i).getNode("width").real())
                if i in cameraArraySetting.mask_camera_list:
                    self.mask_camera.append(camera)
                if i in cameraArraySetting.validate_camera_list:
                    self.validate_camera.append(camera)
        self.camera=self.mask_camera+self.validate_camera
        print("end Yaml")
        #print("Note",self.camera,self.mask_camera,self.validate_camera)
class Camera:
    def __init__(self):
        self.intrinsic=None
        self.extrinsic=None
        self.inv_extrinsic=None
        self.height=None   # y
        self.width=None    # x
        self.depth=None
        self.color=None
        self.camera_id=None

        self.depth=None
        self.color=None
        self.segmentColor=None
        
        self.points=[]
        self.colors=[]
        
    def setSize(self,height,width):
        self.height,self.width=height,width
    def findPyramid(self):
        if(self.height is None or self.width is None):
            raise Exception("set the size of the picture before find Pyramid")
        corners_pixel=[
                np.array([0,0,1]).T,
                np.array([self.width,0,1]).T,
                np.array([self.width,self.height,1]).T,
                np.array([0,self.height,1]).T
            ]
        corners_coord=[]
        intrinsic_inv=np.linalg.inv(self.intrinsic)
        for o in corners_pixel:
            corner_new=self.toGlobal(np.dot(intrinsic_inv,o))
            corners_coord.append(corner_new)
        return corners_coord
    def findLocation(self):
        self.inverseExtrinsic()
        return self.inv_extrinsic[0:3,3]
    def findAxis(self):
        self.inverseExtrinsic()
        return self.inv_extrinsic[0:3,0],self.inv_extrinsic[0:3,1],self.inv_extrinsic[0:3,2]
    def toGlobal(self,coord):
        x,y,z=self.findAxis()
        o=self.findLocation()
        return x*coord[0]+y*coord[1]+z*coord[2]+o
    # load image

    def loadImage(self,color_path,depth_path):
        print("load image from:")
        print("    color:%s"%(color_path))
        print("    depth:%s"%(depth_path))
        self.color=cv2.imread(color_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH |cv2.IMREAD_UNCHANGED)
        self.depth=cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH |cv2.IMREAD_UNCHANGED)
    def segment(self,dis):
        if self.color is not None and self.depth is not None:
            #print("test the dis",dis)
            #for i in range(self.depth.shape[0]):
            #    print(self.depth[i,:])
            #print(self.depth>dis)
            fdepth=self.depth.astype(np.float64)/1000
            
            self.segmentColor=np.zeros(self.color.shape,dtype=self.color.dtype)
            self.segmentColor[:,:,0][fdepth>=dis]=self.color[:,:,0][fdepth>=dis]
            self.segmentColor[:,:,1][fdepth>=dis]=self.color[:,:,1][fdepth>=dis]
            self.segmentColor[:,:,2][fdepth>=dis]=self.color[:,:,2][fdepth>=dis]
            self.segmentColor[:,:,0][fdepth==0]=self.color[:,:,0][fdepth==0]
            self.segmentColor[:,:,1][fdepth==0]=self.color[:,:,1][fdepth==0]
            self.segmentColor[:,:,2][fdepth==0]=self.color[:,:,2][fdepth==0]
            #self.segmentColor[:,:,0]=self.color[:,:,0]
            #self.segmentColor[:,:,1]=self.color[:,:,1]            
            #self.segmentColor[:,:,2]=self.color[:,:,2]
    def getPoints(self,level=1):
        intrinsic_inv=np.linalg.inv(self.intrinsic)
        points,color=[],[]
        for x in range(0,int(self.width),level):
            for y in range(0,int(self.height),level):
                index=np.argmax(self.depth[y:y+level,x:x+level])
                indexx,indexy=int(index/level)+y,index%level+x
                d,c=self.depth[indexx,indexy]/1000,self.color[indexx,indexy]
                points.append(self.toGlobal(d*np.dot(intrinsic_inv,np.array([x,y,1]))))
                color.append(c)
                #print(self.toGlobal(d*np.dot(intrinsic_inv,np.array([x,y,1]))))
        self.points,self.colors=points,color
        
    # helper function
    def inverseExtrinsic(self):
        if(self.extrinsic is None):
            raise Exception("set extrinsic first and then find axis")
        R,T=self.extrinsic[0:3,0:3],self.extrinsic[0:3,3]
        inv_R=R.T
        inv_T=-np.dot(inv_R,T)
        self.inv_extrinsic=np.eye(4)
        self.inv_extrinsic[0:3,0:3]=inv_R
        self.inv_extrinsic[0:3,3]=inv_T