# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 07:21:42 2022

@author: v-zhiqili
"""
import os,cv2
class ReadImage:
    def __init__(self):
        self.dataset_path=None
        self.clip_id=None
        self.target_view_camera=None
        self.validate_view_camera=None
        self.target_view_color_path=[]
        self.target_view_depth_path=[]
        self.target_view_handmask_path=[]
        self.validate_view_color_path=[]
        self.validate_view_depth_path=[]
    
    def set_read_args(self,dataset_path,clip_id,target_view_camera,validate_view_camera):
        self.dataset_path=dataset_path
        self.clip_id=clip_id
        self.target_view_camera=target_view_camera
        self.validate_view_camera=validate_view_camera

    def get_path_of_frame(self):
        self.target_view_color_path=[] # [[cam1_frame1,cam1_frame2,cam1_frame3,...],[cam2_frame1,cam2_frame2,cam2_frame3,...],...]
        self.target_view_depth_path=[]
        self.target_view_handmask_path=[]
        self.validate_view_color_path=[]
        self.validate_view_depth_path=[]
        if(self.dataset_path is None):
            raise Exception("Please set read args first")
        frame_ids_different_camera=[]
        
        for cam in self.target_view_camera:
            clip_dir=os.path.join(self.dataset_path,("%02d"%(cam))+"_rgbd","clip"+str(self.clip_id))
            frame_ids,file_image_type=self.get_frame_ids(clip_dir),self.get_image_type(clip_dir)
            frame_ids_different_camera.append(frame_ids)
            cam_color_dir,cam_depth_dir,cam_handmask_dir=[],[],[]
            for frame_id in frame_ids:    
                color_dir=os.path.join(clip_dir,("%06d"%(frame_id))+"_color."+file_image_type[0])
                depth_dir=os.path.join(clip_dir,("%06d"%(frame_id))+"_depth."+file_image_type[1])
                handmask_dir=os.path.join(clip_dir,("%06d"%(frame_id))+"_hand_mask.png")
                cam_color_dir.append(color_dir)
                cam_depth_dir.append(depth_dir)
                cam_handmask_dir.append(handmask_dir)
            self.target_view_color_path.append(cam_color_dir)
            self.target_view_depth_path.append(cam_depth_dir)
            self.target_view_handmask_path.append(cam_handmask_dir)
            
        for cam in self.validate_view_camera:
            clip_dir=os.path.join(self.dataset_path,("%02d"%(cam))+"_rgbd","clip"+str(self.clip_id))
            frame_ids,file_image_type=self.get_frame_ids(clip_dir),self.get_image_type(clip_dir)
            frame_ids_different_camera.append(frame_ids)
            cam_color_dir,cam_depth_dir=[],[]
            for frame_id in frame_ids:    
                color_dir=os.path.join(clip_dir,("%06d"%(frame_id))+"_color."+file_image_type[0])
                depth_dir=os.path.join(clip_dir,("%06d"%(frame_id))+"_depth."+file_image_type[1])
                cam_color_dir.append(color_dir)
                cam_depth_dir.append(depth_dir)
            self.validate_view_color_path.append(cam_color_dir)
            self.validate_view_depth_path.append(cam_depth_dir)
        self.check_frame_compatity(frame_ids_different_camera)    
        
    def get_frame_number(self):
        if(len(self.target_view_color_path)==0):
            return 0;
        return len(self.target_view_color_path[0])
    
    def get_handmask_path(self,cursor):
        return [ cam_handmask[cursor] for cam_handmask in self.target_view_handmask_path]
    
    def read_image(self,cursor,read_handmask=False,prompt=False):
        # now only target_color target_depth validate_color is used
        colors,depths,handmasks=[],[],[]
        validate_colors=[]
        
        for cam_index in range(len(self.target_view_color_path)):
            if(not os.path.exists(self.target_view_color_path[cam_index][cursor])):
                raise Exception(" when reading color, file %s is not found"%(self.target_view_color_path[cam_index][cursor]))
            if(not os.path.exists(self.target_view_depth_path[cam_index][cursor])):
                raise Exception(" when reading depth, file %s is not found"%(self.target_view_depth_path[cam_index][cursor]))
            if(prompt):
                print("Reading target view color of cursor %d from "%(cursor),self.target_view_color_path[cam_index][cursor])
                print("Reading target view depth of cursor %d from "%(cursor),self.target_view_depth_path[cam_index][cursor])
            color=cv2.imread(self.target_view_color_path[cam_index][cursor], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH |cv2.IMREAD_UNCHANGED)
            depth=cv2.imread(self.target_view_depth_path[cam_index][cursor], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH |cv2.IMREAD_UNCHANGED)
            colors.append(color),depths.append(depth)
            if(read_handmask):
                if(not os.path.exists(self.target_view_handmask_path[cam_index][cursor])):
                    raise Exception(" when reading depth, file %s is not found"%(self.target_view_handmask_path[cam_index][cursor]))
            if(prompt):
                print("Reading target view hand_mask of cursor %d from "%(cursor),self.target_view_handmask_path[cam_index][cursor])
                handmask=cv2.imread(self.target_view_handmask_path[cam_index][cursor], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH |cv2.IMREAD_UNCHANGED)
                handmasks.append(handmask)
                
        for cam_index in range(len(self.validate_view_color_path)):
            if(not os.path.exists(self.validate_view_color_path[cam_index][cursor])):
                raise Exception(" when reading color, file %s is not found"%(self.validate_view_color_path[cam_index][cursor]))
            print("Reading validate view color of cursor %d from "%(cursor),self.validate_view_color_path[cam_index][cursor])
            color=cv2.imread(self.validate_view_color_path[cam_index][cursor], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH |cv2.IMREAD_UNCHANGED)
            validate_colors.append(color)
        if not read_handmask:
            return colors,depths,validate_colors
        else:
            return colors,depths,handmasks,validate_colors
    #Helper function
    
    def get_frame_ids(self,path):
        files=os.listdir(path)
        frame_id=[]
        for file in files:
           if(file[-9:-4]=='color'):
               frame_id.append(int(file[0:-10]))
        frame_id.sort()
        return frame_id       
    
    def get_image_type(self,clip_dir):
        files=os.listdir(clip_dir)
        ans=[None,None]
        for file in files:
            if(file[-9:-4]=='color' and ans[0] is None):
                ans[0]=file[-3:]
            if(file[-9:-4]=='depth' and ans[1] is None):
                ans[1]=file[-3:]
            if(ans[0] is not None and ans[1] is not None):
                break
        return ans
    
    def check_frame_compatity(self,frame_ids_different_camera):
        for i in range(len(frame_ids_different_camera)-1):
            for frame1,frame2 in zip(frame_ids_different_camera[i],frame_ids_different_camera[i+1]):
                if(not frame1 == frame2):
                    raise Exception("the frame %d,%d from camera %d,%d is not equal "%(frame1,frame2,i,i+1))    