# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 16:14:56 2022

@author: v-zhiqili
"""
import VideoGenerate,HandMaskSegment
import os
import sys 
from argparse import ArgumentParser
import setting
from multiprocessing import Process



def TranverseGenerateMask(dataset,prompt,segment_mode=HandMaskSegment.SegmentMode.ColorBasedSegment):    
    print("In Process:",os.getpid())
    print("dataset:",[dataset_xml.path for dataset_xml in dataset])
    
    for dataset_xml in dataset:
        print("Dataset:",dataset_xml.path)
        print("    mask_camera:",dataset_xml.mask_camera)
        print("    validate_camera:",dataset_xml.validate_camera)
        print("    clips:",dataset_xml.clip_list)    
        HandMaskSegment.HandMaskSegment(dataset_path=dataset_xml.path,
                                            clip_ids=dataset_xml.clip_list,
                                            target_camera=dataset_xml.mask_camera,
                                            validate_camera=dataset_xml.validate_camera,
                                            prompt=prompt,mode=segment_mode)


def TranverseGenerateVideo(dataset,fps,prompt): 
    print("In Process:",os.getpid())
    print("dataset:",[dataset_xml.path for dataset_xml in dataset])
    print("fps:",fps)
    for dataset_xml in dataset:
        print("video_path:",dataset_xml.video_path)
        print("mask_camera:",dataset_xml.mask_camera)
        print("validate_camera:",dataset_xml.validate_camera)
        print("clips:",dataset_xml.clip_list)
        VideoGenerate.VideoGenerate(dataset_path=dataset_xml.path,
                                    video_root=dataset_xml.video_path,
                                    clip_ids=dataset_xml.clip_list,
                                    target_camera=dataset_xml.mask_camera,
                                    validate_camera=dataset_xml.validate_camera,
                                    fps=fps,
                                    prompt=prompt)
        

if __name__ == "__main__":
    maskArgs=setting.MaskArgs()
    maskArgs.args_generate()
    mod= len(maskArgs.setting_XML) % maskArgs.thread
    div= int(len(maskArgs.setting_XML) / maskArgs.thread)
    dataset_per_thread=[div+1]*mod+[div]*(maskArgs.thread-mod)
    dataset_assignment=[]
    begin_slice,end_slice=0,0
    for dataset_slice in dataset_per_thread:
        end_slice+=dataset_slice
        dataset_assignment.append(maskArgs.setting_XML[begin_slice:end_slice])
        begin_slice+=dataset_slice
    process_list=[]
    if(maskArgs.mode=="generate"):
        for dataset in dataset_assignment:
            p=Process(target=TranverseGenerateMask,args=(dataset,
                                                         maskArgs.prompt,
                                                         maskArgs.segment_mode))
            process_list.append(p)    
            p.start()
        
    elif(maskArgs.mode=="validate"):
        for dataset in dataset_assignment:
            p=Process(target=TranverseGenerateVideo,args=(dataset,
                                                          maskArgs.fps,
                                                          maskArgs.prompt))
            process_list.append(p)    
            p.start()
            
    for p in process_list:
        p.join()    