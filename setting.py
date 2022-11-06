# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 15:20:27 2022

@author: v-zhiqili

read XML

"""
from Common import *
import os
from argparse import ArgumentParser
import HandMaskSegment


def readSettingFromXml(xml_filename):
    '''
    Parse the xml file to get a list of scans
    :param xml_filename: the xml file to get scans
    :return: a list of scans
    '''
    try:
        parser = xml.etree.ElementTree.parse(xml_filename)
        node_root = parser.getroot()
        node_datasets = node_root.findall('Dataset')
    except:
        return []

    datasets = []
    for node_dataset in node_datasets:
        dataset = MaskSetting('')
        dataset.parse_xml_node(node_dataset)
        datasets.append(dataset)
    return datasets


def writeSettingToXml(scans, xml_filename):
    '''
    Write the list of scans into a xml file. In this way, we can edit each individual parameter of the scan.
    :param scans: list of scans
    :param xml_filename: absolute path of the output xml file
    :return: whether write xml succeed
    '''
    root = Element('Datasets')

    for scan in scans:
        root.append(scan.create_xml_node('Dataset'))

    with open(xml_filename, "w") as f:
        xml_ugly_str = xml.etree.ElementTree.tostring(root, 'utf-8')
        xml_parser = xml.dom.minidom.parseString(xml_ugly_str)
        xml_pretty_str = xml_parser.toprettyxml()
        f.write(xml_pretty_str)
        return True

    return False

class MaskSetting(ClassBase):
    def __init__(self, scan_path):
        super().__init__()
        # camera setting
        self.mask_camera=[]
        self.validate_camera=[]
        # clip setting
        self.clip_list=[]
        # dataset setting
        self.dataset=[]
        # path
        self.path=""
        self.video_path=""
        
        
class MaskArgs:
    def __init__(self):
        self.arg_parse=ArgumentParser()
        
        # items that occur in setting file
        self.arg_parse.add_argument("--path", help="the root for the path",dest="path",default=None)
        self.arg_parse.add_argument("--mask_camera", help="the list for mask_camera",dest="mask_camera",default="0,6,8")
        self.arg_parse.add_argument("--validate_camera", help="the list for validate_camera",dest="validate_camera",default="1,5")
        self.arg_parse.add_argument("--clips", help="the clip to begin",dest="clips",default="7,8") 
        # video path have 2 options, occur in setting file or not
        self.arg_parse.add_argument("--video_path", help="the path to generate the video",dest="video_path",default=None)        

        # generate the setting or not
        self.arg_parse.add_argument("--setting", help="the path for the setting file",dest="setting",default=None)
        self.arg_parse.add_argument("--setting_format",help="generate the example file for the setting file",dest="setting_format",default=None)
        self.arg_parse.add_argument("--use_command",help="use the command not the xml",dest="use_command",default="False")
        self.arg_parse.add_argument("--override",help="override the xml by command",dest="override",default="False")
        self.arg_parse.add_argument("--dataset", help="dataset that will be used",dest="dataset",default=None)

        # items that not occur in setting file
        self.arg_parse.add_argument("--thread", help="set the how many process to use",dest="thread",default=1)
        self.arg_parse.add_argument("--fps", help="fps of the generated video",dest="fps",default=15)
        self.arg_parse.add_argument("--mode", help="validate/generate",dest="mode",default="generate")
        self.arg_parse.add_argument("--prompt",help="whether to output the prompt information",dest="prompt",default="True")        
        self.arg_parse.add_argument("--segment_mode",help="choose the segment mode",dest="segment_mode",default="color")        
        
        #self.arg_parse.add_argument("--camera_list", help="the list for all camera",dest="camera_list",default=None)
        # camera setting
        #self.camera_list=None
        self.mask_camera=None
        self.validate_camera=None
        # clip setting
        self.clip_list=None
        # dataset setting
        self.datasets=None    
        # path
        self.mode="generate"
        self.path=None
        self.video_path=None
        self.setting_path=None
        self.setting_XML=None
        self.prompt=None
        self.segment_mode=None
        self.thread=1
        self.fps=15
    def check_args1(self):
        if(self.mode not in ["generate","validate"]):
            raise Exception(' invalidate mode, which should be "generate" and "validate" ')
        if(self.setting_path is None):
            if(self.path is None or len(self.path)==0):
                raise Exception(" the root path for dataset is empty")
            if((self.video_path is None or len(self.video_path)==0)and self.mode=="validate"):
                raise Exception(" the root path for validation video is empty")
    def check_args2(self):
        # then validate if the dataset exist
        for dataset in self.setting_XML:
            dataset_path=dataset.path
            if not os.path.exists(dataset_path):
                raise Exception(" Path for dataset %s doesnot exist!"%(dataset_path))
        # chech the compatibility of the args for camera
        #if(not (max(self.mask_camera+self.validate_camera)+1<=len(self.camera_list))):
        #    raise Exception(" the args for camera is not compatibility")
        
        
    def parse_list(self,list_str):
        if(list_str is None):
            return None
        pre_list,ans=list_str.split(","),[]
        for item in pre_list:
            ans.append(int(item))
        return ans
    
    
    def output_prompt(self):
        print("-----------------------------------------------------------------------------------")
        print("-----------------------------------------------------------------------------------")
        print("-----------------------------Information for this batch----------------------------")
        print("-----------------------------------------------------------------------------------")
        print("-----------------------------------------------------------------------------------")
        print("MODE:",self.mode)
        print("thread",self.thread)
        print("setting path:",self.setting_path)
        if(self.override):
            print("Warning: the setting file will be overrided by command line")
        print("The following datasets will be handled, total:%d"%(len(self.setting_XML)))
        for dataset_xml in self.setting_XML:
            print("    Datset:",dataset_xml.path)
            print("        mask_camera:",self.mask_camera)
            print("        validate_camera:",self.validate_camera)
            print("        The following clips will be handled, total:%d"%(len(self.clip_list)))
            print("        ",self.clip_list)
            print("        video_path:",self.video_path)
        
        will_coninue=input("continue or not(y/n):")
        if(not(will_coninue=="y" or will_coninue=="Y")):
            print("cancel by users")
            exit(1)
            
            
    def check_path_one(self,path):
        # this function is to check if the path is for only one dataset
        try:
            files=os.listdir(path)
        except:
            raise Exception("Read the data root meet error, maybe it not exist")
        for file in files:
            if(file[-4:]=="rgbd"):
                return True
        return False
    
    def fetch_dataset(self,path):
        if(not self.check_path_one(path)):
            return [file for file in os.listdir(path)],path
        else:
            return [os.path.split(path)[1]],os.path.split(path)[0]
    def parse_bool(self,var):
        if(var=="True" or var=="true" or var=="T" or var=="t" or var=="Yes" or var=="yes" or var=="Y" or var=="y"):
            return True
        elif(var=="False" or var=="false" or var=="F" or var=="f" or var=="No" or var=="no" or var=="N" or var=="n"):
            return False
        return None
    def args_generate(self):
        # the rule here is:
        #   First parse the args and only query the xml if the result is empty
        # Step 1: parse the general command line
        args=self.arg_parse.parse_args()
        
        self.fps=int(args.fps)
        self.mode=str(args.mode)
        self.thread=int(args.thread)
        self.segment_mode=HandMaskSegment.SegmentMode.parseName(args.segment_mode)
        
        # Step 2: if generate the xml file
        if(args.setting_format is not None):
            self.path,self.video_path=args.path,args.video_path
            self.mask_camera=self.parse_list(args.mask_camera)
            self.validate_camera=self.parse_list(args.validate_camera)
            self.clip_list=self.parse_list(args.clips)
            self.datasets,self.path=self.fetch_dataset(self.path)
            if args.dataset is not None:
                self.datasets=self.parse_list(args.dataset)       
            generateExample(str(args.setting_format),self.datasets,self.validate_camera,self.mask_camera,self.clip_list,self.path,self.video_path)

        # Step 3: if setting exist or not 
        self.setting_path=args.setting        
        if(self.setting_path is not None):
            try:
                self.setting_XML=readSettingFromXml(args.setting)
            except:
                raise Exception("The xml file format is wrong")
        
        # Step 4: then override or used by command line
        self.path,self.video_path=args.path,args.video_path
        self.mask_camera=self.parse_list(args.mask_camera)
        self.validate_camera=self.parse_list(args.validate_camera)
        self.clip_list=self.parse_list(args.clips)
        self.datasets,self.path=self.fetch_dataset(self.path)
        self.check_args1()
        if args.dataset is not None:
            self.datasets=self.parse_list(args.dataset)
        if self.setting_XML is None:
            self.setting_XML=[]
            for dataset in self.datasets:
                dataset_xml=MaskSetting("")
                dataset_xml.mask_camera=self.mask_camera
                dataset_xml.validate_camera=self.validate_camera
                dataset_xml.clip_list=self.clip_list
                dataset_xml.path=os.path.join(self.path, str(dataset))
                dataset_xml.video_path=self.video_path
                self.setting_XML.append(dataset_xml)

        # then override
        self.override=self.parse_bool(args.override)
        if(self.parse_bool(args.use_command) or self.parse_bool(args.override)):
            if(self.setting_path is None):
                print("Warning: use_command or override is not used, for no setting file")
            else:
                for dataset_xml in self.setting_XML:
                    dataset_xml.mask_camera=self.mask_camera
                    dataset_xml.validate_camera=self.validate_camera
                    dataset_xml.clip_list=self.clip_list
                    dataset_xml.video_path=self.video_path
            if(args.override):
                if(self.setting_path is None):
                    raise Exception("the file to override have no been input")
                
                
        # Step 5:check and output
        self.check_args2()
        self.prompt=False
        if(self.parse_bool(args.prompt)):
            self.output_prompt()
            self.prompt=True
            
        # Step 6: Extra
        if(self.override):
            writeSettingToXml(self.setting_XML, self.setting_path)
            
def generateExample(fileName,datasets,validate_camera,mask_camera,clip_list,data_path,video_path):
    mask_settings=[]
    for dataset in datasets:
        mask_setting=MaskSetting("")
        mask_setting.mask_camera=mask_camera
        mask_setting.validate_camera=validate_camera
        mask_setting.clip_list=clip_list
        mask_setting.path=os.path.join(data_path, str(dataset))
        mask_setting.video_path=video_path
        mask_settings.append(mask_setting)
    writeSettingToXml(mask_settings, fileName)
    exit(1)
if __name__ == "__main__":
    maskArgs=MaskArgs()
    maskArgs.args_generate()