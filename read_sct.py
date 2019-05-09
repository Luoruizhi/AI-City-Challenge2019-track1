import numpy as np
import json
import base64
import cv2
import io
from operator import itemgetter
import os

def select_feature(json_path,index,txt_path):
    with open(json_path,'r') as f:
        track_info=json.load(f)
   
    bbox_list=[]
    feature_MAX1_list=[]
    feature_MAX2_list=[]
    feature_MAX3_list=[]
    feature_MAX4_list=[]
    feature_MAX5_list=[]
    
    
    camera_id=json_path.split('/')[-1].split('.')[0].split('_')[-1][1:]
    for track_id in track_info.keys():
        static = track_info[track_id]['static']
        if static==True:
            continue
        frame_info_list = track_info[track_id]['frame_info']
        MAX1_frame_id=-1
        MAX1_area=-1
        MAX1_feature=[]
        
        MAX2_frame_id=-1
        MAX2_area=-1
        MAX2_feature=[]
        
        MAX3_frame_id=-1
        MAX3_area=-1
        MAX3_feature=[]

        MAX4_frame_id=-1
        MAX4_area=-1
        MAX4_feature=[]
        
        MAX5_frame_id=-1
        MAX5_area=-1
        MAX5_feature=[] 
        
        
        frame_num=0
        
        for frame_info in frame_info_list:
            frame_id = frame_info['frame_id']
            bbox = frame_info['bbox']
            feature = frame_info['feature']
            bbox[0]=int(bbox[0])
            bbox[1]=int(bbox[1])
            bbox[2]=int(bbox[2])-bbox[0]
            bbox[3]=int(bbox[3])-bbox[1]
            if bbox[2]*bbox[3]<10000:
                continue
            
            if MAX1_area<bbox[2]*bbox[3]:
                MAX5_area=MAX4_area
                MAX5_feature=MAX4_feature
                MAX5_frame_id=MAX4_frame_id
                MAX4_area=MAX3_area
                MAX4_feature=MAX3_feature
                MAX4_frame_id=MAX3_frame_id
                MAX3_area=MAX2_area
                MAX3_feature=MAX2_feature
                MAX3_frame_id=MAX2_frame_id
                MAX2_area=MAX1_area
                MAX2_feature=MAX1_feature
                MAX2_frame_id=MAX1_frame_id
                MAX1_area=bbox[2]*bbox[3]
                MAX1_frame_id=frame_id
                MAX1_feature=feature
#                 print(bbox[2]*bbox[3])
            
            elif MAX2_area<bbox[2]*bbox[3]:
                MAX5_area=MAX4_area
                MAX5_feature=MAX4_feature
                MAX5_frame_id=MAX4_frame_id
                MAX4_area=MAX3_area
                MAX4_feature=MAX3_feature
                MAX4_frame_id=MAX3_frame_id
                MAX3_area=MAX2_area
                MAX3_feature=MAX2_feature
                MAX3_frame_id=MAX2_frame_id
                MAX2_area=bbox[2]*bbox[3]
                MAX2_frame_id=frame_id
                MAX2_feature=feature
#                 print(bbox[2]*bbox[3])
                
                
            elif  MAX3_area<bbox[2]*bbox[3]:
                MAX5_area=MAX4_area
                MAX5_feature=MAX4_feature
                MAX5_frame_id=MAX4_frame_id
                MAX4_area=MAX3_area
                MAX4_feature=MAX3_feature
                MAX4_frame_id=MAX3_frame_id
                MAX3_area=bbox[2]*bbox[3]
                MAX3_frame_id=frame_id
                MAX3_feature=feature
#                 print(bbox[2]*bbox[3])

            elif MAX4_area<bbox[2]*bbox[3]:
                MAX5_area=MAX4_area
                MAX5_feature=MAX4_feature
                MAX5_frame_id=MAX4_frame_id
                MAX4_area=bbox[2]*bbox[3]
                MAX4_frame_id=frame_id
                MAX4_feature=feature
            
            elif MAX5_area<bbox[2]*bbox[3]:
                MAX5_area=bbox[2]*bbox[3]
                MAX5_frame_id=frame_id
                MAX5_feature=feature 
    
            frame_num+=1

            bbox_list_member=[camera_id,index,frame_id]
            bbox_list_member+=bbox
            bbox_list.append(bbox_list_member)
            del bbox_list_member
            
            
       
        if frame_num>5:
            feature_MAX1_member=[camera_id,index,MAX1_frame_id,MAX1_area]
            feature_MAX1_member+=MAX1_feature

            feature_MAX2_member=[camera_id,index,MAX2_frame_id,MAX2_area]
            feature_MAX2_member+=MAX2_feature

            feature_MAX3_member=[camera_id,index,MAX3_frame_id,MAX3_area]
            feature_MAX3_member+=MAX3_feature

            feature_MAX4_member=[camera_id,index,MAX4_frame_id,MAX4_area]
            feature_MAX4_member+=MAX4_feature
            
            feature_MAX5_member=[camera_id,index,MAX5_frame_id,MAX5_area]
            feature_MAX5_member+=MAX5_feature
            
            feature_MAX1_list.append(feature_MAX1_member)
            feature_MAX2_list.append(feature_MAX2_member)
            feature_MAX3_list.append(feature_MAX3_member)
            feature_MAX4_list.append(feature_MAX4_member)
            feature_MAX5_list.append(feature_MAX5_member)

            print(MAX1_area,MAX3_area,MAX5_area,MAX1_frame_id,MAX3_frame_id,MAX5_frame_id,
                  index,frame_num)

            del MAX1_feature
            del MAX2_feature
            del MAX3_feature
            del MAX4_feature
            del MAX5_feature
            
            del feature_MAX1_member
            del feature_MAX2_member
            del feature_MAX3_member
            del feature_MAX4_member
            del feature_MAX5_member
            
            del MAX1_area
            del MAX2_area
            del MAX3_area
            del MAX4_area
            del MAX5_area
            
            index+=1
            
            
    bbox_list=sorted(bbox_list,key=itemgetter(0))
#     feature_list=sorted(feature_list,key=itemgetter(0))
#     print(bbox_list)
    bbox_path=txt_path+'bbox.txt'
    feature_MAX1_path=txt_path+'feature_MAX1.txt'
    feature_MAX2_path=txt_path+'feature_MAX2.txt'
    feature_MAX3_path=txt_path+'feature_MAX3.txt'
    feature_MAX4_path=txt_path+'feature_MAX4.txt'
    feature_MAX5_path=txt_path+'feature_MAX5.txt'
                
    
    
    write_txt(bbox_path,bbox_list)
    write_txt(feature_MAX1_path,feature_MAX1_list)
    write_txt(feature_MAX2_path,feature_MAX2_list)
    write_txt(feature_MAX3_path,feature_MAX3_list)
    write_txt(feature_MAX4_path,feature_MAX4_list)
    write_txt(feature_MAX5_path,feature_MAX5_list)
    
    
    return index

def write_txt(path,label_list):
    f=open(path,'a')
    for i in range(len(label_list)):
#         print (label_list[i])
        for j in range(len(label_list[i])):
            f.write(str(label_list[i][j]))    #write函数不能写int类型的参数，所以使用str()转化
            f.write('\t')   
        f.write('\n')
    f.close()

            

if __name__=='__main__':
    
    txt2_path='/home/luoruizhi/isilon-home/S02_'
    txt3_path='/home/luoruizhi/isilon-home/S03_'

    txt5_path='/home/luoruizhi/isilon-home/S05_'
    
#     index=1
#     path = "/home/luoruizhi/isilon-home/SCT-megreid-v2/" 
#     files= os.listdir(path) 
#     files=sorted(files)
#     print(files)
    
#     for file in files:
#         json_path=path+file
#         index=select_feature(json_path,index,txt3_path)
    
    index=1
    path = "/home/luoruizhi/isilon-home/SCT-test/S02/" 
    files= os.listdir(path) 
    files=sorted(files)
    print(files)
    for file in files:
        json_path=path+file
        index=select_feature(json_path,index,txt2_path)
        
    
    path = "/home/luoruizhi/isilon-home/SCT-test/S05/" 
    files= os.listdir(path) 
    files=sorted(files)
    print(files)
    for file in files:
        json_path=path+file
        index=select_feature(json_path,index,txt5_path)
        
        
        
        
