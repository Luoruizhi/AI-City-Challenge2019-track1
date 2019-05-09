## -----------------------------------------------------------------------------------
## 
##   Copyright (c) 2018 Alex Xiao.  All rights reserved.
## 
##   Description:
##       Implementation of calculating apperance features.
## 
## -----------------------------------------------------------------------------------

import numpy as np
import os
import sys
import glob
# import caffe
import cv2
from matplotlib import pyplot as plt
# from skimage.feature import hog
from operator import itemgetter



def get_frame_id(bbox_list_path,feature_list_path):
    id_list=[]
    bbox_list=[]
    appearance_list=[]
    with open(feature_list_path,'r') as f:
        data=f.readlines()
        for i in data:
            i=i.split()
            camera_id=i[0]
            track_id=i[1]
            frame_id=i[2]
            id_list.append([camera_id,track_id,frame_id])
    
    with open(bbox_list_path,'r') as f:
        data=f.readlines()
        for i in data:
            i=i.split()
            bbox_list.append(i)
            
    for id_item in id_list:
        for i in range(0,len(bbox_list)):
            if id_item[1]==bbox_list[i][1]:
                if id_item[2]==bbox_list[i][2]:
                    appearance_list.append(bbox_list[i])
                    break
    return appearance_list
        
def write_txt(path,feature_list):
    with open(path,'a') as f:
        for item in feature_list:
            for data in item:
                f.write(data)
                f.write('\t')
            f.write('\n')

def get_frame_from_video(video_name, appearance_list_item, img_dir):  ##视频切帧，切bbox
    """
    get a specific frame of a video by time in milliseconds
    :param video_name: video name
    :param frame_time: time of the desired frame
    :param img_dir: path which use to store output image
    :param img_name: name of output image
    :return: None
    """
    frame_id=int(appearance_list_item[2])
    obj_id=str(appearance_list_item[1])
    vidcap = cv2.VideoCapture(video_name)
    vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_id-1)
#     print(frame_id,obj_id,video_name)
    x=int(appearance_list_item[3])
    y=int(appearance_list_item[4])
    w=int(appearance_list_item[5])
    h=int(appearance_list_item[6])
#     print(frame_id_list_item[0],frame_id,obj_id,x1,y1,w,h)
    success, img = vidcap.read()
    height=img.shape[0]
    width=img.shape[1]
    if x+w>width:
        x=width-1-x
    if y+h>height:
        h=height-1-y
#     print(image)
    img_name=obj_id+'.jpg'

    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    if success:
        # save frame as JPEG file
#         image=cv2.rectangle(image,(x1,y1),(x1+w,y1+h),(0,255,0))
        img=img[y:y+h,x:x+w]
        cv2.imwrite(img_dir + img_name, img)  
        
        
        # cv2.imshow("frame%s" % frame_time, image)
        # cv2.waitKey()
    return img

def get_color_feature(bbox_list_path,feature_list_path,appearance_path,
                      video_path,img_path): ##得到feature
    appearance_list=get_frame_id(bbox_list_path,feature_list_path)
#     print(appearance_list)
#     appearance_list=sorted(appearance_list,key=itemgetter(0))
#     for frame_id_list_item in frame_id_list:
#         image=get_frame_from_video(video_path,frame_id_list_item,
#                            img_path)
    
    camera_name=video_path.split('/')[-2][1:]
#     print(appearance_list)
    with open(appearance_path, 'a') as f:
        for appearance_list_item in appearance_list:
            if appearance_list_item[0]==camera_name:
                image=get_frame_from_video(video_path,appearance_list_item,
                               img_path)

                obj_id=appearance_list_item[1]
                f.write(obj_id)
                print(obj_id,camera_name)
                f.write('\t')

                mask = np.zeros(image.shape[:2], np.uint8)
                mask = cv2.ellipse(mask, (int(image.shape[1] / 2),int(image.shape[0] / 2)), 
                                       (int(image.shape[1] / 2),int(image.shape[0] / 2)), 0, 0, 360, 255, -1)
                          # masked_img = cv2.bitwise_and(image,image, mask=mask)
                          # plt.imshow(masked_img, 'gray')
                          # plt.show()

                hist1 = cv2.calcHist([image], [0], mask, [16], [0, 256]).reshape(1, -1)
                hist2 = cv2.calcHist([image], [1], mask, [16], [0, 256]).reshape(1, -1)
                hist3 = cv2.calcHist([image], [2], mask, [16], [0, 256]).reshape(1, -1)
                          # cv2.normalize(hist1, hist1)
                          # cv2.normalize(hist2, hist2)
                          # cv2.normalize(hist3, hist3)
                rgb_feat = np.concatenate((hist1, hist2, hist3), axis=1)
                cv2.normalize(rgb_feat, rgb_feat)

                img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                hist1 = cv2.calcHist([img_hsv], [0], mask, [16], [0, 256]).reshape(1, -1)
                hist2 = cv2.calcHist([img_hsv], [1], mask, [16], [0, 256]).reshape(1, -1)
                hist3 = cv2.calcHist([img_hsv], [2], mask, [8], [0, 256]).reshape(1, -1)
                          # cv2.normalize(hist1, hist1)
                          # cv2.normalize(hist2, hist2)
                          # cv2.normalize(hist3, hist3)
                hsv_feat = np.concatenate((hist1, hist2), axis=1)
                cv2.normalize(hsv_feat, hsv_feat)

                img_YCrCb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
                hist1 = cv2.calcHist([img_YCrCb], [0], mask, [8], [0, 256]).reshape(1, -1)
                hist2 = cv2.calcHist([img_YCrCb], [1], mask, [16], [0, 256]).reshape(1, -1)
                hist3 = cv2.calcHist([img_YCrCb], [2], mask, [16], [0, 256]).reshape(1, -1)
                          # cv2.normalize(hist1, hist1)
                          # cv2.normalize(hist2, hist2)
                          # cv2.normalize(hist3, hist3)
                YCrCb_feat = np.concatenate((hist2, hist3), axis=1)
                cv2.normalize(YCrCb_feat, YCrCb_feat)

                img_lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
                hist1 = cv2.calcHist([img_lab], [0], mask, [8], [0, 256]).reshape(1, -1)
                hist2 = cv2.calcHist([img_lab], [1], mask, [16], [0, 256]).reshape(1, -1)
                hist3 = cv2.calcHist([img_lab], [2], mask, [16], [0, 256]).reshape(1, -1)
                          # cv2.normalize(hist1, hist1)
                          # cv2.normalize(hist2, hist2)
                          # cv2.normalize(hist3, hist3)
                lab_feat = np.concatenate((hist2, hist3), axis=1)
                cv2.normalize(lab_feat, lab_feat)

                          # image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                          # image_gray = cv2.resize(image_gray, (200,200))
                          # hog_feat = hog(image_gray, orientations=8, pixels_per_cell=(50,50), cells_per_block=(1,1), 
    #                     visualise=False).reshape(1, -1)
                          # cv2.normalize(hog_feat, hog_feat)

        #             type_feat = np.zeros(8).reshape(1,8) + 0.5
        #             print(type_feat)
        #             type_feat[0, 2] = 1
        #             print(type_feat)
        #             cv2.normalize(type_feat, type_feat)
    #             print(rgb_feat)

                feat = np.concatenate((3 * rgb_feat, hsv_feat, YCrCb_feat, lab_feat), axis=1)
                np.savetxt(f, feat, fmt='%.8g')
    print('write over')
            
    

        
if __name__=="__main__":
    
#     bbox_list_path="/home/luoruizhi/isilon-home/S03_bbox.txt"
    
#     feature_list_path="/home/luoruizhi/isilon-home/S03_feature_MAX1.txt"
#     appearance_path="/home/luoruizhi/isilon-home/S03_appearance_MAX1.txt"
#     path="/home/luoruizhi/isilon-home/train/S03/"
#     files=os.listdir(path)
#     files=sorted(files)
#     for file in files:
#         video_path=path+file+'/vdo.avi'
#         img_path=path+file+'/image'
#         get_color_feature(bbox_list_path,feature_list_path,appearance_path,video_path,img_path)
#         del video_path
#         del img_path
        
#     feature_list_path="/home/luoruizhi/isilon-home/S03_feature_MAX2.txt"
#     appearance_path="/home/luoruizhi/isilon-home/S03_appearance_MAX2.txt"
#     path="/home/luoruizhi/isilon-home/train/S03/"
#     files=os.listdir(path)
#     files=sorted(files)
#     for file in files:
#         video_path=path+file+'/vdo.avi'
#         img_path=path+file+'/image'
#         get_color_feature(bbox_list_path,feature_list_path,appearance_path,video_path,img_path)
#         del video_path
#         del img_path
        
#     feature_list_path="/home/luoruizhi/isilon-home/S03_feature_MAX3.txt"
#     appearance_path="/home/luoruizhi/isilon-home/S03_appearance_MAX3.txt"
#     path="/home/luoruizhi/isilon-home/train/S03/"
#     files=os.listdir(path)
#     files=sorted(files)
#     for file in files:
#         video_path=path+file+'/vdo.avi'
#         img_path=path+file+'/image'
#         get_color_feature(bbox_list_path,feature_list_path,appearance_path,video_path,img_path)
#         del video_path
#         del img_path
        
        
    bbox_list_path="/home/luoruizhi/isilon-home/S02_bbox.txt"
    
    feature_list_path="/home/luoruizhi/isilon-home/S02_feature_MAX1.txt"
    appearance_path="/home/luoruizhi/isilon-home/S02_appearance_MAX1.txt"
    path="/home/luoruizhi/isilon-home/test/S02/"
    files=os.listdir(path)
    files=sorted(files)
    
    for file in files:
        video_path=path+file+'/vdo.avi'
        img_path=path+file+'/image'
        get_color_feature(bbox_list_path,feature_list_path,appearance_path,video_path,img_path)
        del video_path
        del img_path
        
    feature_list_path="/home/luoruizhi/isilon-home/S02_feature_MAX2.txt"
    appearance_path="/home/luoruizhi/isilon-home/S02_appearance_MAX2.txt"
    path="/home/luoruizhi/isilon-home/test/S02/"
    files=os.listdir(path)
    files=sorted(files)
    
    for file in files:
        video_path=path+file+'/vdo.avi'
        img_path=path+file+'/image'
        get_color_feature(bbox_list_path,feature_list_path,appearance_path,video_path,img_path)
        del video_path
        del img_path
        
    feature_list_path="/home/luoruizhi/isilon-home/S02_feature_MAX3.txt"
    appearance_path="/home/luoruizhi/isilon-home/S02_appearance_MAX3.txt"
    path="/home/luoruizhi/isilon-home/test/S02/"
    files=os.listdir(path)
    files=sorted(files)
    
    for file in files:
        video_path=path+file+'/vdo.avi'
        img_path=path+file+'/image'
        get_color_feature(bbox_list_path,feature_list_path,appearance_path,video_path,img_path)
        del video_path
        del img_path
        
        
        
    feature_list_path="/home/luoruizhi/isilon-home/S02_feature_MAX4.txt"
    appearance_path="/home/luoruizhi/isilon-home/S02_appearance_MAX4.txt"
    path="/home/luoruizhi/isilon-home/test/S02/"
    files=os.listdir(path)
    files=sorted(files)
    
    for file in files:
        video_path=path+file+'/vdo.avi'
        img_path=path+file+'/image'
        get_color_feature(bbox_list_path,feature_list_path,appearance_path,video_path,img_path)
        del video_path
        del img_path
        

    feature_list_path="/home/luoruizhi/isilon-home/S02_feature_MAX5.txt"
    appearance_path="/home/luoruizhi/isilon-home/S02_appearance_MAX5.txt"
    path="/home/luoruizhi/isilon-home/test/S02/"
    files=os.listdir(path)
    files=sorted(files)
    
    for file in files:
        video_path=path+file+'/vdo.avi'
        img_path=path+file+'/image'
        get_color_feature(bbox_list_path,feature_list_path,appearance_path,video_path,img_path)
        del video_path
        del img_path  
        
        
        
        
    bbox_list_path="/home/luoruizhi/isilon-home/S05_bbox.txt"
    
    feature_list_path="/home/luoruizhi/isilon-home/S05_feature_MAX1.txt"
    appearance_path="/home/luoruizhi/isilon-home/S05_appearance_MAX1.txt"
    path="/home/luoruizhi/isilon-home/test/S05/"
    files=os.listdir(path)
    files=sorted(files)

    for file in files:
        video_path=path+file+'/vdo.avi'
        img_path=path+file+'/image'
        get_color_feature(bbox_list_path,feature_list_path,appearance_path,video_path,img_path)
        del video_path
        del img_path
        
    feature_list_path="/home/luoruizhi/isilon-home/S05_feature_MAX2.txt"
    appearance_path="/home/luoruizhi/isilon-home/S05_appearance_MAX2.txt"
    path="/home/luoruizhi/isilon-home/test/S05/"
    files=os.listdir(path)
    files=sorted(files)

    for file in files:
        video_path=path+file+'/vdo.avi'
        img_path=path+file+'/image'
        get_color_feature(bbox_list_path,feature_list_path,appearance_path,video_path,img_path)
        del video_path
        del img_path
        
    feature_list_path="/home/luoruizhi/isilon-home/S05_feature_MAX3.txt"
    appearance_path="/home/luoruizhi/isilon-home/S05_appearance_MAX3.txt"
    path="/home/luoruizhi/isilon-home/test/S05/"
    files=os.listdir(path)
    files=sorted(files)

    for file in files:
        video_path=path+file+'/vdo.avi'
        img_path=path+file+'/image'
        get_color_feature(bbox_list_path,feature_list_path,appearance_path,video_path,img_path)
        del video_path
        del img_path        
        
    feature_list_path="/home/luoruizhi/isilon-home/S05_feature_MAX4.txt"
    appearance_path="/home/luoruizhi/isilon-home/S05_appearance_MAX4.txt"
    path="/home/luoruizhi/isilon-home/test/S05/"
    files=os.listdir(path)
    files=sorted(files)
    
    for file in files:
        video_path=path+file+'/vdo.avi'
        img_path=path+file+'/image'
        get_color_feature(bbox_list_path,feature_list_path,appearance_path,video_path,img_path)
        del video_path
        del img_path
        

    feature_list_path="/home/luoruizhi/isilon-home/S05_feature_MAX5.txt"
    appearance_path="/home/luoruizhi/isilon-home/S05_appearance_MAX5.txt"
    path="/home/luoruizhi/isilon-home/test/S05/"
    files=os.listdir(path)
    files=sorted(files)
    
    for file in files:
        video_path=path+file+'/vdo.avi'
        img_path=path+file+'/image'
        get_color_feature(bbox_list_path,feature_list_path,appearance_path,video_path,img_path)
        del video_path
        del img_path        
       
