import numpy as np
import json
from operator import itemgetter
import sys
import argparse
import base64
import math
from meghair.utils.imgproc import imdecode
sys.path.append('/home/luoruizhi/vehicle-script/')
from  calculate_feature_distance import get_feature_distance_matrix

def read_txt(path):
    label_list=[]
    with open(path,'r') as f:
        data=f.readlines()
        for i in data:
            i=i.split()
#             print(i[1])
            label_list.append(i)
    return label_list


def write_txt(path,id_list):
    f=open(path,'w')
    for i in range(len(id_list)):
#         print(similar_id_list[i])
        for item in id_list[i]:
            f.write(str(item))
            f.write('\t')
        f.write('\n')
    f.close()
    
def write_dict(path,dic):
    f=open(path,'w')
    f.write(str(dic))
    f.close()           

def select_loss(path,threshold,id_length):
    
    appearance_score=cal_appearance(path)
    bbox_path=path+'bbox.txt'
    feature_MAX1_path=path+'feature_MAX1.txt'
    feature_MAX3_path=path+'feature_MAX3.txt'
    feature_MAX5_path=path+'feature_MAX5.txt'
    
    
    bbox_list=read_txt(bbox_path)
    feature_MAX1_list=read_txt(feature_MAX1_path)
    feature_MAX3_list=read_txt(feature_MAX3_path)
    feature_MAX5_list=read_txt(feature_MAX5_path)
#     print(feature_MAX1_list[-1])
    
#     for i in feature_MAX2_list:
#         print(len(i),i[0],i[1],i[2])
    
#     feature_list=sorted(feature_list,key=itemgetter(0))
#     frame_id_list=sorted(frame_id_list,key=itemgetter(0))

    temp_MAX1=[]
    for item2 in feature_MAX1_list:
#         b=np.frombuffer(base64.b64decode(item[2]))
        b=item2[4:]
        temp_MAX1.append(b)
    MAX1_feature_array=np.array(temp_MAX1)
    
    MAX1_feature_array=MAX1_feature_array.astype(float)
    
    temp_MAX3=[]
    for item1 in feature_MAX3_list:
#         b=np.frombuffer(base64.b64decode(item[2]))
        a=item1[4:]
        temp_MAX3.append(a)    
    MAX3_feature_array=np.array(temp_MAX3)
    MAX3_feature_array=MAX3_feature_array.astype(float)
    

        
    temp_MAX5=[]
    for item3 in feature_MAX5_list:
#         b=np.frombuffer(base64.b64decode(item[2]))
        c=item3[4:]
        temp_MAX5.append(c)
    MAX5_feature_array=np.array(temp_MAX5)
    MAX5_feature_array=MAX5_feature_array.astype(float)
    
      
    print(MAX1_feature_array.shape,MAX3_feature_array.shape,MAX5_feature_array.shape)
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--slice', default=3)
    args = parser.parse_args()
#     print(MAX1_feature_array.shape,MAX2_feature_array.shape,MAX3_feature_array.shape)
#     print(MAX2_feature_array[0])
    MSE1_array=get_feature_distance_matrix(MAX1_feature_array,MAX1_feature_array,args.slice)
    MSE2_array=get_feature_distance_matrix(MAX3_feature_array,MAX3_feature_array,args.slice)
    MSE3_array=get_feature_distance_matrix(MAX5_feature_array,MAX5_feature_array,args.slice)
    MSE4_array=get_feature_distance_matrix(MAX1_feature_array,MAX3_feature_array,args.slice)
    MSE5_array=get_feature_distance_matrix(MAX1_feature_array,MAX5_feature_array,args.slice)
    MSE6_array=get_feature_distance_matrix(MAX3_feature_array,MAX5_feature_array,args.slice)
    
    MSE_array=(MSE1_array+MSE2_array+MSE3_array+MSE4_array+MSE5_array+MSE6_array)/6
    
    
    
#     reid_score1=np.zeros((MAX1_feature_array.shape[0],MAX1_feature_array.shape[0]))
#     reid_score2=np.zeros((MAX1_feature_array.shape[0],MAX1_feature_array.shape[0]))
#     reid_score3=np.zeros((MAX1_feature_array.shape[0],MAX1_feature_array.shape[0]))
#     reid_score4=np.zeros((MAX1_feature_array.shape[0],MAX1_feature_array.shape[0]))
#     reid_score5=np.zeros((MAX1_feature_array.shape[0],MAX1_feature_array.shape[0]))
#     reid_score6=np.zeros((MAX1_feature_array.shape[0],MAX1_feature_array.shape[0]))
    
    
#     for i in range(0,MAX1_feature_array.shape[0]):
#         appearance_score_item=[]
#         score=0
#         for j in range(i+1,MAX1_feature_array.shape[0]):
#             score=bhattacharyya(MAX1_feature_array[i,:]/sum(MAX1_feature_array[i,:]), 
#                                  MAX1_feature_array[j,:]/sum(MAX1_feature_array[j,:]))
#             reid_score1[i][j]=score
#     print(reid_score1)
            
#     for i in range(MAX1_feature_array.shape[0]):
#         appearance_score_item=[]
#         score=0
#         for j in range(i+1,MAX2_feature_array.shape[0]):
#             score=bhattacharyya(MAX1_feature_array[i,:]/sum(MAX1_feature_array[i,:]),
#                                  MAX2_feature_array[j,:]/sum(MAX2_feature_array[j,:])) 
#             reid_score2[i][j]=score
#     print(reid_score2)
            
            
#     for i in range(MAX1_feature_array.shape[0]):
#         appearance_score_item=[]
#         score=0
#         for j in range(i+1,MAX3_feature_array.shape[0]):
#             score=bhattacharyya(MAX1_feature_array[i,:]/sum(MAX1_feature_array[i,:]), 
#                                  MAX3_feature_array[j,:]/sum(MAX3_feature_array[j,:])) 
#             reid_score3[i][j]=score
#     print(reid_score3)
            
            
            
#     for i in range(MAX2_feature_array.shape[0]):
#         appearance_score_item=[]
#         score=0
#         for j in range(i+1,MAX2_feature_array.shape[0]):
#             score=bhattacharyya(MAX2_feature_array[i,:]/sum(MAX2_feature_array[i,:]), 
#                                  MAX2_feature_array[j,:]/sum(MAX2_feature_array[j,:])) 
#             reid_score4[i][j]=score  
#     print(reid_score4)
            
            
            
#     for i in range(MAX2_feature_array.shape[0]):
#         appearance_score_item=[]
#         score=0
#         for j in range(i+1,MAX3_feature_array.shape[0]):
#             score=bhattacharyya(MAX2_feature_array[i,:]/sum(MAX2_feature_array[i,:]), 
#                                  MAX3_feature_array[j,:]/sum(MAX3_feature_array[j,:])) 
#             reid_score5[i][j]=score
#     print(reid_score5)
            
            
            
#     for i in range(MAX3_feature_array.shape[0]):
#         appearance_score_item=[]
#         score=0
#         for j in range(i+1,MAX3_feature_array.shape[0]):
#             score=bhattacharyya(MAX3_feature_array[i,:]/sum(MAX3_feature_array[i,:]), 
#                                  MAX3_feature_array[j,:]/sum(MAX3_feature_array[j,:])) 
#             reid_score6[i][j]=score    
#     print(reid_score6)
           
            
#     MSE_array=(reid_score1+reid_score2+reid_score3+  \
#         reid_score4+reid_score5+reid_score6)
    
    MSE_list=MSE_array.tolist()
    alpha=0.09807970309922766
    beta=-4.064333715378715
    for i in range(0,len(MSE_list)):
        for j in range(0,len(MSE_list[0])):
            i_area=int(feature_MAX1_list[i][3])
            j_area=int(feature_MAX1_list[j][3])
            appearance_score=appearance_score[i][j]
            if j_area/i_area>4 or i_area/j_area>4:
                MSE_list[i][j]=10000
            if i_area<80000 and j_area<80000:
                appearance_score=1
            else:
                MSE_list[i][j]*=appearance_score
            MSE_list[i][j]=100. / (1. + np.exp(alpha * MSE_list[i][j] + beta))

    MSE_array=np.array(MSE_list)
    all_similar_id={}
    similar_id={}
    result_path=path+'id_merge_result_v1.txt'
    for i in range(0,len(MSE_list)):
        sort_list=sorted(range(i+1,len(MSE_list[i])),key=lambda k:MSE_list[i][k],reverse=True)
#         print(len(MSE_list))
        for j in sort_list:
#             break
#             print(j,MSE_list[i][j])
            if MSE_list[i][j]>threshold and feature_MAX1_list[i][0]!=feature_MAX1_list[j][0]:
                all_similar_id[i+1+id_length]=j+1+id_length
                print(j,MSE_list[i][j])
                break 
    
    
    for key in all_similar_id.keys():
        temp_key=key
        temp_value=all_similar_id[key]
#         key_list=list(list(all_similar_id.keys())[list(all_similar_id.values()).index(temp_value)])
#         print(key_list)
        MAX=0
        MAX_key=0
        MAX_value=0
        for k,v in all_similar_id.items():
            if temp_value==v:
                if MSE_list[k-1-id_length][v-1-id_length]>MAX:
                    MAX=MSE_list[k-1-id_length][v-1-id_length]
                    MAX_key=k
                    MAX_value=v
        similar_id[MAX_key]=MAX_value
    print(similar_id)    
       
    merge_id_list=merge_id(MSE_array,similar_id,threshold,id_length)
    
#     write_dict(result_path,similar_id)
    write_txt(result_path,merge_id_list)

    id_length=len(feature_MAX1_list)
    return bbox_list,merge_id_list,id_length

def merge_id(MSE_array,similar_id,threshold,id_length):
    merge_id_list=[]
    for key in similar_id.keys():
        merge_id_sublist=[]
        temp_key=key
        while temp_key in similar_id.keys():
            merge_id_sublist.append(temp_key)
#             value=similar_id[temp_key]
#             MAX=MSE_array[temp_key][value]
#             list(similar_id.keys())[list(similar_id.values()).index(value)]
#             while value in similar_id.values():
                
            merge_id_sublist.append(similar_id[temp_key])
            temp_key=similar_id[temp_key]
        merge_id_sublist=list(set(merge_id_sublist))
        merge_id_sublist.sort()
        merge_id_list.append(merge_id_sublist)
        del merge_id_sublist
            
    return merge_id_list
                            
def write_bbox_result(path,threshold,ID,id_length):
    bbox_list,merge_id_list,id_length=select_loss(path,threshold,id_length)
#     bbox_list=sorted(bbox_list,key=itemgetter(0))
    accurated_bbox_list=[]
    bbox_result_path=path+'bbox_result_v1.txt'
    for i in range(0,len(merge_id_list)):
        index=i+ID
        for j in range(0,len(merge_id_list[i])):
            for item in bbox_list:
                if int(item[1])==merge_id_list[i][j]:
                    if path=='/home/luoruizhi/isilon-home/S02_':
                        camera_id=int(item[0][2:])
                    else:
                        camera_id=int(item[0][1:])
                    track=[camera_id,index]
                    track+=[item[2],item[3],item[4],item[5],item[6]]
                    accurated_bbox_list.append(track)
#                     print(track)
                    del track
 
    ID=len(merge_id_list)
    write_txt(bbox_result_path,accurated_bbox_list)   
    
    return ID,id_length          
        
def cal_appearance(path):
    appearance_MAX1_list_path=path+'appearance_MAX1.txt'
    appearance_MAX3_list_path=path+'appearance_MAX3.txt'
    appearance_MAX5_list_path=path+'appearance_MAX5.txt'
    
    appearance_MAX1_list=read_txt(appearance_MAX1_list_path)
    appearance_MAX3_list=read_txt(appearance_MAX3_list_path)
    appearance_MAX5_list=read_txt(appearance_MAX5_list_path)
    
#     temp1=np.array(appearance_MAX1_list).astype(float)
#     appearance_MAX1_list=temp1.tolist()
#     temp2=np.array(appearance_MAX2_list).astype(float)
#     appearance_MAX2_list=temp2.tolist()
#     temp3=np.array(appearance_MAX3_list).astype(float)
#     appearance_MAX3_list=temp3.tolist()
    
    
#     appearance_MAX1_list=sorted(appearance_MAX1_list,key=itemgetter(0))
#     appearance_MAX2_list=sorted(appearance_MAX2_list,key=itemgetter(0))
#     appearance_MAX3_list=sorted(appearance_MAX3_list,key=itemgetter(0))
    
#     print(appearance_MAX1_list)
    
    appearance_MAX1=[]
    appearance_MAX3=[]
    appearance_MAX5=[]
    
    
    for item in appearance_MAX1_list:
        appearance_MAX1.append(item[1:])
    
    for item in appearance_MAX3_list:
        appearance_MAX3.append(item[1:])
        
    for item in appearance_MAX5_list:
        appearance_MAX5.append(item[1:])        
        
    appearance_MAX1_array=np.array(appearance_MAX1)
    appearance_MAX3_array=np.array(appearance_MAX3)
    appearance_MAX5_array=np.array(appearance_MAX5)
    
    appearance_MAX1_array=appearance_MAX1_array.astype(float)
    appearance_MAX3_array=appearance_MAX3_array.astype(float)
    appearance_MAX5_array=appearance_MAX5_array.astype(float)
    
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--slice', default=3)
#     args = parser.parse_args()
# #     print(MAX1_feature_array.shape,MAX2_feature_array.shape,MAX3_feature_array.shape)
# #     print(MAX2_feature_array[0])
#     appearance_score1=get_feature_distance_matrix(appearance_MAX1_array,appearance_MAX1_array,args.slice)
#     appearance_score2=get_feature_distance_matrix(appearance_MAX1_array,appearance_MAX2_array,args.slice)
#     appearance_score3=get_feature_distance_matrix(appearance_MAX1_array,appearance_MAX3_array,args.slice)
#     appearance_score4=get_feature_distance_matrix(appearance_MAX2_array,appearance_MAX2_array,args.slice)
#     appearance_score5=get_feature_distance_matrix(appearance_MAX2_array,appearance_MAX3_array,args.slice)
#     appearance_score6=get_feature_distance_matrix(appearance_MAX3_array,appearance_MAX3_array,args.slice)
    
#     appearance_score=(appearance_score1+appearance_score2+
#                       appearance_score3+appearance_score4+
#                       appearance_score5+appearance_score6)/6
    
    appearance_score1=np.zeros((appearance_MAX1_array.shape[0],appearance_MAX1_array.shape[0]))
    appearance_score2=np.zeros((appearance_MAX1_array.shape[0],appearance_MAX1_array.shape[0]))
    appearance_score3=np.zeros((appearance_MAX1_array.shape[0],appearance_MAX1_array.shape[0]))
    appearance_score4=np.zeros((appearance_MAX1_array.shape[0],appearance_MAX1_array.shape[0]))
    appearance_score5=np.zeros((appearance_MAX1_array.shape[0],appearance_MAX1_array.shape[0]))
    appearance_score6=np.zeros((appearance_MAX1_array.shape[0],appearance_MAX1_array.shape[0]))
    
    print(len(appearance_MAX1_list),appearance_MAX1_array.shape)
    
    for i in range(0,appearance_MAX1_array.shape[0]):
        appearance_score_item=[]
        score=0
        for j in range(i+1,appearance_MAX1_array.shape[0]):
            score=bhattacharyya(appearance_MAX1_array[i,:]/sum(appearance_MAX1_array[i,:]), 
                                 appearance_MAX1_array[j,:]/sum(appearance_MAX1_array[j,:]))
            appearance_score1[i][j]=score
    print(appearance_score1)
            
    for i in range(appearance_MAX1_array.shape[0]):
        appearance_score_item=[]
        score=0
        for j in range(i+1,appearance_MAX3_array.shape[0]):
            score=bhattacharyya(appearance_MAX1_array[i,:]/sum(appearance_MAX1_array[i,:]),
                                 appearance_MAX3_array[j,:]/sum(appearance_MAX3_array[j,:])) 
            appearance_score2[i][j]=score
    print(appearance_score2)
            
            
    for i in range(appearance_MAX1_array.shape[0]):
        appearance_score_item=[]
        score=0
        for j in range(i+1,appearance_MAX5_array.shape[0]):
            score=bhattacharyya(appearance_MAX1_array[i,:]/sum(appearance_MAX1_array[i,:]), 
                                 appearance_MAX5_array[j,:]/sum(appearance_MAX5_array[j,:])) 
            appearance_score3[i][j]=score
    print(appearance_score3)
            
            
            
    for i in range(appearance_MAX3_array.shape[0]):
        appearance_score_item=[]
        score=0
        for j in range(i+1,appearance_MAX3_array.shape[0]):
            score=bhattacharyya(appearance_MAX3_array[i,:]/sum(appearance_MAX3_array[i,:]), 
                                 appearance_MAX3_array[j,:]/sum(appearance_MAX3_array[j,:])) 
            appearance_score4[i][j]=score  
    print(appearance_score4)
            
            
            
    for i in range(appearance_MAX3_array.shape[0]):
        appearance_score_item=[]
        score=0
        for j in range(i+1,appearance_MAX5_array.shape[0]):
            score=bhattacharyya(appearance_MAX3_array[i,:]/sum(appearance_MAX3_array[i,:]), 
                                 appearance_MAX5_array[j,:]/sum(appearance_MAX5_array[j,:])) 
            appearance_score5[i][j]=score
    print(appearance_score5)
            
            
            
    for i in range(appearance_MAX5_array.shape[0]):
        appearance_score_item=[]
        score=0
        for j in range(i+1,appearance_MAX5_array.shape[0]):
            score=bhattacharyya(appearance_MAX5_array[i,:]/sum(appearance_MAX5_array[i,:]), 
                                 appearance_MAX5_array[j,:]/sum(appearance_MAX5_array[j,:])) 
            appearance_score6[i][j]=score    
    print(appearance_score6)
           
            
    appearance_score=(appearance_score1+appearance_score2+appearance_score3+  \
        appearance_score4+appearance_score5+appearance_score6)
    
    print(appearance_score)

    return appearance_score
            
            
    

def bhattacharyya(a, b):
    if not len(a) == len(b):
        raise ValueError("a and b must be of the same size")
    return -math.log(sum((math.sqrt(u * w) for u, w in zip(a, b))))



                
if __name__ =='__main__':
    path_s2='/home/luoruizhi/isilon-home/S02_'
    path_s3='/home/luoruizhi/isilon-home/S03_'
    
    path_s5='/home/luoruizhi/isilon-home/S05_'
#     cal_appearance(path_s3)
    ID=0
    id_length=0
#     ID,id_length=write_bbox_result(path_s3,15,ID,id_length)
    ID,id_length=write_bbox_result(path_s2,15,ID,id_length)
    ID,id_length=write_bbox_result(path_s5,15,ID,id_length)
    
    
