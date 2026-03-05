import os
import torch
import scipy.io as sio
from torch.utils.data.dataset import Dataset
from torch.utils.data import Sampler
from collections import defaultdict
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
import pandas as pd
import cv2
import csv
import random
from scipy import stats
import nibabel as nib

def split_age_groups(age_dict,sub_list,k):
    # 对年龄进行排序
    ages = [age_dict[i] for i in sub_list]
    sorted_ages = np.argsort(ages)

    # 创建空的二维列表
    age_groups = [[] for _ in range(k)]

    # 遍历排序后的年龄列表，将年龄添加到相应的分组中
    current_group = 0
    for age in sorted_ages:
        age_groups[current_group].append(sub_list[age])
        current_group = (current_group + 1) % k  # 循环填充每一列

    sub_list_age = []
    for group in age_groups:
        sub_list_age += group
    return sub_list_age

class MRIandGenedataset(Dataset):
    def __init__(self,label=-1,fold=0,k=5,phase="train", data_path="/data/qinfeng/datasets/", gene_path="/data/qinfeng/datasets/UKB/UKB_all_array_completed_imputed/",age_range=None,task=None): #ADNIALL_SNPs ADNIALL_PPMI_IMMUNO_fuse
        self.cls = label
        self.task = task
        self.fold = fold
        self.path = data_path + "UKB/T1_select_white4/"
        self.path2 = data_path  + "UKB/T1_select_white4/"
        self.csv_path = data_path + "UKB/label_5.csv"
        self.data_path="/data/qinfeng/datasets/UKB/KLGANAE_MRI_f/"#KLGANAE_MRI_f KLAE_MRI_f AE_MRI_f CLIP_f
        self.data_path2="/data/qinfeng/datasets/UKB/CLIP_f_cs1000_joint/"#KLAE_MRI_f AE_MRI_f CLIP_f CLIP_ff CLIP_f3
        self.gene_path = gene_path
  
        data = pd.read_csv(self.csv_path).values.tolist()
        self.subject = [str(i[0]) for i in data]
        self.cn_subject = [str(i[0]) for i in data if i[3]== 0]
        label_dict = {}
        age_dict = {}
        sex_dict = {}
        # temp = np.sum(np.array([1 for index in range(len(self.subject) ) if data[index][2]==2]))
        # temp2 = np.sum(np.array([1 for index in range(len(self.subject) ) if data[index][2]==1]))
        for index, sub in enumerate(self.subject):
            label_dict[sub] = data[index][3]
            age_dict[sub] = data[index][1]
            sex_dict[sub] = data[index][2]-1
        self.age_dict = age_dict
        self.sex_dict = sex_dict    

        self.tasks = ["reaction_time","symbol_digit","trail_making"]
        score_dict = {}
        for sub in self.subject:
            score_dict[sub] = []
        for task_index, task in enumerate(self.tasks):
            data = pd.read_csv(data_path + f"UKB/{task}.csv").values.tolist()
            value_list = [int(row[2]) for row in data]
            value_max = max(value_list)
            sub_list = [row[0] for row in data]
            for sub in self.subject:
                try:
                    index = sub_list.index(sub)
                    score_dict[sub].append(value_list[index])#1.0*value_list[index] / value_max
                except:
                    score_dict[sub].append( -1)#-1.0
        self.score_dict = score_dict

        ref_score_dict = {}
        for sub in self.subject:
            key = str(self.age_dict[sub])+"_"+str(self.sex_dict[sub])
            if key not in ref_score_dict.keys():
                ref_score_dict[key] = [score_dict[sub]]
            else:
                ref_score_dict[key].append(score_dict[sub])
        for key in ref_score_dict.keys():
            score_array = np.array(ref_score_dict[key])
            score_mask = score_array != -1
            ref_score_dict[key] = np.sum(score_array*score_mask,axis=0) / (np.sum(score_mask,axis=0) + 1e-12)
        self.ref_score_dict = ref_score_dict

        gene_sub_list = [file.split(".")[0] for file in os.listdir(gene_path)]
        self.cn_subject = [sub  for sub in self.cn_subject  if (sub.split("_")[0] in gene_sub_list)]#and (self.age_dict[sub] >= 70) and (self.age_dict[sub]< 75)]# and (self.sex_dict[sub]==1)
        self.cn_subject.sort()  
        self.cn_subject_male = [sub  for sub in self.cn_subject  if (self.sex_dict[sub]==0)]
        self.cn_subject_female = [sub  for sub in self.cn_subject  if (self.sex_dict[sub]==1)]
        self.cn_subject_male.sort()  
        self.cn_subject_female.sort()  

        # self.cn_subject = split_age_groups(age_dict, self.cn_subject,2)[0:len(self.cn_subject)//2]        
        # self.cn_subject = split_age_groups(age_dict, self.cn_subject,k)
        self.cn_subject_male = split_age_groups(age_dict, self.cn_subject_male,k)
        self.cn_subject_female = split_age_groups(age_dict, self.cn_subject_female,k)
        
        self.subject_list = []
        self.add_list = []
        assert k > 1
        # fold_size = len(self.cn_subject) // k 
        fold_size = len(self.cn_subject_male) // k  # 每份的个数:数据总条数/折数（组数）
        fold_size2 = len(self.cn_subject_female) // k  # 每份的个数:数据总条数/折数（组数）

        for j in range(k):
            # if j == k-1:
            #     idx = slice(j * fold_size, len(self.cn_subject))   
            # else:
            #     idx = slice(j * fold_size, (j + 1) * fold_size)   
            # if phase == "train":
            #     if j is not fold: 
            #         add_list = self.cn_subject[idx]
            #         self.subject_list = self.subject_list + add_list
            # elif phase == "all":
            #     add_list = self.cn_subject[idx]
            #     self.subject_list = self.subject_list + add_list
            # else:
            #     if j == fold:  ###第i折作valid
            #         add_list = self.cn_subject[idx]
            #         self.subject_list = self.subject_list + add_list
            
            if j == k-1:
                idx = slice(j * fold_size, len(self.cn_subject_male))   
                idx2 = slice(j * fold_size2, len(self.cn_subject_female))   
            else:
                idx = slice(j * fold_size, (j + 1) * fold_size)   
                idx2 = slice(j * fold_size2, (j + 1) * fold_size2)   
           
            if phase == "train":
                if j is not fold: 
                    add_list = self.cn_subject_male[idx]
                    add_list2 = self.cn_subject_female[idx2]
                    self.subject_list = self.subject_list+add_list+add_list2
            elif phase == "all":
                add_list = self.cn_subject_male[idx]
                add_list2 = self.cn_subject_female[idx2]
                self.subject_list = self.subject_list+add_list+add_list2
            else:
                if j == fold:  ###第i折作valid
                    add_list = self.cn_subject_male[idx]
                    add_list2 = self.cn_subject_female[idx2]
                    self.subject_list = self.subject_list +add_list+add_list2
                
        if age_range is not None:
            self.subject_list = [sub  for sub in self.subject_list  if (self.age_dict[sub] >= age_range[0]) and (self.age_dict[sub]< age_range[1])]
        
        if self.task is not None:
            task_index = self.tasks.index(self.task)
            if self.cls == 0:
                self.subject_list = [sub  for sub in self.subject_list  if (self.score_dict[sub][task_index] ==0)]
            elif self.cls ==1:
                self.subject_list = [sub  for sub in self.subject_list  if (self.score_dict[sub][task_index] ==1)]
            else:
                self.subject_list = [sub  for sub in self.subject_list  if (self.score_dict[sub][task_index] ==0) or (self.score_dict[sub][task_index] ==1)]
       
        self.files = []
        self.label = []
        for file in os.listdir(self.path2):
            sub =  file.split('.')[0]
            if sub in self.subject_list:
                self.files.append(file)
                self.label.append(label_dict[sub])
        # The LabelEncoder encodes a sequence of bases as a sequence of integers.
        # temp = self.files.index("5853731_2_0.nii.gz")
        self.integer_encoder = LabelEncoder()
        # The OneHotEncoder converts an array of integers to a sparse matrix where
        self.one_hot_encoder = OneHotEncoder(categories='auto')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        fid = self.files[index]
        sub =  fid.split('.')[0]
        # age_sex = np.array([1.0*(int(self.age_dict[sub])//5)/20, self.sex_dict[sub]]).astype(np.float32)
        age_sex = np.array([1.0*self.age_dict[sub]/100, self.sex_dict[sub]]).astype(np.float32)
        # age_sex = np.array([0, self.sex_dict[sub]]).astype(np.float32)
        
        # t1_data = self.get_img(sub+".nii.gz")
        t1_data = np.load(self.data_path+"fold0_"+fid.replace("nii.gz","npy")).astype(np.float32)#t1_data = self.get_img(sub+".nii.gz")
        data_range = np.load(f"UKB_klgant1_range.npy").astype(np.float32)
        t1_std = np.load("UKB_klgant1_std.npy").astype(np.float32)[np.newaxis,:,:]
        t1_mean = np.load("UKB_klgant1_mean.npy").astype(np.float32)[np.newaxis,:,:]
        data_range = np.concatenate([data_range,t1_std,t1_mean],axis=0)
        
        scores = np.array(self.score_dict[sub]).astype(np.int64)
        label =  np.array(self.label[index]).astype(np.int64)#

        # sequence = pd.read_csv(self.gene_path+sub.split("_")[0]+".csv",usecols=['value']).values[:,0].tolist()
        # integer_encoded = self.integer_encoder.fit_transform(sequence)
        # integer_encoded = integer_encoded.astype(np.int64)
         
        # integer_encoded = np.load(self.gene_path+sub.split("_")[0]+".npy")    
        integer_encoded = np.load(self.data_path2+"fold0_snp_"+fid.replace("nii.gz","npy")).astype(np.float32)#np.ones(1).astype(np.int64)
        
        return fid, t1_data,label,data_range, age_sex, integer_encoded,scores
    
    def get_img(self,fid):
        sub_path = self.path2+fid
        img = self.nii_loader(sub_path)
        img = np.expand_dims(img, axis=0)
        img = np.ascontiguousarray(img, dtype= np.float32)
        img = torch.from_numpy(img).type(torch.FloatTensor)
        return img
    
    def nii_loader(self, path):
        img = nib.load(str(path))
        data = img.get_fdata()
        return data
   
class GroupedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.batch_size = batch_size
        self.grouped_indices = defaultdict(list)

        # for idx in range(len(dataset)):
        #     item = dataset[idx]
        #     key = (int(item[-2][0]*100), item[-2][1])
        #     key = (item[-2][1])
        for idx,fid in enumerate(dataset.files):
            sub =  fid.split('.')[0]
            key = (int(dataset.age_dict[sub]), int(dataset.sex_dict[sub]))
            self.grouped_indices[key].append(idx)

        self.batches = []
        for key, indices in self.grouped_indices.items():
            for i in range(0, len(indices), self.batch_size):
                 indices_cut = indices[i:i + self.batch_size]
                 remainder = len(indices_cut)% self.batch_size
                 if remainder !=0:
                    continue
                    # pad_len = self.batch_size - remainder
                    # indices_cut = np.resize(indices_cut,len(indices_cut)+pad_len)
                 self.batches.append(indices_cut)

    def __iter__(self):
        self.batches = []
        for key, indices in self.grouped_indices.items():
            random.shuffle(indices)
            for i in range(0, len(indices), self.batch_size):
                indices_cut = indices[i:i + self.batch_size]
                remainder = len(indices_cut)% self.batch_size
                if remainder !=0:
                    continue
                    # pad_len = self.batch_size - remainder
                    # indices_cut = np.resize(indices_cut,len(indices_cut)+pad_len)
                self.batches.append(indices_cut)
        random.shuffle(self.batches)

        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)
