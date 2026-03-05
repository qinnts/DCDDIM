import os
import torch
import scipy.io as sio
from torch.utils.data.dataset import Dataset
from torch.utils.data import Sampler
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
import pandas as pd
import cv2
import csv
import random
from scipy import stats
import nibabel as nib
from collections import defaultdict

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
    def __init__(self,label=-1,fold=0,k=5,phase="train", data_path="/data/qinfeng/datasets/", gene_path="/data/qinfeng/datasets/PPMI/PPMI_array_completed_imputed/"): #PPMI_SNPs2 PPMI_SNPs PPMI_PPMI_IMMUNO_fuse
        self.cls = label
        self.fold = fold
        self.path = data_path + "PPMI_2mm/AAL_ROISignals/"
        self.path2 = data_path + "PPMI/T1ImgMNI_cropwhite4/"
        self.gene_path = gene_path
        self.csv_path = data_path + "PPMI/label6.csv" #label5.csv 
        self.data_path="/data/qinfeng/datasets/PPMI/KLGANAE_MRI_f/" #KLAE_MRI_f AE_MRI_f
        self.data_path2="/data/qinfeng/datasets/PPMI/CLIP_f_cs400/"

        data = pd.read_csv(self.csv_path).values.tolist()
        self.subject = [str(i[0]) for i in data]
        self.cn_subject = [str(i[0]) for i in data if i[3]== 0]
        self.pd_subject = [str(i[0]) for i in data  if i[3]== 1]
        label_dict = {}
        age_dict = {}
        sex_dict = {}
        for index, sub in enumerate(self.subject):
            label_dict[sub] = data[index][3]
            age_dict[sub] = data[index][1]
            sex_dict[sub] = data[index][2]
        self.age_dict = age_dict
        self.sex_dict = sex_dict    

        self.gene_dict = {}
        gene_sub_list = [ file.split(".")[0] for file in os.listdir(gene_path)]

        # self.cn_subject = list(set(self.cn_subject).intersection(set(gene_sub_list))) 
        self.cn_subject.sort()            

        self.pd_subject = list(set(self.pd_subject).intersection(set(gene_sub_list))) 
        self.pd_subject.sort()          

        self.cn_subject = split_age_groups(age_dict, self.cn_subject,k)
        # self.pd_subject = split_age_groups(age_dict, self.pd_subject,2)[0:len(self.pd_subject)//2]
        self.pd_subject = split_age_groups(age_dict, self.pd_subject,k)
        
        self.subject_list = []
        self.add_list = []
        self.add_list2 = []
        assert k > 1
        fold_size = len(self.cn_subject) // k  # 每份的个数:数据总条数/折数（组数）
        fold_size2 = len(self.pd_subject) // k  # 每份的个数:数据总条数/折数（组数）

        for j in range(k):
            # idx = slice(j * fold_size, (j + 1) * fold_size)   
            # idx2 = slice(j * fold_size2, (j + 1) * fold_size2)  
            if j == k-1:
                idx = slice(j * fold_size, len(self.cn_subject))   
                idx2 = slice(j * fold_size2, len(self.pd_subject))   
            else:
                idx = slice(j * fold_size, (j + 1) * fold_size)   
                idx2 = slice(j * fold_size2, (j + 1) * fold_size2)     
                
            if phase == "train":
                if j is not fold: 
                    add_list = self.cn_subject[idx]
                    add_list2 = self.pd_subject[idx2]
                    self.add_list =  self.add_list + add_list
                    self.add_list2 =  self.add_list2 + add_list2
                    if self.cls == 0:
                        self.subject_list = self.subject_list + add_list
                    elif self.cls == 1:
                        self.subject_list = self.subject_list + add_list2
                    else:
                        self.subject_list = self.subject_list + add_list + add_list2
            elif  phase == "all":
                add_list = self.cn_subject[idx]
                add_list2 = self.pd_subject[idx2]
                self.add_list =  self.add_list + add_list
                self.add_list2 =  self.add_list2 + add_list2
                if self.cls == 0:
                    self.subject_list = self.subject_list + add_list
                elif self.cls == 1:
                    self.subject_list = self.subject_list + add_list2
                else:
                    self.subject_list = self.subject_list + add_list + add_list2
            else:
                if j == fold:  ###第i折作valid
                    add_list = self.cn_subject[idx]
                    add_list2 = self.pd_subject[idx2]
                    self.add_list =  self.add_list + add_list
                    self.add_list2 =  self.add_list2 + add_list2
                    if self.cls == 0:
                        self.subject_list = self.subject_list + add_list
                    elif self.cls == 1:
                        self.subject_list = self.subject_list + add_list2
                    else:
                        self.subject_list = self.subject_list + add_list + add_list2
                        
        # self.subject_list =  list(set(self.subject_list).intersection(set(gene_sub_list)))

        self.files = []
        self.label = []
        for file in os.listdir(self.path2):
            sub =  file.split('.')[0]
            if sub in self.subject_list:
                self.files.append(file)
                self.label.append(label_dict[sub])
        # The LabelEncoder encodes a sequence of bases as a sequence of integers.
        self.integer_encoder = LabelEncoder()
        # The OneHotEncoder converts an array of integers to a sparse matrix where
        self.one_hot_encoder = OneHotEncoder(categories='auto')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        fid = self.files[index]
        sub =  fid.split('.')[0]
        age_sex = np.array([self.age_dict[sub]/100, self.sex_dict[sub]]).astype(np.float32)
        
        # t1_data = self.get_img(sub+".nii.gz")
        t1_data = np.load(self.data_path+"fold0_"+fid.replace("nii.gz","npy")).astype(np.float32)#t1_data = self.get_img(sub+".nii.gz")
        data_range = np.load(f"PPMI_klgant1_range.npy").astype(np.float32)
        t1_std = np.load("PPMI_klgant1_std.npy").astype(np.float32)[np.newaxis,:,:]
        t1_mean = np.load("PPMI_klgant1_mean.npy").astype(np.float32)[np.newaxis,:,:]
        data_range = np.concatenate([data_range,t1_std,t1_mean],axis=0)

        label = np.array(self.label[index]).astype(np.int64)
        label[label != 0] = 1

        # integer_encoded = np.ones(1).astype(np.int64)
        # integer_encoded = np.load(self.gene_path+sub.split("_")[0]+".npy")    
        integer_encoded = np.load(self.data_path2+"fold0_snp_"+fid.replace("nii.gz","npy")).astype(np.float32)#np.ones(1).astype(np.int64)

        return fid, t1_data,label, data_range, age_sex, integer_encoded
    
    
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
            key = (int(dataset.age_dict[sub])//10, int(dataset.sex_dict[sub]))
            self.grouped_indices[key].append(idx)

        self.batches = []
        for key, indices in self.grouped_indices.items():
            for i in range(0, len(indices), self.batch_size):
                self.batches.append(indices[i:i + self.batch_size])

    def __iter__(self):
        self.batches = []
        for key, indices in self.grouped_indices.items():
            random.shuffle(indices)
            for i in range(0, len(indices), self.batch_size):
                self.batches.append(indices[i:i + self.batch_size])
        random.shuffle(self.batches)

        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)
