#!/usr/bin/env python
# coding: utf-8
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import math
from abc import abstractmethod
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from t1f_gene_clip_adniall_interp_dataset2 import  MRIandGenedataset  as MRIandGenedatasetADNI
from t1f_gene_clip_ppmi_interp_dataset2 import  MRIandGenedataset  as MRIandGenedatasetPPMI
from t1f_gene_clip_ukb_interp_dataset2 import  MRIandGenedataset, GroupedBatchSampler
from options.train_options import TrainOptions
from bigbird_model import Maskcompute7
from attention import SpatialTransformer5
from ScaleDense import GeneMLPEncoder,ScaleDense_VAE3
from vit import MRIMambaMAE,GeneMambaMAE
from mamba_model import Mamba_dflow
import copy
import time
import nibabel as nib 
import itertools

def mem(tag=""):
    import torch
    torch.cuda.synchronize()
    a = torch.cuda.memory_allocated() / 1024**2
    r = torch.cuda.memory_reserved()  / 1024**2
    print(f"[{tag}] allocated={a:.1f} MiB, reserved={r:.1f} MiB")

def topk_keep(x: torch.Tensor, k: int):
    topk_vals, _ = torch.topk(x, k, dim=-1, largest=True)
    thresh = topk_vals[..., -1, None]
    mask = (x >= thresh).to(x.dtype)          # (B, L) 中 0 / 1
    return mask + x - x.detach()#(x * mask).detach() + x - x.detach()#

class CLIP(nn.Module):
    def __init__(self, temperature=0.07,feature_num=512):
        super(CLIP, self).__init__()
        self.image_proj = nn.Linear(feature_num, 512)
        self.snp_proj = nn.Linear(feature_num, 512)
        self.male_fc = nn.Linear(2,1024)
        self.image_pooling = nn.AdaptiveMaxPool2d((1,feature_num))#nn.AdaptiveAvgPool2d((1,config.hidden_size))#
        self.snp_pooling = nn.AdaptiveMaxPool2d((1,feature_num))#nn.AdaptiveAvgPool2d((1,config.hidden_size))#
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))
        self.nolin = nn.ELU()
        self.norm = nn.GroupNorm(4,feature_num)

    def forward(self, image_features=None, snp_features=None, mask=None, group=False, age_sex=None):
        # image_features = self.image_pooling(image_features)
        image_features = image_features.view(image_features.size(0), -1)   
        image_features = self.image_proj(image_features)
       
        # snp_features = self.snp_pooling(snp_features)
        snp_features = snp_features.view(snp_features.size(0), -1)
        style = self.nolin(self.male_fc(age_sex))
        snp_features = (1.0+style[:,0:512]) *  self.norm(snp_features) + style[:,512:]
        snp_features = self.snp_proj(snp_features)

        clip_scores = F.cosine_similarity(image_features, snp_features)
        # normalized features
        image_features_norm = image_features / image_features.norm(dim=1, keepdim=True)
        snp_features_norm = snp_features / snp_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features_norm @ snp_features_norm.t()

        if group:
            loss_img = torch.sum(-1.0 * F.log_softmax(logits, dim=0) * mask  / (torch.sum(mask,dim=0,keepdim=True)+1e-12) ) / logits.shape[0]
            loss_snp = torch.sum(-1.0 * F.log_softmax(logits.t(), dim=0) * mask  / (torch.sum(mask,dim=0,keepdim=True)+1e-12) ) / logits.shape[0]
            loss = loss_img + loss_snp
        else:
            # 计算对角线元素（正样本）的索引
            labels = torch.arange(logits.shape[0], device=logits.device)
            # 计算损失
            loss = F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)

        return loss / 2,clip_scores,image_features_norm,snp_features_norm
    
    def forward2(self, image_features=None, snp_features=None,age_sex=None):
        # image_features = self.image_pooling(image_features)
        image_features = image_features.view(image_features.size(0), -1)
        image_features = self.image_proj(image_features)
       
        # snp_features = self.snp_pooling(snp_features)
        snp_features = snp_features.view(snp_features.size(0), -1)
        style = self.nolin(self.male_fc(age_sex))
        snp_features = (1.0+style[:,0:512]) *  self.norm(snp_features) + style[:,512:]
        snp_features = self.snp_proj(snp_features)
        
        # normalized features
        image_features_norm = image_features / image_features.norm(dim=1, keepdim=True)
        snp_features_norm = snp_features / snp_features.norm(dim=1, keepdim=True)

        return image_features_norm,snp_features_norm
    
    def forward3(self, image_features=None, snp_features=None,age_sex=None, mask=None, group=False):
       # image_features = self.image_pooling(image_features)
        image_features = image_features.view(image_features.size(0), -1)
        image_features = self.image_proj(image_features)
        image_features_norm = image_features / image_features.norm(dim=1, keepdim=True)

        snp_features_norm = snp_features / snp_features.norm(dim=1, keepdim=True)
        
        clip_scores = F.cosine_similarity(image_features_norm, snp_features_norm)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features_norm @ snp_features_norm.t()

        if group:
            loss_img = torch.sum(-1.0 * F.log_softmax(logits, dim=0) * mask  / (torch.sum(mask,dim=0,keepdim=True)+1e-12) ) / logits.shape[0]
            loss_snp = torch.sum(-1.0 * F.log_softmax(logits.t(), dim=0) * mask  / (torch.sum(mask,dim=0,keepdim=True)+1e-12) ) / logits.shape[0]
            loss = loss_img + loss_snp
        else:
            # 计算对角线元素（正样本）的索引
            labels = torch.arange(logits.shape[0], device=logits.device)
            # 计算损失
            loss = F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)

        return loss/2, image_features, snp_features

    def forward4(self, snp_features=None,age_sex=None):
        # snp_features = self.snp_pooling(snp_features)
        snp_features = snp_features.view(snp_features.size(0), -1)
        style = self.nolin(self.male_fc(age_sex))
        snp_features = (1.0+style[:,0:512]) *  self.norm(snp_features) + style[:,512:]
        snp_features = self.snp_proj(snp_features)
        
        # normalized features
        snp_features_norm = snp_features / snp_features.norm(dim=1, keepdim=True)

        return snp_features_norm
    
def nii_loader(path):
    img = nib.load(str(path))
    data = img.get_fdata()
    img_affine = img._affine
    return data,img_affine
      
# beta schedule
def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)

def sigmoid_beta_schedule(timesteps):
    betas = torch.linspace(-6, 6, timesteps)
    betas = torch.sigmoid(betas)/(betas.max()-betas.min())*(0.02-betas.min())/10
    return betas

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class GaussianDiffusion:
    def __init__(
        self,
        timesteps=1000,
        beta_schedule='linear'
    ):
        self.timesteps = timesteps
        
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        elif beta_schedule == 'sigmoid':
            betas = sigmoid_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')
        self.betas = betas
            
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.)
        
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # below: log calculation clipped because the posterior variance is 0 at the beginning
        # of the diffusion chain
        #self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min =1e-20))
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        )
        
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * torch.sqrt(self.alphas)
            / (1.0 - self.alphas_cumprod)
        )

        # self.t1_std = torch.from_numpy(np.load("klgant1_std.npy")).unsqueeze(0)
        # self.t1_mean = torch.from_numpy(np.load("klgant1_mean.npy")).unsqueeze(0)
    
    # get the param of given timestep t
    def _extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t).float()
        out = out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
        return out
    
    # forward diffusion (using the nice property): q(x_t | x_0)
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return (sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise) /  (sqrt_alphas_cumprod_t+sqrt_one_minus_alphas_cumprod_t)
    
    # use ddim to sample
    @torch.no_grad()
    def ddim_sample(
        self,
        model,
        image_size,
        n_sample=None,
        batch_size=8,
        channels=3,
        ddim_timesteps=50,
        n_class = 10,
        w = 2,
        mode= 'random',
        ddim_discr_method="uniform",
        ddim_eta=0.0,
        clip_denoised=True,
        t1_range=None, context=None, age_sex=None, snp=None,template=None):
             
        min_tensor = t1_range[:,0,:,:].unsqueeze(1)
        max_tensor = t1_range[:,1,:,:].unsqueeze(1)
        t1_std = t1_range[:,2,:,:].unsqueeze(1)
        t1_mean = t1_range[:,3,:,:].unsqueeze(1)
        
        # make ddim timestep sequence
        if ddim_discr_method == 'uniform':
            c = self.timesteps // ddim_timesteps
            ddim_timestep_seq = np.asarray(list(range(0, self.timesteps, c)))
        elif ddim_discr_method == 'quad':
            ddim_timestep_seq = (
                (np.linspace(0, np.sqrt(self.timesteps*.99), ddim_timesteps)) ** 2
            ).astype(int)
        else:
            raise NotImplementedError(f'There is no ddim discretization method called "{ddim_discr_method}"')
        # add one to get the final alpha values right (the ones from first scale to data during sampling)
        ddim_timestep_seq = ddim_timestep_seq + 1
        # previous sequence
        ddim_timestep_prev_seq = np.append(np.array([0]), ddim_timestep_seq[:-1])
        
        device = next(model.parameters()).device

        # start from pure noise (for each example in the batch)
        # sample_img_list = []
        # for i in range(n_sample):
        #     sample_img = torch.randn((1, channels, image_size[0], image_size[1]), device=device) 
        #     sample_img = sample_img* t1_std + t1_mean
        #     sample_img = torch.where(sample_img <  min_tensor.repeat(sample_img.shape[0],1,1,1), min_tensor.repeat(sample_img.shape[0],1,1,1), sample_img)
        #     sample_img = torch.where(sample_img >  max_tensor.repeat(sample_img.shape[0],1,1,1), max_tensor.repeat(sample_img.shape[0],1,1,1), sample_img)
        #     sample_img = sample_img.repeat(batch_size,1,1,1)
        #     sample_img_list.append(sample_img)
        # sample_img = torch.cat(sample_img_list,dim=0)

        sample_img = torch.randn((batch_size*n_sample, channels, image_size[0], image_size[1]), device=device) 
        sample_img = sample_img* t1_std + t1_mean
        sample_img = torch.where(sample_img <  min_tensor.repeat(sample_img.shape[0],1,1,1), min_tensor.repeat(sample_img.shape[0],1,1,1), sample_img)
        sample_img = torch.where(sample_img >  max_tensor.repeat(sample_img.shape[0],1,1,1), max_tensor.repeat(sample_img.shape[0],1,1,1), sample_img)
        
        context = context.repeat(n_sample)
        age_sex = age_sex.repeat(n_sample,1)
        template = template.repeat(n_sample,1,1,1)
        if snp is not None:
            snp = snp.repeat(n_sample,1)
        
        for i in tqdm(reversed(range(0, ddim_timesteps)), desc='sampling loop time step', total=ddim_timesteps):
            t = torch.full((batch_size*n_sample,), ddim_timestep_seq[i], device=device, dtype=torch.long)
            prev_t = torch.full((batch_size*n_sample,), ddim_timestep_prev_seq[i], device=device, dtype=torch.long)
            
            # 1. get current and previous alpha_cumprod
            alpha_cumprod_t = self._extract(self.alphas_cumprod, t, sample_img.shape)
            alpha_cumprod_t_prev = self._extract(self.alphas_cumprod, prev_t, sample_img.shape)
    
            # 2. predict noise using model
            pred_noise_c = model(sample_img, t/self.timesteps, context, torch.ones(batch_size*n_sample).int().cuda(),age_sex, snp,template)
            pred_noise_none = model(sample_img, t/self.timesteps, context, torch.zeros(batch_size*n_sample).int().cuda(),age_sex, snp,template)
            pred_noise = (1+w)*pred_noise_c - w*pred_noise_none
           
            if clip_denoised:
                pred_noise = torch.where(pred_noise <  min_tensor.repeat(pred_noise.shape[0],1,1,1), min_tensor.repeat(pred_noise.shape[0],1,1,1), pred_noise)
                pred_noise = torch.where(pred_noise >  max_tensor.repeat(pred_noise.shape[0],1,1,1), max_tensor.repeat(pred_noise.shape[0],1,1,1), pred_noise)
                
            # 3. get the predicted x_0
            c_t = torch.sqrt(alpha_cumprod_t) + torch.sqrt(1. - alpha_cumprod_t)
            prev_c_t = torch.sqrt(alpha_cumprod_t_prev) + torch.sqrt(1. - alpha_cumprod_t_prev)
            pred_x0 = (c_t*sample_img - torch.sqrt((1. - alpha_cumprod_t)) * pred_noise) / torch.sqrt(alpha_cumprod_t)
           
            if clip_denoised:
                pred_x0 = torch.where(pred_x0 <  min_tensor.repeat(pred_x0.shape[0],1,1,1), min_tensor.repeat(pred_x0.shape[0],1,1,1), pred_x0)
                pred_x0 = torch.where(pred_x0 >  max_tensor.repeat(pred_x0.shape[0],1,1,1), max_tensor.repeat(pred_x0.shape[0],1,1,1), pred_x0)

            # 4. compute variance: "sigma_t(η)" -> see formula (16)
            # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
            sigmas_t = ddim_eta * torch.sqrt(
                (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_t_prev)) / prev_c_t
            
            # 5. compute "direction pointing to x_t" of formula (12)
            pred_dir_xt = torch.sqrt((1 - alpha_cumprod_t_prev)/(prev_c_t**2) - sigmas_t**2) * pred_noise
            
            # 6. compute x_{t-1} of formula (12)
            # noise_t_list = []
            # for i in range(n_sample):
            #     noise_t = torch.randn((1, channels,  image_size[0], image_size[1]), device=device) 
            #     noise_t = noise_t * t1_std + t1_mean
            #     noise_t = torch.where(noise_t <  min_tensor.repeat(noise_t.shape[0],1,1,1), min_tensor.repeat(noise_t.shape[0],1,1,1), noise_t)
            #     noise_t = torch.where(noise_t >  max_tensor.repeat(noise_t.shape[0],1,1,1), max_tensor.repeat(noise_t.shape[0],1,1,1), noise_t)
            #     noise_t = noise_t.repeat(batch_size,1,1,1)
            #     noise_t_list.append(noise_t)
            # noise_t = torch.cat(noise_t_list,dim=0)
            noise_t = torch.randn((n_sample*batch_size, channels,  image_size[0], image_size[1]), device=device) 
            noise_t = noise_t * t1_std + t1_mean
            noise_t = torch.where(noise_t <  min_tensor.repeat(noise_t.shape[0],1,1,1), min_tensor.repeat(noise_t.shape[0],1,1,1), noise_t)
            noise_t = torch.where(noise_t >  max_tensor.repeat(noise_t.shape[0],1,1,1), max_tensor.repeat(noise_t.shape[0],1,1,1), noise_t)
                
            x_prev = torch.sqrt(alpha_cumprod_t_prev / (prev_c_t**2)) * pred_x0 + pred_dir_xt + sigmas_t * noise_t
           
            if clip_denoised:
                x_prev = torch.where(x_prev <  min_tensor.repeat(x_prev.shape[0],1,1,1), min_tensor.repeat(x_prev.shape[0],1,1,1), x_prev)
                x_prev = torch.where(x_prev >  max_tensor.repeat(x_prev.shape[0],1,1,1), max_tensor.repeat(x_prev.shape[0],1,1,1), x_prev)     
            
            sample_img = x_prev
            # if mode == 'all':
            #     seq_img.append(sample_img.cpu().numpy())
                 
        sample_img = torch.where(sample_img <  min_tensor.repeat(sample_img.shape[0],1,1,1), min_tensor.repeat(sample_img.shape[0],1,1,1), sample_img)
        sample_img = torch.where(sample_img >  max_tensor.repeat(sample_img.shape[0],1,1,1), max_tensor.repeat(sample_img.shape[0],1,1,1), sample_img) 
        return sample_img
        # if mode == 'all':
        #     return seq_img
        # else:
        #     return sample_img.cpu().numpy()
        
     # compute train losses
    
    # compute train losses
    def train_losses(self, model, x_start, t, c, mask_c, t1_range,age_sex, snp=None):
        min_tensor = t1_range[:,0,:,:].unsqueeze(1)
        max_tensor = t1_range[:,1,:,:].unsqueeze(1)
        t1_std = t1_range[:,2,:,:].unsqueeze(1)
        t1_mean = t1_range[:,3,:,:].unsqueeze(1)
        # x_start = (x_start-min_tensor) / (max_tensor-min_tensor+1e-10) * 2 - 1
        # generate random noise
        noise = torch.randn_like(x_start) * t1_std +  t1_mean
        # noise = torch.clamp(noise, min=-1., max=1.)
        noise = torch.where(noise <  min_tensor, min_tensor, noise)
        noise = torch.where(noise >  max_tensor, max_tensor, noise)
        
        x_noisy = self.q_sample(x_start, t, noise=noise) 
        predicted_noise = model(x_noisy, t/self.timesteps, c, mask_c,age_sex, snp,x_start)       
        loss = F.mse_loss(noise, predicted_noise) #* 10

        alpha_cumprod_t = self._extract(self.alphas_cumprod, t, x_noisy.shape)
        c_t = torch.sqrt(alpha_cumprod_t) + torch.sqrt(1. - alpha_cumprod_t)
        predicted_x0 = (c_t*x_noisy - torch.sqrt((1. - alpha_cumprod_t)) * predicted_noise) / torch.sqrt(alpha_cumprod_t)
           
        return loss, predicted_x0

criterion = nn.CrossEntropyLoss().cuda()

data_path = "/data/qinfeng/datasets/UKB/T1_select_white/"
ori_array, img_affine = nii_loader(data_path+"3664415_2_0.nii.gz")

opt = TrainOptions().parse()

# define model and diffusion
device = "cuda" if torch.cuda.is_available() else "cpu"
datasets_all = ["UKB","ADNI","PPMI"]
datasets_use = ["UKB","ADNI","PPMI"]

ae_dir = f"./generation_models/UKB-ADNI-PPMI_MRI_VAE-32" #140
pre_ep = 200

SNP_TOPK = 1000
clip_dir = f"./generation_models/UKB-ADNI-PPMI_T1-GENE-CLIP_MAE_SNPMask_1000_v3_ff" #_Mask_{opt.mask_dropout}_{opt.mask_dropout2}
pre_ep2 = 100

dm_dir = f"./generation_models/{'-'.join(datasets_use)}_iddim_genetranst1"
pre_ep3 = 250#450

timesteps = 500
WIDTH = 32
TRAIN_BATCH_SIZE = 8
TEST_BATCH_SIZE = 8
WORKERS = TRAIN_BATCH_SIZE
PATCH = 512
NUM = 10*13*11#1200#1200 #
NUM2 = 6316#12622#12622#
SNP_NUM = 3233344#3231148
test_timesteps = 100
N_SAMPLE = 5

fold = 0

dataset_train_1 = MRIandGenedataset(fold=fold,phase="train")
# sampler = GroupedBatchSampler(dataset_train_1, TRAIN_BATCH_SIZE)
# data_loader_train_1 = torch.utils.data.DataLoader(dataset_train_1, batch_sampler=sampler,num_workers=WORKERS,pin_memory=True)
data_loader_train_1 = torch.utils.data.DataLoader(dataset_train_1, batch_size=TRAIN_BATCH_SIZE, shuffle=False,
                                                num_workers=WORKERS,pin_memory=True)
dataset_size_1 = len(data_loader_train_1)
print("dataset_size_1: ", dataset_size_1)

dataset_test_1 = MRIandGenedataset(fold=fold,phase="test")
data_loader_test_1 = torch.utils.data.DataLoader(dataset_test_1, batch_size=TEST_BATCH_SIZE, shuffle=False,
                                                num_workers=WORKERS,pin_memory=True)
dataset_size_test_1 = len(data_loader_test_1)
print("data_loader_test_1: ", dataset_size_test_1)

if "ADNI" in datasets_use:
    dataset_train_2 = MRIandGenedatasetADNI(fold=fold,phase="train",label=0)
    data_loader_train_2 = torch.utils.data.DataLoader(dataset_train_2, batch_size=TRAIN_BATCH_SIZE//2, shuffle=True,
                                                    num_workers=WORKERS)
    dataset_train_3 = MRIandGenedatasetADNI(fold=fold,phase="train",label=1)
    data_loader_train_3 = torch.utils.data.DataLoader(dataset_train_3, batch_size=TRAIN_BATCH_SIZE//2, shuffle=True,
                                                    num_workers=WORKERS)
    data_loader_train_3 = itertools.cycle(data_loader_train_3)

    dataset_size_train_2 = len(data_loader_train_2)
    print("data_loader_train_2: ", dataset_size_train_2)

    dataset_test_2 = MRIandGenedatasetADNI(fold=fold,phase="test",label=[0,1])
    data_loader_test_2 = torch.utils.data.DataLoader(dataset_test_2, batch_size=TEST_BATCH_SIZE, shuffle=False,
                                                    num_workers=WORKERS)
    dataset_size_test_2 = len(data_loader_test_2)
    print("data_loader_test_2: ", dataset_size_test_2)

if "PPMI" in datasets_use:
    dataset_train_4 = MRIandGenedatasetPPMI(fold=fold,phase="train",label=0)
    data_loader_train_4 = torch.utils.data.DataLoader(dataset_train_4, batch_size=TRAIN_BATCH_SIZE//2, shuffle=True,
                                                    num_workers=WORKERS)
    dataset_train_5 = MRIandGenedatasetPPMI(fold=fold,phase="train",label=1)
    data_loader_train_5 = torch.utils.data.DataLoader(dataset_train_5, batch_size=TRAIN_BATCH_SIZE//2, shuffle=True,
                                                    num_workers=WORKERS)
    data_loader_train_4 = itertools.cycle(data_loader_train_4)
    dataset_size_train_5 = len(data_loader_train_5)
    print("data_loader_train_5: ", dataset_size_train_5)

    dataset_test_3 = MRIandGenedatasetPPMI(fold=fold,phase="test",label=[0,1])
    data_loader_test_3 = torch.utils.data.DataLoader(dataset_test_3, batch_size=TEST_BATCH_SIZE, shuffle=False,
                                                    num_workers=WORKERS)
    dataset_size_test_3 = len(data_loader_test_3)
    print("data_loader_test_3: ", dataset_size_test_3)

VAE = ScaleDense_VAE3(nb_filter=32, latent_dim=32).cuda()
VAE = nn.DataParallel(VAE)
VAE.load_state_dict(torch.load(ae_dir+f"/fold_{fold}_E_{pre_ep}.pth"))

gaussian_diffusion = GaussianDiffusion(timesteps=timesteps, beta_schedule='linear')
model = SpatialTransformer5(in_channels=32, depth=6, n_heads=4, d_head=256, n_classes=4,num_tokens=1200).cuda()
model = nn.DataParallel(model)
model.load_state_dict(torch.load(dm_dir+f"/fold_{fold}_ddim_{pre_ep3}.pth"),strict=False)

opt.hidden_dropout_prob = 0.2
opt.emb_dropout_prob = 0.2
opt.classifier_dropout = 0
opt2 = copy.deepcopy(opt)
opt2.max_position_embeddings = 30000#

G = Mamba_dflow(opt).cuda()#Mamba_dflow_MoE(opt).cuda()#BigBird3(opt).cuda()
G2 = Mamba_dflow(opt2).cuda()#Mamba_dflow_MoE(opt).cuda()#BigBird3(opt).cuda()
E = MRIMambaMAE(opt).cuda()
E2  =  GeneMLPEncoder(opt2,input_channels=PATCH*3, latent_feature=512).cuda() #GeneMambaMAE(opt2).cuda() #
model_clip = CLIP().cuda()
M2 = Maskcompute7(opt2,NUM2).cuda()#Maskcompute3(opt2,PATCH).cuda()#
G2 = nn.DataParallel(G2)
E2 = nn.DataParallel(E2)
G = nn.DataParallel(G)
E = nn.DataParallel(E)
M2 = nn.DataParallel(M2)

E.load_state_dict(torch.load(clip_dir+f"/fold_{fold}_E_{pre_ep2}.pth")) #100
G.load_state_dict(torch.load(clip_dir+f"/fold_{fold}_G_{pre_ep2}.pth")) #100
E2.load_state_dict(torch.load(clip_dir+f"/fold_{fold}_E2_{pre_ep2}.pth")) #100
G2.load_state_dict(torch.load(clip_dir+f"/fold_{fold}_G2_{pre_ep2}.pth")) #100
M2.load_state_dict(torch.load(clip_dir+f"/fold_{fold}_M2_{pre_ep2}.pth")) #100
model_clip.load_state_dict(torch.load(clip_dir+f"/fold_{fold}_clip_{pre_ep2}.pth")) #100

w_sl = torch.ones([NUM,1])
w_sl = w_sl.cuda()
w_sl.requires_grad = False
w_sl2 = torch.ones([NUM2,1])
w_sl2 = w_sl2.cuda()
w_sl2.requires_grad = False

for p in model.parameters():
    p.requires_grad = False
for p in E.parameters():
    p.requires_grad = False
for p in G.parameters():
    p.requires_grad = False
for p in E2.parameters():
    p.requires_grad = False
for p in G2.parameters():
    p.requires_grad = False
for p in model_clip.parameters():
    p.requires_grad = False
for p in M2.parameters():
    p.requires_grad = False
for p in VAE.parameters():
    p.requires_grad = False

model.eval()
E.eval()
G.eval()
E2.eval()
G2.eval()
M2.eval()
model_clip.eval()
VAE.eval()

ws_test = [2]

term = "MAPT" 
snp_index_list = [1220748]
term_value = torch.Tensor([0]).unsqueeze(0)
term_value2 = 2 - term_value

# term = "APOE"
# snp_index_list = [1416801]
# term_value = torch.Tensor([0]).unsqueeze(0)
# term_value2 = 2 - term_value

sparse_flag2 = 1
for w_i, w in enumerate(ws_test):
    save_dir =  f"./generation_models/UKB_iddim_genetranst132_sample/"
    os.system("mkdir -p {}".format(save_dir))

    with torch.no_grad():
        for  train_data in data_loader_train_1:
            fids, x,  c, t1_range, age_sex, integer_encoded,_  = train_data
            print(fids)
            B = x.shape[0]
            x = x.to(device)
            c = c.to(device)
            t1_range = t1_range.to(device)
            age_sex = age_sex.to(device)
            snp = integer_encoded.to(device)

            batch_mask = torch.ones(x.shape[0]).int().to(device)           
            feature = x.unsqueeze(1)
            t1_range = t1_range[0].unsqueeze(0)
            
            w_sl_use2 = torch.ones_like(w_sl2).cuda().unsqueeze(0).repeat(B,1,1)
            gate2 = F.gumbel_softmax(torch.log(torch.cat([w_sl_use2, 1.0-w_sl_use2],dim=2)+1e-10)*10, tau = 1, dim = 2, hard = False)[:,:,0]
            
            for snp_value_index,snp_value in enumerate([term_value,term_value2]):
                snp[:,snp_index_list] = snp_value.to(snp).repeat(B,1)

                feature2,mask_snp =  E2(snp,p=PATCH)
                _,_, snp_feature, feature2_emb = G2(gate = gate2,inputs_embeds=feature2,output_embedding=True, use_embedding = False) 
                snp_feature_s = model_clip.forward4(snp_feature,age_sex)

                if sparse_flag2:
                    mask2 = M2(input_features=feature2_emb,input_features2=snp_feature_s)[:,0:SNP_NUM]
                else:
                    mask2 = torch.ones([SNP_NUM]).cuda().unsqueeze(0).repeat(B,1)
                if sparse_flag2:
                    gate2_c_hard = topk_keep(F.softmax(mask2,dim=-1),SNP_TOPK) 
                else:
                    gate2_c_hard =  mask2

                feature2_c,_ =  E2(snp,p=PATCH,gate=gate2_c_hard)
                _,_, snp_feature_c = G2(gate = gate2, inputs_embeds=feature2_c,batch = B, use_embedding = False)     
                snp_feature_c=  model_clip.forward4(snp_feature_c,age_sex=age_sex)

                feature_gen = gaussian_diffusion.ddim_sample(model, image_size=[1200,32], n_sample=N_SAMPLE, batch_size=B, 
                                                        channels=1, ddim_timesteps=test_timesteps,w=w, mode='all', 
                                                        ddim_discr_method='quad', ddim_eta=0, clip_denoised=False, 
                                                        t1_range=t1_range, context=c,age_sex=age_sex,snp = snp_feature_c,template=feature)
                feature_gen = feature_gen.squeeze(1).permute([0,2,1]).view([B*N_SAMPLE,32,10,12,10])
                
                x_0 = VAE.module.decoder_conv(feature_gen)
                x_1 =  VAE.module.up1(x_0)
                x_2 =  VAE.module.up2(x_1)
                x_3 =  VAE.module.up3(x_2)
                x_gen = VAE.module.out3(x_3)
                x_gen_array = x_gen.detach().cpu().numpy()
                for i in range(N_SAMPLE):
                    for  j in range(B):
                        x_gen_array_now = x_gen_array[i*B+j,0,:,:,:]
                        sub  =  fids[j].split(".")[0]
                        new_image = nib.Nifti1Image(x_gen_array_now, img_affine) 
                        nib.save(new_image,save_dir + f"ep{pre_ep3}w{w}_{sub}_sample{i}_{term}-value{snp_value_index}.nii.gz")
                        print('saved image at ' + save_dir + f"ep{pre_ep3}w{w}_{sub}_sample{i}_{term}-value{snp_value_index}.nii.gz")
