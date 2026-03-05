#!/usr/bin/env python
# coding: utf-8
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import math
from abc import abstractmethod
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from t1f_gene_clip_adniall_interp_dataset import  MRIandGenedataset  as MRIandGenedatasetADNI
from t1f_gene_clip_ppmi_interp_dataset import  MRIandGenedataset  as MRIandGenedatasetPPMI
from t1f_gene_clip_ukb_interp_dataset import  MRIandGenedataset, GroupedBatchSampler
from options.train_options import TrainOptions
from torch.cuda.amp import autocast, GradScaler

from tensorboardX import SummaryWriter
from attention import SpatialTransformer5
from ScaleDense import ScaleDense_VAE3
from vit import MRIMambaMAE
from mamba_model import Mamba_dflow
import copy
import time
import itertools

def check_finite(name, x):
    if x is None: 
        return
    if not torch.isfinite(x).all():
        bad = x[~torch.isfinite(x)]
        print(f"[NaN/Inf] {name}: shape={tuple(x.shape)} dtype={x.dtype} "
              f"min={x.min().item()} max={x.max().item()} bad_sample={bad.flatten()[:5]}")
        raise RuntimeError(f"Non-finite detected in {name}")
    
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
    
def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
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
        sample_img_list = []
        for i in range(n_sample):
            sample_img = torch.randn((1, channels, image_size[0], image_size[1]), device=device) 
            sample_img = sample_img* t1_std + t1_mean
            sample_img = torch.where(sample_img <  min_tensor.repeat(sample_img.shape[0],1,1,1), min_tensor.repeat(sample_img.shape[0],1,1,1), sample_img)
            sample_img = torch.where(sample_img >  max_tensor.repeat(sample_img.shape[0],1,1,1), max_tensor.repeat(sample_img.shape[0],1,1,1), sample_img)
            # sample_img_mask = torch.triu(torch.ones_like(sample_img),diagonal=1).cuda()
            # sample_img_eye = torch.eye(sample_img.shape[-1]).unsqueeze(0).unsqueeze(0).cuda()
            # sample_img = sample_img*sample_img_mask
            # sample_img = sample_img + sample_img.permute([0,1,3,2]) + sample_img_eye
            sample_img = sample_img.repeat(batch_size,1,1,1)
            sample_img_list.append(sample_img)
        sample_img = torch.cat(sample_img_list,dim=0)
        
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
            noise_t_list = []
            for i in range(n_sample):
                noise_t = torch.randn((1, channels,  image_size[0], image_size[1]), device=device) 
                noise_t = noise_t * t1_std + t1_mean
                noise_t = torch.where(noise_t <  min_tensor.repeat(noise_t.shape[0],1,1,1), min_tensor.repeat(noise_t.shape[0],1,1,1), noise_t)
                noise_t = torch.where(noise_t >  max_tensor.repeat(noise_t.shape[0],1,1,1), max_tensor.repeat(noise_t.shape[0],1,1,1), noise_t)
                # noise_t_mask = torch.triu(torch.ones_like(noise_t),diagonal=1).cuda()
                # noise_t_eye = torch.eye(noise_t.shape[-1]).unsqueeze(0).unsqueeze(0).cuda()
                # noise_t = noise_t*noise_t_mask
                # noise_t = noise_t + noise_t.permute([0,1,3,2]) + noise_t_eye
                noise_t = noise_t.repeat(batch_size,1,1,1)
                noise_t_list.append(noise_t)
            noise_t = torch.cat(noise_t_list,dim=0)
            
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
    def train_losses(self, model, x_start, t, c, mask_c, t1_range,age_sex, snp=None, B1=None, B2=None):
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

        alpha_cumprod_t = self._extract(self.alphas_cumprod, t, x_noisy.shape)
        c_t = torch.sqrt(alpha_cumprod_t) + torch.sqrt(1. - alpha_cumprod_t)
        predicted_x0 = (c_t*x_noisy - torch.sqrt((1. - alpha_cumprod_t)) * predicted_noise) / torch.sqrt(alpha_cumprod_t)

        if B1 is not None and B2 is not None:     
            loss = F.mse_loss(noise[0:B1,:,:,:], predicted_noise[0:B1,:,:,:]) #* 10
            loss2 = F.mse_loss(noise[B1:B1+B2,:,:,:], predicted_noise[B1:B1+B2,:,:,:])
            loss3 = F.mse_loss(noise[B1+B2:,:,:,:], predicted_noise[B1+B2:,:,:,:])
            return loss, loss2, loss3, predicted_x0
        else:
            loss = F.mse_loss(noise, predicted_noise)
            return loss, predicted_x0

criterion = nn.CrossEntropyLoss().cuda()

opt = TrainOptions().parse()

# define model and diffusion
device = "cuda" if torch.cuda.is_available() else "cpu"
datasets_all = ["UKB","ADNI","PPMI"]
datasets_use = ["UKB","ADNI","PPMI"]

pretrain_dir = f"./generation_models/UKB_dcddim_genetranst1" 
pre_ep = 500

pretrain_dir2 = f"./generation_models/UKB-ADNI-PPMI_T1-GENE-CLIP_MAE_SNPMask_1000_v3_ff" #_Mask_{opt.mask_dropout}_{opt.mask_dropout2}
pre_ep2 = 100

ae_dir = f"./generation_models/UKB-ADNI-PPMI_MRI_VAE-32" #140
pre_ep3 = 200

save_dir = f"./generation_models/{'-'.join(datasets_use)}_dcddim_genetranst1" 
log_path =  f"./logs/log_{'-'.join(datasets_use)}_dcddim_genetranst1" 

# log_UKB-ADNI_cs400_iddim_genetranst1
os.system("mkdir -p {}".format(save_dir))
os.system("mkdir -p {}".format(log_path))

timesteps = 500
WIDTH = 32
TRAIN_BATCH_SIZE = 16
TEST_BATCH_SIZE = 16
WORKERS = TRAIN_BATCH_SIZE
PATCH = 512
NUM = 10*13*11#1200#1200 #
NUM2 = 6316#12622#12622#
lrate = 1e-4
epochs = 1000
p_uncound = 0.2
th = 1e-2

writer = SummaryWriter(logdir=log_path, comment='diffusion t1')

fold = 0
dataset_train_list = []
dataset_test_list = []

dataset_train_1 = MRIandGenedataset(fold=fold,phase="train")
sampler = GroupedBatchSampler(dataset_train_1, TRAIN_BATCH_SIZE)
data_loader_train_1 = torch.utils.data.DataLoader(dataset_train_1, batch_sampler=sampler,num_workers=WORKERS,pin_memory=True)
# data_loader_train_1 = torch.utils.data.DataLoader(dataset_train_1, batch_size=TRAIN_BATCH_SIZE, shuffle=True,
#                                                 num_workers=WORKERS,drop_last=True,pin_memory=True)
dataset_train_list.append(data_loader_train_1)
dataset_sizE = len(data_loader_train_1)
print("dataset_sizE: ", dataset_sizE)

dataset_test_1 = MRIandGenedataset(fold=fold,phase="test")
data_loader_test_1 = torch.utils.data.DataLoader(dataset_test_1, batch_size=TEST_BATCH_SIZE//2, shuffle=False,
                                                num_workers=WORKERS//2,drop_last=True,pin_memory=True)
dataset_test_list.append(data_loader_test_1)
dataset_size_test_1 = len(data_loader_test_1)
print("data_loader_test_1: ", dataset_size_test_1)

if "ADNI" in datasets_use:
    dataset_train_2 = MRIandGenedatasetADNI(fold=fold,phase="train",label=0)
    data_loader_train_2 = torch.utils.data.DataLoader(dataset_train_2, batch_size=TRAIN_BATCH_SIZE//4, shuffle=True,
                                                    num_workers=WORKERS//4)
    data_loader_train_2 = itertools.cycle(data_loader_train_2)
    dataset_train_3 = MRIandGenedatasetADNI(fold=fold,phase="train",label=1)
    data_loader_train_3 = torch.utils.data.DataLoader(dataset_train_3, batch_size=TRAIN_BATCH_SIZE//4, shuffle=True,
                                                    num_workers=WORKERS//4)
    data_loader_train_3 = itertools.cycle(data_loader_train_3)

    dataset_train_list.append(data_loader_train_2)
    dataset_train_list.append(data_loader_train_3)

    dataset_test_2 = MRIandGenedatasetADNI(fold=fold,phase="test",label=[0,1])
    data_loader_test_2 = torch.utils.data.DataLoader(dataset_test_2, batch_size=TEST_BATCH_SIZE//2, shuffle=False,
                                                    num_workers=WORKERS//2,drop_last=True)
    dataset_test_list.append(data_loader_test_2)
    dataset_size_test_2 = len(data_loader_test_2)
    print("data_loader_test_2: ", dataset_size_test_2)

if "PPMI" in datasets_use:
    dataset_train_4 = MRIandGenedatasetPPMI(fold=fold,phase="train",label=0)
    data_loader_train_4 = torch.utils.data.DataLoader(dataset_train_4, batch_size=TRAIN_BATCH_SIZE//4, shuffle=True,
                                                    num_workers=WORKERS//4)
    data_loader_train_4 = itertools.cycle(data_loader_train_4)
    dataset_train_5 = MRIandGenedatasetPPMI(fold=fold,phase="train",label=1)
    data_loader_train_5 = torch.utils.data.DataLoader(dataset_train_5, batch_size=TRAIN_BATCH_SIZE//4, shuffle=True,
                                                    num_workers=WORKERS//4)
    data_loader_train_5 = itertools.cycle(data_loader_train_5)

    dataset_train_list.append(data_loader_train_4)
    dataset_train_list.append(data_loader_train_5)

    dataset_test_3 = MRIandGenedatasetPPMI(fold=fold,phase="test",label=[0,1])
    data_loader_test_3 = torch.utils.data.DataLoader(dataset_test_3, batch_size=TEST_BATCH_SIZE//2, shuffle=False,
                                                    num_workers=WORKERS//2,drop_last=True)
    dataset_test_list.append(data_loader_test_3)
    dataset_size_test_3 = len(data_loader_test_3)
    print("data_loader_test_3: ", dataset_size_test_3)

VAE = ScaleDense_VAE3(nb_filter=32, latent_dim=32).cuda()
VAE = nn.DataParallel(VAE)
VAE.load_state_dict(torch.load(ae_dir+f"/fold_{fold}_E_{pre_ep3}.pth"))

gaussian_diffusion = GaussianDiffusion(timesteps=timesteps, beta_schedule='linear')
model = SpatialTransformer5(in_channels=32, depth=6, n_heads=4, d_head=256, n_classes=4,num_tokens=1200).cuda()
model = nn.DataParallel(model)
model.load_state_dict(torch.load(pretrain_dir+f"/fold_{fold}_ddim_{pre_ep}.pth"),strict=False)

opt.hidden_dropout_prob = 0.2
opt.emb_dropout_prob = 0.2
opt.classifier_dropout = 0
opt2 = copy.deepcopy(opt)
opt2.max_position_embeddings = 30000#

G = Mamba_dflow(opt).cuda()
E = MRIMambaMAE(opt).cuda()
model_clip = CLIP().cuda()
G = nn.DataParallel(G)
E = nn.DataParallel(E)

E.load_state_dict(torch.load(pretrain_dir2+f"/fold_{fold}_E_{pre_ep2}.pth")) #100
G.load_state_dict(torch.load(pretrain_dir2+f"/fold_{fold}_G_{pre_ep2}.pth")) #100
model_clip.load_state_dict(torch.load(pretrain_dir2+f"/fold_{fold}_clip_{pre_ep2}.pth")) #100

w_sl = torch.ones([NUM,1])
w_sl = w_sl.cuda()
w_sl.requires_grad = False
w_sl2 = torch.ones([NUM2,1])
w_sl2 = w_sl2.cuda()
w_sl2.requires_grad = False

for p in model.parameters():
    p.requires_grad = True
for p in E.parameters():
    p.requires_grad = False
for p in G.parameters():
    p.requires_grad = False
for p in model_clip.parameters():
    p.requires_grad = False

optim = torch.optim.AdamW([{'params': model.parameters()}], lr=lrate)
# optim = torch.optim.AdamW([{'params': model.parameters()}], lr=lrate)
scaler = GradScaler()

for ep in range(epochs):
    start_time = time.time()
    print(f'epoch {ep+1}')
    
    total_loss_clip = 0
    total_loss_clip2 = 0
    total_loss_clip3 = 0
    total_loss_dm = 0
    total_loss_dm2 = 0
    total_loss_dm3 = 0
    model.train()
    E.eval()
    G.eval()
    VAE.eval()
    model_clip.eval()
    for  train_data_ukb, train_data_adni0, train_data_adni1, train_data_ppmi0, train_data_ppmi1  in zip(*dataset_train_list):
        
        fid_1,x_1, c_1, t1_rangE, age_sex_1, snp_1, scores  = train_data_ukb
        fid_2,x_2, c_2, t1_range_2, age_sex_2, snp_2  = train_data_adni0 
        fid_3,x_3, c_3, t1_range_3, age_sex_3, snp_3  = train_data_adni1
        fid_4,x_4, c_4, t1_range_4, age_sex_4, snp_4   = train_data_ppmi0 
        fid_5,x_5, c_5, t1_range_5, age_sex_5, snp_5   = train_data_ppmi1  
        x = torch.cat([x_1,x_2,x_3,x_4,x_5],dim=0)
        t1_range = torch.cat([t1_rangE,t1_range_2,t1_range_3,t1_range_4,t1_range_5],dim=0)
        age_sex_1 = torch.flip(age_sex_1,dims=[0])
        age_sex = torch.cat([age_sex_1,age_sex_3,age_sex_2,age_sex_5,age_sex_4],dim=0)
        snp_1 = torch.flip(snp_1,dims=[0])
        snp = torch.cat([snp_1,snp_3,snp_2,snp_5,snp_4],dim=0)
        c_1 = torch.flip(c_1,dims=[0])
        c = torch.cat([c_1,c_3,c_2,c_5,c_4],dim=0)
        
        B = x.shape[0]
        B1 = x_1.shape[0]
        B2 = x_2.shape[0] + x_3.shape[0]
        B3 = x_4.shape[0] + x_5.shape[0]
        x = x.cuda()
        c = c.cuda()
        t1_range = t1_range.cuda()
        age_sex = age_sex.cuda()
        snp = snp.cuda()
       
        while 1:
            z_uncound = torch.rand(x.shape[0])
            batch_mask = (z_uncound>p_uncound).int().cuda()
            if torch.sum(batch_mask[0:B1]).item() != 0 and  torch.sum(batch_mask[B1:B1+B2]).item() != 0 and  torch.sum(batch_mask[B1+B2:]).item() != 0 :
                break
        # sample t uniformally for every example in the batch
        t = torch.randint(0, timesteps, (x.shape[0],)).long().cuda()

        with autocast(dtype=torch.bfloat16):
            feature = x.unsqueeze(1)
            
            with torch.cuda.amp.autocast(enabled=False):
                loss_dm, loss_dm2, loss_dm3, pre_x = gaussian_diffusion.train_losses(model, feature, t, c, batch_mask, t1_range, age_sex, snp,B1,B2)

            # check_finite('pre_x',pre_x)
            # check_finite('loss_dm',loss_dm)
            
            x_0 = VAE.module.decoder_conv(pre_x.squeeze(1).permute([0,2,1]).view([B,32,10,12,10]))
            x_1 =  VAE.module.up1(x_0)
            x_2 =  VAE.module.up2(x_1)
            x_3 =  VAE.module.up3(x_2)
            x_gen = VAE.module.out3(x_3)

            w_sl_use = torch.ones_like(w_sl).cuda() * torch.bernoulli(torch.full_like(w_sl,opt.mask_dropout)).unsqueeze(0).repeat(B,1,1) #opt.mask_dropout
            # w_sl_use = torch.ones_like(w_sl).cuda().unsqueeze(0).repeat(B,1,1)
            gate = F.gumbel_softmax(torch.log(torch.cat([w_sl_use,1.0-w_sl_use],dim=2)+1e-10)*10, tau = 1, dim = 2, hard = False)[:,:,0]
            
            feature = E(x_gen[0:B1,:,:,:,:])
            _,_, mri_feature = G(gate = gate[0:B1,:],inputs_embeds=feature)
            
            mask = torch.ones((B,B)).cuda()
            for i in range(B1,B1+B2):
                for j in range(B1,B1+B2):
                    if c[i] == c[j]:
                        mask[i-B1,j-B1] = 1
                    else:
                        mask[i-B1,j-B1] = 0
            for i in range(B1+B2,B):
                for j in range(B1+B2,B):
                    if c[i] == c[j]:
                        mask[i-B1,j-B1] = 1
                    else:
                        mask[i-B1,j-B1] = 0

            mask_index_s1= (batch_mask != 0)[0:B1]
            loss_clip,_,_ = model_clip.forward3(mri_feature[mask_index_s1,:], snp[0:B1,:][mask_index_s1,:],age_sex=age_sex[0:B1,:][mask_index_s1,:], mask = mask[0:B1,0:B1][mask_index_s1,mask_index_s1],group=False)
            mask_index_s2= (batch_mask != 0)[B1:B1+B2]
            loss_clip2,_,_ = model_clip.forward3(mri_feature[mask_index_s2,:], snp[B1:B1+B2,:][mask_index_s2,:],age_sex=age_sex[B1:B1+B2,:][mask_index_s2,:], mask = mask[B1:B1+B2,B1:B1+B2][mask_index_s2,mask_index_s2],group=True)
            mask_index_s3= (batch_mask != 0)[B1+B2:]
            loss_clip3,_,_ = model_clip.forward3(mri_feature[mask_index_s3,:], snp[B1+B2:,:][mask_index_s3,:],age_sex=age_sex[B1+B2:,:][mask_index_s3,:], mask = mask[B1+B2:,B1+B2:][mask_index_s3,mask_index_s3],group=True)

            loss = loss_dm + loss_dm2 + loss_dm3 + 0.01 * (loss_clip + loss_clip2 + loss_clip3)
                
        # optim.zero_grad()
        # loss.backward()
        # optim.step()

        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()
        total_loss_clip += loss_clip.item()   
        total_loss_clip2 += loss_clip2.item()   
        total_loss_clip3 += loss_clip3.item()   
        total_loss_dm += loss_dm.item()   
        total_loss_dm2 += loss_dm2.item()   
        total_loss_dm3 += loss_dm3.item()   
            
    print("loss_dm:", np.round(total_loss_dm/dataset_sizE,5),
          "loss_dm2:", np.round(total_loss_dm2/dataset_sizE,5),
          "loss_dm3:", np.round(total_loss_dm3/dataset_sizE,5),
          "loss_clip:", np.round(total_loss_clip/dataset_sizE,5),
          "loss_clip2:", np.round(total_loss_clip2/dataset_sizE,5),
          "loss_clip3:", np.round(total_loss_clip3/dataset_sizE,5))
    writer.add_scalars('loss_clip', {'loss_clip' + str(fold): total_loss_clip/dataset_sizE, }, ep+1)        
    writer.add_scalars('loss_clip2', {'loss_clip2' + str(fold): total_loss_clip2/dataset_sizE, }, ep+1)  
    writer.add_scalars('loss_clip3', {'loss_clip3' + str(fold): total_loss_clip3/dataset_sizE, }, ep+1)
    writer.add_scalars('loss_dm', {'loss_dm' + str(fold): total_loss_dm/dataset_sizE, }, ep+1)        
    writer.add_scalars('loss_dm2', {'loss_dm2' + str(fold): total_loss_dm2/dataset_sizE, }, ep+1) 
    writer.add_scalars('loss_dm3', {'loss_dm3' + str(fold): total_loss_dm3/dataset_sizE, }, ep+1)         
    
    model.eval()
    E.eval()
    G.eval()
    VAE.eval()
    model_clip.eval()
    with torch.no_grad():
        for dataset_index, dataset in enumerate(datasets_use):
            total_test_loss_clip = 0
            total_test_loss_dm = 0
            for  test_data in dataset_test_list[dataset_index]:
                if dataset == "UKB":
                    fid, x,  c, t1_range, age_sex, snp,_  = test_data
                else:
                    fid, x,  c, t1_range, age_sex, snp  = test_data
                
                B = x.shape[0]
                x = x.cuda()
                c = c.cuda()
                t1_range = t1_range.cuda()
                age_sex = age_sex.cuda()
                snp = snp.cuda()
                age_sex = torch.flip(age_sex,dims=[0])
                snp = torch.flip(snp,dims=[0])
                c = torch.flip(c,dims=[0])
                # random generate mask
                batch_mask = torch.ones(x.shape[0]).int().to(device)  
                
                # sample t uniformally for every example in the batch
                t = torch.randint(0, timesteps, (x.shape[0],)).long().cuda()

                feature = x.unsqueeze(1)
                loss_dm, pre_x = gaussian_diffusion.train_losses(model, feature, t, c, batch_mask, t1_range,age_sex, snp)

                x_0 = VAE.module.decoder_conv(pre_x.squeeze(1).permute([0,2,1]).view([B,32,10,12,10]))
                x_1 =  VAE.module.up1(x_0)
                x_2 =  VAE.module.up2(x_1)
                x_3 =  VAE.module.up3(x_2)
                x_gen = VAE.module.out3(x_3)

                w_sl_use = torch.ones_like(w_sl).cuda().unsqueeze(0).repeat(B,1,1)
                feature = E(x_gen)
                _,_, mri_feature = G(w_sl = w_sl_use,inputs_embeds=feature)
                if dataset == "UKB":
                    mask = torch.ones((B,B)).cuda()
                    loss_clip,_,_ = model_clip.forward3(mri_feature, snp,age_sex=age_sex,mask=mask,group=False)
                elif dataset == "ADNI":
                    mask = torch.zeros((B,B)).cuda()
                    for i in range(B):
                        for j in range(B):
                            if c[i] == c[j]:
                                mask[i,j] = 1
                    loss_clip,_,_ = model_clip.forward3(mri_feature, snp,age_sex=age_sex,mask=mask,group=True)
                elif dataset == "PPMI":
                    mask = torch.zeros((B,B)).cuda()
                    for i in range(B):
                        for j in range(B):
                            if c[i] == c[j]:
                                mask[i,j] = 1
                    loss_clip,_,_ = model_clip.forward3(mri_feature, snp,age_sex=age_sex,mask=mask,group=True)

                total_test_loss_clip += loss_clip.item()   
                total_test_loss_dm += loss_dm.item()   
                # optionally save model
            print(f"{dataset}_test_loss_dm:", np.round(total_test_loss_dm/len(dataset_test_list[dataset_index]),5),
                  f"{dataset}_test_loss_clip:", np.round(total_test_loss_clip/len(dataset_test_list[dataset_index]),5))                          
            writer.add_scalars(f'{dataset}_test_loss_dm', {'loss_dm' + str(fold): total_test_loss_dm/len(dataset_test_list[dataset_index]), }, ep+1)         
            writer.add_scalars(f'{dataset}_test_loss_clip', {'loss_clip' + str(fold): total_test_loss_clip/len(dataset_test_list[dataset_index]), }, ep+1)         
        
    end_time = time.time()
    print(f"time: {end_time-start_time}s")
    
    if ((ep+1) % 10) == 0:
        torch.save(model.state_dict(), save_dir + f"/fold_{fold}_ddim_{ep+1}.pth")
        print('saved model at ' + save_dir + f"/fold_{fold}_ddim_{ep+1}.pth")



