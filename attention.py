import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
import numpy as np

def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor

# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)

class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)

class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        '''
        generic one layer FC NN for embedding things  
        '''
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)

class CrossAttention(nn.Module): # Optimize this module as well
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0., cond_scale=1.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads
        self.cond_scale = cond_scale
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context) 
        v = self.to_v(context) 

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v) 
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)

class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, cond_scale=1., n_classes=2):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,cond_scale=cond_scale)  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout,cond_scale=cond_scale)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.timeembed = EmbedFC(1, dim)
        self.contextembed = EmbedFC(n_classes+1, dim)
        self.embed =  nn.Sequential(
            # nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim*2))
        
    def forward(self, x, context=None, t = None):
        if t  is not None:
            t_emed = self.timeembed(t)
            c_emed = self.contextembed(context)  
            emed = self.embed(c_emed+t_emed).unsqueeze(1)
        else:
            c_emed = self.contextembed(context)  
            emed = self.embed(c_emed).unsqueeze(1)
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), c_emed.unsqueeze(1)) + x
        alpha, beta = torch.chunk(emed, 2, dim=2)
        x = self.ff(self.norm3(x)* alpha + beta) + x
        
        return x

class FC_MAP(nn.Module):
    def __init__(self, input_dim, emb_dim, glu=False):
        super(FC_MAP, self).__init__()
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class FCEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    # Copied from transformers.models.bert.modeling_bert.BertEmbeddings.__init__
    def __init__(self, max_position_embeddings, hidden_size):
        super().__init__()
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.position_embeddings2 = nn.Embedding(max_position_embeddings, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.position_ids = torch.Tensor([i for i in range(max_position_embeddings)]).long().cuda()

    def forward(self, inputs_embeds=None,  position_ids=None):
        if position_ids == None:
            position_ids = self.position_ids
        position_embeddings = self.position_embeddings(position_ids)
        position_embeddings2 = self.position_embeddings2(position_ids)
        embeddings = position_embeddings * inputs_embeds + position_embeddings2
        embeddings = self.LayerNorm(embeddings)
        return embeddings

class T1Embeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    # Copied from transformers.models.bert.modeling_bert.BertEmbeddings.__init__
    def __init__(self, max_position_embeddings, hidden_size):
        super().__init__()
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.position_embeddings2 = nn.Embedding(max_position_embeddings, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        # self.position_ids = torch.Tensor([i for i in range(max_position_embeddings)]).long().cuda()
        self.register_buffer("position_ids", torch.arange(max_position_embeddings).expand((1, -1)))

    def forward(self, inputs_embeds=None,  position_ids=None):
        if position_ids == None:
            position_ids = self.position_ids
        position_embeddings = self.position_embeddings(position_ids)
        position_embeddings2 = self.position_embeddings2(position_ids)
        embeddings = position_embeddings * inputs_embeds + position_embeddings2
        embeddings = self.LayerNorm(embeddings)
        return embeddings

class SNPEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    # Copied from transformers.models.bert.modeling_bert.BertEmbeddings.__init__
    def __init__(self, max_position_embeddings, hidden_size):
        super().__init__()
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.position_embeddings2 = nn.Embedding(max_position_embeddings, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        # self.position_ids = torch.Tensor([i for i in range(max_position_embeddings)]).long().cuda()
        self.register_buffer("position_ids", torch.arange(max_position_embeddings).expand((1, -1)))

    def forward(self, inputs_embeds=None,  position_ids=None):
        if position_ids == None:
            position_ids = self.position_ids
        position_embeddings = self.position_embeddings(position_ids)
        position_embeddings2 = self.position_embeddings2(position_ids)
        embeddings = position_embeddings * inputs_embeds + position_embeddings2
        embeddings = self.LayerNorm(embeddings)
        return embeddings

class SpatialTransformer_D(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None, cond_scale=1.,n_classes=2,out_channels=1):
        super().__init__()
        self.n_classes = n_classes
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.emb = FCEmbeddings(in_channels,inner_dim)
        self.proj_in  = nn.Sequential(
            nn.Linear(in_channels, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, inner_dim),
        )
     
        # self.proj_in = FC_MAP(in_channels, inner_dim)
        
        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim,cond_scale=cond_scale,n_classes=n_classes)
                for d in range(depth)]
        )

        self.proj_out  = nn.Sequential(
            nn.LayerNorm(inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, out_channels),
            nn.Sigmoid(),
        )
        self.pooling = nn.AdaptiveMaxPool2d((1,inner_dim))
        
        # self.proj_out = FC_MAP(inner_dim, in_channels)

    def forward(self, input, t=None, context=None, context_mask=None, age_sex=None):
        # note: if no context is given, cross-attention defaults to self-attention
        # context = F.one_hot(context, num_classes=self.n_classes).type(torch.float)
        # context = context.unsqueeze(1)
        # context = torch.cat([1.0-context,context], dim=1)

        # 创建映射字典
        if self.n_classes == 2:
            mapping = {0: [1, 0], 1: [0, 1], 3: [0, 0]}
        elif self.n_classes == 4:
            mapping = {0: [1, 0, 0, 0], 1: [0, 1, 0, 0], 2: [0, 0, 1, 0], 3: [0, 0, 0, 1], 4:[0,0,0,0]}
        # 将tensor转换为numpy array
        numpy_array = context.cpu().detach().numpy()
        # 使用列表解析来转换每一个元素
        encoded_list = [mapping[i] for i in numpy_array]
        # 将编码列表转换回tensor
        context = torch.tensor(encoded_list).cuda().type(torch.float)
        context = torch.cat([context,age_sex],dim=1)
        if context_mask is not None:
            context_mask = context_mask[:, None]
            context_mask = context_mask.repeat(1,self.n_classes+2)
            context = context * context_mask 
        x = self.proj_in(input.squeeze(1))     
        x = self.emb(x)
        # b,c,w,h = x.shape
        # x = x.view(b,c,-1).permute([0,2,1])
        for block in self.transformer_blocks:
            x = block(x, context=context,t=t)
        # x = x.permute([0,2,1]).view(b,c,w,h)
        x = self.pooling(x).squeeze(1)
        x = self.proj_out(x)
        return x

class SpatialTransformer_G(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None, cond_scale=1.,n_classes=2):
        super().__init__()
        self.n_classes = n_classes
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.emb = FCEmbeddings(in_channels,inner_dim)
        self.pre_layer = nn.Linear(128,200*200)
        self.proj_in  = nn.Sequential(
            nn.Linear(in_channels, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, inner_dim),
        )
     
        # self.proj_in = FC_MAP(in_channels, inner_dim)
        
        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim,cond_scale=cond_scale,n_classes=n_classes)
                for d in range(depth)]
        )

        self.proj_out  = nn.Sequential(
            nn.LayerNorm(inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, in_channels),
            nn.Tanh()
        )
        
        # self.proj_out = FC_MAP(inner_dim, in_channels)

    def forward(self, input, t=None, context=None, context_mask=None, age_sex=None):
        # note: if no context is given, cross-attention defaults to self-attention
        # context = F.one_hot(context, num_classes=self.n_classes).type(torch.float)
        # context = context.unsqueeze(1)
        # context = torch.cat([1.0-context,context], dim=1)

        # 创建映射字典
        if self.n_classes == 2:
            mapping = {0: [1, 0], 1: [0, 1], 3: [0, 0]}
        elif self.n_classes == 4:
            mapping = {0: [1, 0, 0, 0], 1: [0, 1, 0, 0], 2: [0, 0, 1, 0], 3: [0, 0, 0, 1], 4:[0,0,0,0]}
        # 将tensor转换为numpy array
        numpy_array = context.cpu().detach().numpy()
        # 使用列表解析来转换每一个元素
        encoded_list = [mapping[i] for i in numpy_array]
        # 将编码列表转换回tensor
        context = torch.tensor(encoded_list).cuda().type(torch.float)
        context = torch.cat([context,age_sex],dim=1)
        if context_mask is not None:
            context_mask = context_mask[:, None]
            context_mask = context_mask.repeat(1,self.n_classes+2)
            context = context * context_mask 
            
        input = self.pre_layer(input).view(input.shape[0],1,200,200)
        x = self.proj_in(input.squeeze(1))     
        x = self.emb(x)
        # b,c,w,h = x.shape
        # x = x.view(b,c,-1).permute([0,2,1])
        for block in self.transformer_blocks:
            x = block(x, context=context,t=t)
        # x = x.permute([0,2,1]).view(b,c,w,h)
        x = self.proj_out(x).unsqueeze(1) #+ input
        return x

class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None, cond_scale=1., n_classes=2):
        super().__init__()
        self.n_classes = n_classes
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.emb = FCEmbeddings(in_channels,inner_dim)
        self.proj_in  = nn.Sequential(
            nn.Linear(in_channels, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, inner_dim),
        )
     
        # self.proj_in = FC_MAP(in_channels, inner_dim)
        
        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim,cond_scale=cond_scale,n_classes=n_classes)
                for d in range(depth)]
        )

        self.proj_out  = nn.Sequential(
            nn.LayerNorm(inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, in_channels),
            # nn.Tanh(),
        )
        
        # self.proj_out = FC_MAP(inner_dim, in_channels)

    def forward(self, input, t=None, context=None, context_mask=None, age_sex=None):
        # note: if no context is given, cross-attention defaults to self-attention
        # context = F.one_hot(context, num_classes=self.n_classes).type(torch.float)
        # context = context.unsqueeze(1)
        # context = torch.cat([1.0-context,context], dim=1)

        # 创建映射字典
        if self.n_classes == 2:
            mapping = {0: [1, 0], 1: [0, 1], 3: [0, 0]}
        elif self.n_classes == 4:
            mapping = {0: [1, 0, 0, 0], 1: [0, 1, 0, 0], 2: [0, 0, 1, 0], 3: [0, 0, 0, 1], 4:[0,0,0,0]}
        # 将tensor转换为numpy array
        numpy_array = context.cpu().detach().numpy()
        # 使用列表解析来转换每一个元素
        encoded_list = [mapping[i] for i in numpy_array]
        # 将编码列表转换回tensor
        context = torch.tensor(encoded_list).cuda().type(torch.float)
        context = torch.cat([context,age_sex[:,0:1]],dim=1)
        
        if context_mask is not None:
            context_mask = context_mask[:, None]
            context_mask = context_mask.repeat(1,self.n_classes+1)
            # context_mask = torch.zeros_like(context_mask)
            context = context * context_mask 
        x = self.proj_in(input.squeeze(1))     
        x = self.emb(x)
        # b,c,w,h = x.shape
        # x = x.view(b,c,-1).permute([0,2,1])
        for block in self.transformer_blocks:
            x = block(x, context=context,t=t)
        # x = x.permute([0,2,1]).view(b,c,w,h)
        x = self.proj_out(x).unsqueeze(1) + input
        return x

class BasicTransformerBlock2(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, cond_scale=1., n_classes=2):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,cond_scale=cond_scale)  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout,cond_scale=cond_scale)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.timeembed = EmbedFC(1, dim)
        self.contextembed = nn.Sequential(
                nn.Conv1d(230, 230//2, 1, bias=True),
                nn.Conv1d(230//2, 1, 1, bias=True)
            ) 
        self.embed =  nn.Sequential(
            # nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim*2))
        
    def forward(self, x, snp=None,  age_sex=None, t = None):
        context = torch.cat([snp,age_sex.unsqueeze(1)],dim=1)
        if t  is not None:
            t_emed = self.timeembed(t).unsqueeze(1)
            c_emed = self.contextembed(context)  
            emed = self.embed(c_emed+t_emed)
        else:
            c_emed = self.contextembed(context)  
            emed = self.embed(c_emed)
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context) + x
        alpha, beta = torch.chunk(emed, 2, dim=2)
        x = self.ff(self.norm3(x)* alpha + beta) + x
        
        return x

class SpatialTransformer2(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None, cond_scale=1., n_classes=2):
        super().__init__()
        self.n_classes = n_classes
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.emb = FCEmbeddings(in_channels,inner_dim)
        self.snp_emb = FCEmbeddings(229,inner_dim)
        self.proj_in  = nn.Sequential(
            nn.Linear(in_channels, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, inner_dim),
        )
        self.snp_proj_in  = nn.Sequential(
            nn.Linear(3, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, inner_dim),
        )
        self.age_sex_proj_in  = nn.Sequential(
            nn.Linear(2, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, inner_dim),
        )
        # self.proj_in = FC_MAP(in_channels, inner_dim)
        
        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock2(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim,cond_scale=cond_scale,n_classes=n_classes)
                for d in range(depth)]
        )

        self.proj_out  = nn.Sequential(
            nn.LayerNorm(inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, in_channels),
            # nn.Tanh(),
        )
        
        # self.proj_out = FC_MAP(inner_dim, in_channels)

    def forward(self, input, t=None, context=None, context_mask=None, age_sex=None, snp=None):
       
        age_sex = self.age_sex_proj_in(age_sex)
        if context_mask is not None:        
            # snp  = snp *  torch.zeros_like(context_mask[:, None, None].repeat(1, snp.shape[1], snp.shape[2])).cuda()
            snp  = snp * context_mask[:, None, None].repeat(1, snp.shape[1], snp.shape[2])
            age_sex = age_sex * context_mask[:, None].repeat(1, age_sex.shape[1])
            
        snp = self.snp_proj_in(snp)
        snp = self.snp_emb(snp)
        
        x = self.proj_in(input.squeeze(1))     
        x = self.emb(x)
        # b,c,w,h = x.shape
        # x = x.view(b,c,-1).permute([0,2,1])
        for block in self.transformer_blocks:
            x = block(x, snp=snp, age_sex=age_sex, t=t)
        # x = x.permute([0,2,1]).view(b,c,w,h)
        x = self.proj_out(x).unsqueeze(1) + input
        return x

class BasicTransformerBlock3(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, cond_scale=1., n_classes=2):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,cond_scale=cond_scale)  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout,cond_scale=cond_scale)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.timeembed = EmbedFC(1, dim)
        self.contextembed = nn.Sequential(
                nn.Conv1d(2, 2//2, 1, bias=True),
                nn.Conv1d(2//2, 1, 1, bias=True)
            ) 
        self.embed =  nn.Sequential(
            # nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim*2))
        
    def forward(self, x, snp=None,  age_sex=None, t = None, mask=None):
        context = torch.cat([snp.unsqueeze(1),age_sex.unsqueeze(1)],dim=1)
        if t  is not None:
            t_emed = self.timeembed(t).unsqueeze(1)
            c_emed = self.contextembed(context)  
            emed = self.embed(c_emed+t_emed)
        else:
            c_emed = self.contextembed(context)  
            emed = self.embed(c_emed)
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context) + x
        alpha, beta = torch.chunk(emed, 2, dim=2)
        x = self.ff(self.norm3(x)* alpha + beta) + x
        return x

class SpatialTransformer3(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None, cond_scale=1., n_classes=2,num_tokens=1200):
        super().__init__()
        self.n_classes = n_classes
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.emb = T1Embeddings(num_tokens,inner_dim)
        self.snp_emb = SNPEmbeddings(num_tokens,inner_dim)
        self.proj_in  = nn.Sequential(
            nn.Linear(in_channels, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, inner_dim),
        )
        self.snp_proj_in  = nn.Sequential(
            nn.Linear(128, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, inner_dim),
        )
        self.age_sex_proj_in  = nn.Sequential(
            nn.Linear(2, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, inner_dim),
        )
        # self.proj_in = FC_MAP(in_channels, inner_dim)
        
        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock3(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim,cond_scale=cond_scale,n_classes=n_classes)
                for d in range(depth)]
        )

        self.proj_out  = nn.Sequential(
            nn.LayerNorm(inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, in_channels),
            # nn.Tanh(),
        )
        
        # self.proj_out = FC_MAP(inner_dim, in_channels)

    def forward(self, input, t=None, context=None, context_mask=None, age_sex=None, snp=None):
        age_sex = self.age_sex_proj_in(age_sex)
        snp = self.snp_proj_in(snp)
        # snp = self.snp_emb(snp)
        
        if context_mask is not None: 
            snp  = snp * context_mask[:, None].repeat(1, snp.shape[1])       
            # snp  = snp *  context_mask[:, None, None].repeat(1, snp.shape[1], snp.shape[2])
            # age_sex  = age_sex *  torch.zeros_like(context_mask[:, None].repeat(1, age_sex.shape[1])).cuda()
            age_sex = age_sex * context_mask[torch.randperm(context_mask.size(0))][:, None].repeat(1, age_sex.shape[1])
            
        x = self.proj_in(input.squeeze(1))     
        x = self.emb(x)
        # b,c,w,h = x.shape
        # x = x.view(b,c,-1).permute([0,2,1])

        for block in self.transformer_blocks:
            x = block(x, snp=snp, age_sex=age_sex, t=t)
        # x = x.permute([0,2,1]).view(b,c,w,h)
        x = self.proj_out(x).unsqueeze(1) + input
        return x

class BasicTransformerBlock4(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, cond_scale=1., n_classes=2):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,cond_scale=cond_scale)  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout,cond_scale=cond_scale)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.timeembed = EmbedFC(1, dim)
        self.conembed = nn.Sequential(
                nn.Conv1d(2, 2//2, 1, bias=True),
                nn.GELU(),
                nn.Conv1d(2//2, 1, 1, bias=True)
            ) 
        self.embed =  nn.Sequential(
            # nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim*2))
       
    def forward(self, x, snp=None,  age_sex=None, t = None, mask=None):
        if snp is not None:
            context = torch.cat([snp.unsqueeze(1),age_sex.unsqueeze(1)],dim=1)
    
            c_emed = self.conembed(context)  
            t_emed = self.timeembed(t).unsqueeze(1)
            emed = self.embed(c_emed+t_emed)

            x = self.attn1(self.norm1(torch.cat([t_emed,context,x],dim=1)))[:,3:,:] + x
            # x = self.attn2(self.norm2(x), context) + x

            alpha, beta = torch.chunk(emed, 2, dim=2)

            x = self.ff(self.norm3(x)* alpha + beta) + x

        else:
            c_emed = age_sex.unsqueeze(1)
            
            t_emed = self.timeembed(t).unsqueeze(1)
            emed = self.embed(c_emed+t_emed)
            
            x = self.attn1(self.norm1(x)) + x
            # x = self.attn2(self.norm2(x), c_emed) + x

            alpha, beta = torch.chunk(emed, 2, dim=2)

            x = self.ff(self.norm3(x)* alpha + beta) + x

        return x

    # def forward(self, x, snp=None,  age_sex=None, t = None, mask=None):
    #     if snp is not None:
    #         context = torch.cat([snp.unsqueeze(1),age_sex.unsqueeze(1)],dim=1)
    #         c_emed = self.contextembed(context)  
    #         c_emed_mask = self.contextembed(torch.zeros_like(context).cuda())  
            
    #         t_emed = self.timeembed(t).unsqueeze(1)
    #         emed = self.embed(c_emed+t_emed)
    #         emed_mask = self.embed(c_emed_mask+t_emed)

    #         x = self.attn1(self.norm1(x)) + x
    #         x = self.attn2(self.norm2(x), context) + x
    #         x_mask = self.attn2(self.norm2(x), torch.zeros_like(context).cuda()) + x

    #         alpha, beta = torch.chunk(emed, 2, dim=2)
    #         alpha_mask, beta_mask = torch.chunk(emed_mask, 2, dim=2)

    #         x = self.ff(self.norm3(x)* alpha + beta) + x
    #         x_mask = self.ff(self.norm3(x_mask)* alpha_mask + beta_mask) + x_mask
    #     else:
    #         c_emed = age_sex.unsqueeze(1)
    #         c_emed_mask = torch.zeros_like(c_emed).cuda()
            
    #         t_emed = self.timeembed(t).unsqueeze(1)
    #         emed = self.embed(c_emed+t_emed)
    #         emed_mask = self.embed(c_emed_mask+t_emed)
            
    #         x = self.attn1(self.norm1(x)) + x

    #         alpha, beta = torch.chunk(emed, 2, dim=2)
    #         alpha_mask, beta_mask = torch.chunk(emed_mask, 2, dim=2)

    #         x = self.ff(self.norm3(x)* alpha + beta) + x
    #         x_mask = self.ff(self.norm3(x)* alpha_mask + beta_mask) + x
            
    #     mask = mask.view(mask.shape[0],1,1)
    #     x = mask * x + (1.0-mask) * x_mask
    #     return x

class SpatialTransformer4(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None, cond_scale=1., n_classes=2,num_tokens=1200,mask=None):
        super().__init__()
        self.n_classes = n_classes
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.emb = T1Embeddings(num_tokens,inner_dim)
        self.snp_emb = SNPEmbeddings(num_tokens,inner_dim)
        self.proj_in  = nn.Sequential(
            nn.Linear(in_channels, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, inner_dim),
        )
        self.gene_proj_in  = nn.Sequential(
            nn.Linear(512, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, inner_dim),
        )
        self.age_sex_proj_in  = nn.Sequential(
            nn.Linear(2, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, inner_dim),
        )

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock4(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim,cond_scale=cond_scale,n_classes=n_classes)
                for d in range(depth)]
        )

        self.proj_out  = nn.Sequential(
            nn.LayerNorm(inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, in_channels),
            # nn.Tanh(),
        )
        self.mask = mask

    def forward(self, input, t=None, context=None, context_mask=None, age_sex=None, snp=None):
        age_sex = self.age_sex_proj_in(age_sex)
        
        if snp is not None:
            snp = self.gene_proj_in(snp)

        if context_mask is not None: 
            age_sex = age_sex *context_mask[:, None].repeat(1, age_sex.shape[1])
            if snp is not None:
                # snp  = snp * context_mask[torch.randperm(context_mask.size(0))][:, None].repeat(1, snp.shape[1])       
                snp  = snp * context_mask[:, None].repeat(1, snp.shape[1])      
                # snp  = snp * context_mask[:, None, None].repeat(1, snp.shape[1], snp.shape[2])

        x = self.proj_in(input.squeeze(1))     
        x = self.emb(x)

        for block in self.transformer_blocks:
            x = block(x, snp=snp, age_sex=age_sex, t=t)

        x = self.proj_out(x).unsqueeze(1) + input
        return x

class BasicTransformerBlock5(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, cond_scale=1., n_classes=2):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,cond_scale=cond_scale)  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout,cond_scale=cond_scale)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.timeembed = EmbedFC(1, dim)
        self.conembed = nn.Sequential(
                nn.Conv1d(2, 2//2, 1, bias=True),
                nn.GELU(),
                nn.Conv1d(2//2, 1, 1, bias=True)
            ) 
        self.embed =  nn.Sequential(
            # nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim*2))
       
    def forward(self, x, snp=None,  age_sex=None, t = None, mask=None):
        if snp is not None:
            context = torch.cat([snp.unsqueeze(1),age_sex.unsqueeze(1)],dim=1)
    
            c_emed = self.conembed(context)  
            t_emed = self.timeembed(t).unsqueeze(1)
            emed = self.embed(c_emed+t_emed)

            x = self.attn1(self.norm1(torch.cat([t_emed,context,x],dim=1)))[:,3:,:] + x
            # x = self.attn2(self.norm2(x), context) + x

            alpha, beta = torch.chunk(emed, 2, dim=2)

            x = self.ff(self.norm3(x)* alpha + beta) + x

        else:
            c_emed = age_sex.unsqueeze(1)
            
            t_emed = self.timeembed(t).unsqueeze(1)
            emed = self.embed(c_emed+t_emed)
            
        
            x = self.attn1(self.norm1(x)) + x
            # x = self.attn2(self.norm2(x), c_emed) + x

            alpha, beta = torch.chunk(emed, 2, dim=2)

            x = self.ff(self.norm3(x)* alpha + beta) + x

        return x

class SpatialTransformer5(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None, cond_scale=1., n_classes=2,num_tokens=1200,mask=None):
        super().__init__()
        self.n_classes = n_classes
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.emb = T1Embeddings(num_tokens,inner_dim)
        self.snp_emb = SNPEmbeddings(num_tokens,inner_dim)
        self.proj_in  = nn.Sequential(
            nn.Linear(in_channels*2, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, inner_dim),
        )
        self.gene_proj_in  = nn.Sequential(
            nn.Linear(512, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, inner_dim),
        )
        self.age_sex_proj_in  = nn.Sequential(
            nn.Linear(2, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, inner_dim),
        )

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock5(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim,cond_scale=cond_scale,n_classes=n_classes)
                for d in range(depth)]
        )

        self.proj_out  = nn.Sequential(
            nn.LayerNorm(inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, in_channels),
            # nn.Tanh(),
        )
        self.mask = mask

    def forward(self, input, t=None, context=None, context_mask=None, age_sex=None, snp=None, template=None):
        age_sex = self.age_sex_proj_in(age_sex)
        
        if snp is not None:
            snp = self.gene_proj_in(snp)

        if context_mask is not None: 
            age_sex = age_sex *context_mask[:, None].repeat(1, age_sex.shape[1])
            if snp is not None:
                # snp  = snp * context_mask[torch.randperm(context_mask.size(0))][:, None].repeat(1, snp.shape[1])       
                snp  = snp * context_mask[:, None].repeat(1, snp.shape[1])      
                # snp  = snp * context_mask[:, None, None].repeat(1, snp.shape[1], snp.shape[2])

        x = self.proj_in(torch.cat([input,template],dim=-1).squeeze(1))     
        x = self.emb(x)
        
        for block in self.transformer_blocks:
            x = block(x, snp=snp, age_sex=age_sex, t=t)

        x = self.proj_out(x).unsqueeze(1) + input
        return x

class SpatialTransformer6(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None, cond_scale=1., n_classes=2,num_tokens=1200,mask=None):
        super().__init__()
        self.n_classes = n_classes
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.emb = T1Embeddings(num_tokens,inner_dim)
        self.snp_emb = SNPEmbeddings(num_tokens,inner_dim)
        self.proj_in  = nn.Sequential(
            nn.Linear(in_channels*2, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, inner_dim),
        )
        self.gene_proj_in  = nn.Sequential(
            nn.Linear(512, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, inner_dim),
        )
        self.age_sex_proj_in  = nn.Sequential(
            nn.Linear(2, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, inner_dim),
        )

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock5(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim,cond_scale=cond_scale,n_classes=n_classes)
                for d in range(depth)]
        )

        self.proj_out  = nn.Sequential(
            nn.LayerNorm(inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, in_channels),
            # nn.Tanh(),
        )
        self.mask = mask

    def forward(self, input, t=None, context=None, context_mask=None, age_sex=None, snp=None, template=None):
        age_sex = self.age_sex_proj_in(age_sex)
        
        if snp is not None:
            snp = self.gene_proj_in(snp)

        if context_mask is not None: 
            age_sex = age_sex *context_mask[:, None].repeat(1, age_sex.shape[1])
            if snp is not None:
                # snp  = snp * context_mask[torch.randperm(context_mask.size(0))][:, None].repeat(1, snp.shape[1])       
                snp  = snp * context_mask[:, None].repeat(1, snp.shape[1])      
                # snp  = snp * context_mask[:, None, None].repeat(1, snp.shape[1], snp.shape[2])

        x = self.proj_in(torch.cat([input,input],dim=-1).squeeze(1))     
        x = self.emb(x)
        
        for block in self.transformer_blocks:
            x = block(x, snp=snp, age_sex=age_sex, t=t)

        x = self.proj_out(x).unsqueeze(1) + input
        return x

# if __name__ == "__main__":
#     model = SpatialTransformer4(in_channels=1024, n_heads=4, d_head=256).cuda()
#     snp = torch.ones((4,128)).cuda()
#     x = torch.ones((4,1,1200,1024)).cuda()
#     t = torch.tensor([0.2,0.4,0.6,0.8]).cuda()
#     content = torch.tensor([0,1,1,0]).cuda()
#     context_mask = torch.tensor([0,0,1.0,1.0]).cuda()
#     age_sex = torch.tensor([[0.5,0],[0.5,1],[0.8,0],[0.8,1]]).cuda()
#     # y = model(x,content,t,context_mask)
#     y = model(x,context=content,age_sex=age_sex,snp=snp)
#     print("finished")