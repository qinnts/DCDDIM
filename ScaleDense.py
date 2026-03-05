import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformer import  TransformerEncoder
from packaging import version
from bigbird_model import BigBird3
import copy
from torch.nn.utils import spectral_norm

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num/1e6, 'Trainable': trainable_num /1e6}

class SE_block(nn.Module):
    def __init__(self, inchannels, reduction = 16 ):
        super(SE_block,self).__init__()
        self.GAP = nn.AdaptiveAvgPool3d((1,1,1))
        self.FC1 = nn.Linear(inchannels,inchannels//reduction)
        self.FC2 = nn.Linear(inchannels//reduction,inchannels)

    def forward(self,x):
        model_input = x
        x = self.GAP(x)
        x = torch.reshape(x,(x.size(0),-1))
        x = self.FC1(x)
        x = nn.ReLU()(x)
        x = self.FC2(x)
        x = nn.Sigmoid()(x)
        x = x.view(x.size(0),x.size(1),1,1,1)
        return model_input * x

class AC_layer(nn.Module):
    def __init__(self,inchannels, outchannels):
        super(AC_layer,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(inchannels,outchannels,(3,3,3),stride=1,padding=1,bias=False),
            nn.InstanceNorm3d(outchannels))
        self.conv2 = nn.Sequential(
            nn.Conv3d(inchannels,outchannels,(1,1,3),stride=1,padding=(0,0,1),bias=False),
            nn.InstanceNorm3d(outchannels))
        self.conv3 = nn.Sequential(
            nn.Conv3d(inchannels,outchannels,(3,1,1),stride=1,padding=(1,0,0),bias=False),
            nn.InstanceNorm3d(outchannels))
        self.conv4 = nn.Sequential(
            nn.Conv3d(inchannels,outchannels,(1,3,1),stride=1,padding=(0,1,0),bias=False),
            nn.InstanceNorm3d(outchannels))
    def forward(self,x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        return x1 + x2 + x3 + x4

class dense_layer(nn.Module):
    def __init__(self,inchannels,outchannels):

        super(dense_layer,self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(inchannels,outchannels,(3,3,3),stride=1,padding=1,bias=False),
            AC_layer(inchannels,outchannels),
            nn.InstanceNorm3d(outchannels),
            nn.ELU(),
            nn.Conv3d(outchannels,outchannels,(3,3,3),stride=1,padding=1,bias=False),
            AC_layer(outchannels,outchannels),       
            nn.InstanceNorm3d(outchannels),
            nn.ELU(),
            SE_block(outchannels),
            nn.MaxPool3d(2,2),
        )
        self.block2 = nn.Sequential(
            nn.Conv3d(inchannels,outchannels,(1,1,1),stride=1,padding=0,bias=False),
            nn.InstanceNorm3d(outchannels),
            nn.ELU(),
            nn.MaxPool3d(2,2),
        )
        #self.drop = nn.Dropout3d(0.1)

    def forward(self,x):
        #x = self.drop(x)
        new_features = self.block(x)
        x = F.max_pool3d(x,2)
        x = torch.cat([new_features,x], 1)
        #x = self.block(new_features) + self.block2(x)
        return x

class dense_layer2(nn.Module):
    def __init__(self,inchannels,outchannels):
        super(dense_layer2,self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(inchannels,outchannels,(3,3,3),stride=1,padding=1,bias=False),
            nn.InstanceNorm3d(outchannels),
            nn.ELU(),
            nn.Conv3d(outchannels,outchannels,(3,3,3),stride=1,padding=1,bias=False),     
            nn.InstanceNorm3d(outchannels),
            nn.ELU(),
            nn.MaxPool3d(2,2),
        )
    #(83,104,79) (41,52,39) (20,26,19) (10,13,9)
    #(88,105,85) (44,52,42) (22,26,21) (11,13,10)
    #(85,100,85) (42,50,42) (21,25,21) (10,12,10)
    #(80,100,85) (40,50,42) (20,25,21) (10,12,10)
    #(84,101,87) (42,50,43) (21,25,21) (10,12,10)
    # (80,100,83)(40,50,41) (20,25,20) (10,12,10)
    def forward(self,x):
        #print(x.shape)
        new_features = self.block(x) # (32,47,34)  (18,23,17) (9 11 8)
        x = F.max_pool3d(x,2)  #(42,52,39) (21,26,19) (10 13 9)
        x = torch.cat([new_features,x], 1)
        return x
    
class dense_layer2_2(nn.Module):
    def __init__(self,inchannels,outchannels):
        super(dense_layer2_2,self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(inchannels,outchannels,(3,3,3),stride=1,padding=2, dilation=2, bias=False),
            nn.InstanceNorm3d(outchannels),
            nn.ELU(),
            nn.Conv3d(outchannels,outchannels,(3,3,3),stride=1,padding=1,bias=False),   
            nn.InstanceNorm3d(outchannels),
            nn.ELU(),
        )

    def forward(self,x):
        new_features = self.block(x)
        x = torch.cat([new_features,x], 1)
        return x
 
class dense_layer3(nn.Module):
    def __init__(self,inchannels,outchannels):
        super(dense_layer3,self).__init__()
        self.block = nn.Sequential(
            spectral_norm(nn.Conv3d(inchannels,outchannels,(3,3,3),stride=1,padding=1,bias=False)),
            # nn.InstanceNorm3d(outchannels),
            nn.ELU(),
            # nn.Conv3d(outchannels,outchannels,(3,3,3),stride=1,padding=1,bias=False),     
            # nn.InstanceNorm3d(outchannels),
            # nn.ELU(),
            nn.MaxPool3d(2,2),
        )
    #(83,104,79) (41,52,39) (20,26,19) (10,13,9)
    #(88,105,85) (44,52,42) (22,26,21) (11,13,10)
    #(85,100,85) (42,50,42) (21,25,21) (10,12,10)
    #(80,100,85) (40,50,42) (20,25,21) (10,12,10)
    #(84,101,87) (42,50,43) (21,25,21) (10,12,10)
    # (80,100,83)(40,50,41) (20,25,20) (10,12,10)
    def forward(self,x):
        #print(x.shape)
        new_features = self.block(x) # (32,47,34)  (18,23,17) (9 11 8)
        x = F.max_pool3d(x,2)  #(42,52,39) (21,26,19) (10 13 9)
        x = torch.cat([new_features,x], 1)
        return x
       
class up_layer(nn.Module):
    def __init__(self,inchannels,outchannels):
        super(up_layer,self).__init__()
        self.pool = nn.UpsamplingNearest2d(scale_factor=2)
        self.block = nn.Sequential(
            self.pool,
            nn.Conv3d(inchannels,outchannels,(3,3,3),stride=1,padding=1, bias=False),
            nn.InstanceNorm3d(outchannels),
            nn.ELU(),
            nn.Conv3d(outchannels,outchannels,(3,3,3),stride=1,padding=1,bias=False),   
            nn.InstanceNorm3d(outchannels),
            nn.ELU(),
        )
        self.bypass = nn.Sequential(
            self.pool,
            nn.Conv3d(inchannels,outchannels,(1,1,1),stride=1,padding=0, bias=False),
            nn.InstanceNorm3d(outchannels),
            nn.ELU()
        )

    def forward(self,x):
        x = self.block(x) + self.bypass(x)
        return x
    
class ScaleDense(nn.Module):
    def __init__(self,nb_filter=16, nb_block=4, use_gender=False):
        '''
        Develop Scale Dense for brain age estimation

        Args:
            nb_filter (int): number of initial convolutional layer filter. Default: 8
            nb_block (int): number of Dense block. Default: 5
            use_gender (bool, optional): if use gender input. Default: True
        '''
        super(ScaleDense,self).__init__()
        self.nb_block = nb_block
        self.use_gender = use_gender
        self.pre = nn.Sequential(
            nn.Conv3d(1,nb_filter,kernel_size=7,stride=1
                     ,padding=1,dilation=2),
            nn.ELU(),
            )
        self.block, last_channels = self._make_block(nb_filter,nb_block)
        self.gap = nn.AdaptiveAvgPool3d((1,1,1))
        self.deep_fc = nn.Sequential(
            nn.Linear(last_channels,32,bias=True),
            nn.ELU(),
            )

        self.male_fc = nn.Sequential(
            nn.Linear(2,16,bias=True),
            nn.Linear(16,8,bias=True),
            nn.ELU(),
            )
        self.end_fc_with_gender = nn.Sequential(
            nn.Linear(40,16),
            nn.Linear(16,2),
            #nn.ReLU()
            )
        self.end_fc_without_gender = nn.Sequential(
            nn.Linear(32,16),
            nn.Linear(16,2),
            #nn.ReLU()
            )


    def _make_block(self, nb_filter, nb_block):
        blocks = []
        inchannels = nb_filter
        for i in range(nb_block):
            outchannels = nb_filter * pow(2,i+1)#inchannels * 2
            blocks.append(dense_layer2(inchannels,outchannels))
            inchannels = outchannels + inchannels
        return nn.Sequential(*blocks), inchannels

    def forward(self, x, male_input=None):
        x = self.pre(x)
        x = self.block(x)
        x = self.gap(x)
        x = torch.reshape(x,(x.size(0),-1))
        x = self.deep_fc(x)
        if self.use_gender:
            male = torch.reshape(male_input,(male_input.size(0),-1))
            male = self.male_fc(male)
            x = torch.cat([x,male.type_as(x)],1)
            x = self.end_fc_with_gender(x)
        else:
            x = self.end_fc_without_gender(x)
        return x

class ScaleDense2(nn.Module):
    def __init__(self,nb_filter=16, nb_block=4):
        '''
        Develop Scale Dense for brain age estimation

        Args:
            nb_filter (int): number of initial convolutional layer filter. Default: 8
            nb_block (int): number of Dense block. Default: 5
            use_gender (bool, optional): if use gender input. Default: True
        '''
        super(ScaleDense2,self).__init__()
        self.nb_block = nb_block
        self.pre = nn.Sequential(
            nn.Conv3d(1,nb_filter,kernel_size=7,stride=1,padding=1,dilation=2),
            nn.ELU(),
            )
        #self.pool1 =  nn.MaxPool3d(3, stride=2)
        self.block, last_channels = self._make_block(nb_filter,nb_block)
        self.out = nn.Sequential(
            nn.Conv3d(last_channels,512,(1,1,1),stride=1,padding=0,bias=False),
            nn.InstanceNorm3d(512),
            nn.ELU(),
            )

    def _make_block(self, nb_filter, nb_block):
        blocks = []
        inchannels = nb_filter
        for i in range(nb_block):
            outchannels = inchannels * 2
            blocks.append(dense_layer2(inchannels,outchannels))
            inchannels = outchannels + inchannels
        return nn.Sequential(*blocks), inchannels

    def forward(self, x):
        x = self.pre(x)
        x = self.block(x)  
        feature = self.out(x)
        return feature

class ScaleDense3(nn.Module):
    def __init__(self,nb_filter=16, nb_block=3, nb_block2=2):
        '''
        Develop Scale Dense for brain age estimation

        Args:
            nb_filter (int): number of initial convolutional layer filter. Default: 8
            nb_block (int): number of Dense block. Default: 5
            use_gender (bool, optional): if use gender input. Default: True
        '''
        super(ScaleDense3,self).__init__()
        self.nb_block = nb_block
        self.pre = nn.Sequential(
            nn.Conv3d(1,nb_filter,kernel_size=7,stride=1,padding=1,dilation=2),
            nn.ELU(),
            )
        self.block, last_channels = self._make_block(nb_filter,nb_block)
        self.block2, last_channels2 = self._make_block_2(last_channels,nb_block2)
        self.out = nn.Sequential(
            nn.Conv3d(last_channels2,512,(1,1,1),stride=1,padding=0,bias=False),
            nn.InstanceNorm3d(512),
            nn.ELU(),
            )
        self.dropout1 =  nn.Dropout3d(0.2)
        self.dropout2 =  nn.Dropout3d(0.2)

    def _make_block(self, nb_filter, nb_block):
        blocks = []
        inchannels = nb_filter
        for i in range(nb_block):         
            outchannels = nb_filter * pow(2,i+1)
            blocks.append(dense_layer2(inchannels,outchannels))
            inchannels = outchannels + inchannels
        return nn.Sequential(*blocks), inchannels
    
    def _make_block_2(self, nb_filter, nb_block):
        blocks = []
        inchannels = nb_filter
        for i in range(nb_block):         
            outchannels = nb_filter
            blocks.append(dense_layer2_2(inchannels,outchannels))
            inchannels = outchannels + inchannels
        return nn.Sequential(*blocks), inchannels
     
    def forward(self, x):
        x = self.pre(x)
        x = self.block(x)
        # x = self.dropout1(x)
        x = self.block2(x)
        # x = self.dropout2(x)
        feature = self.out(x)
        return feature
    
class ScaleDense_VAE(nn.Module):
    def __init__(self,nb_filter=32, nb_block=3, nb_block2=2):
        '''
        Develop Scale Dense for brain age estimation

        Args:
            nb_filter (int): number of initial convolutional layer filter. Default: 8
            nb_block (int): number of Dense block. Default: 5
            use_gender (bool, optional): if use gender input. Default: True
        '''
        super(ScaleDense_VAE,self).__init__()
        self.nb_block = nb_block
        # self.pre = nn.Sequential(
        #     nn.Conv3d(1,nb_filter,kernel_size=7,stride=1,padding=1,dilation=2),
        #     nn.ELU(),
        #     )
        self.block, last_channels = self._make_block(nb_filter,nb_block)
        self.block2, last_channels2 = self._make_block_2(last_channels,nb_block2)
        self.out = nn.Sequential(
            nn.Conv3d(last_channels2,512,(1,1,1),stride=1,padding=0,bias=False),
            nn.InstanceNorm3d(512),
            nn.ELU(),
            )
        self.dropout1 =  nn.Dropout3d(0.2)
        self.dropout2 =  nn.Dropout3d(0.2)
        #(83,104,79) (41,52,39) (20,26,19) (10,13,9)
        #(85,100,85) (42,50,42) (21,25,21) (10,12,10)
        #(77,97,79) (38,48,39) (19,24,19) (9,12,9)
        #(80,100,85) (40,50,42) (20,25,21) (10,12,10)
        #(84,101,87) (42,50,43) (21,25,21) (10,12,10)
        self.up1 = nn.Sequential(
            nn.Conv3d(512,nb_filter*4,(3,3,3),stride=1,padding=1,bias=False), 
            #nn.ConvTranspose3d(512, 64, kernel_size=3, stride=2, padding=1, output_padding=1),      
            nn.InstanceNorm3d(nb_filter*4),
            nn.ELU(),     
            nn.Upsample((21,25,21)),
            nn.Conv3d(nb_filter*4,nb_filter*4,(3,3,3),stride=1,padding=1,bias=False),     
            nn.InstanceNorm3d(nb_filter*4),
            nn.ELU(),
        )
        self.up2 = nn.Sequential(
            nn.Conv3d(nb_filter*4,nb_filter*2,(3,3,3),stride=1,padding=1,bias=False), 
            #nn.ConvTranspose3d(nb_filter*2, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  
            nn.InstanceNorm3d(nb_filter*2),
            nn.ELU(),
            nn.Upsample((42,50,43)),
            nn.Conv3d(nb_filter*2,nb_filter*2,(3,3,3),stride=1,padding=1,bias=False),     
            nn.InstanceNorm3d(nb_filter*2),
            nn.ELU(),
        )
        self.up3 = nn.Sequential(
            nn.Conv3d(nb_filter*2,nb_filter,(3,3,3),stride=1,padding=1,bias=False), 
            #nn.ConvTranspose3d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), 
            nn.InstanceNorm3d(nb_filter),
            nn.ELU(),
            nn.Upsample((84,101,87) ),
            nn.Conv3d(nb_filter,nb_filter,(3,3,3),stride=1,padding=1,bias=False),     
            nn.InstanceNorm3d(nb_filter),
            nn.ELU(),
        )
        
        self.out1 = nn.Sequential(nn.ReflectionPad3d(int(3/2)), 
                                  nn.Conv3d(nb_filter*4,1,(3,3,3),stride=1,padding=0,bias=False))
        self.out2 = nn.Sequential(nn.ReflectionPad3d(int(3/2)), 
                                  nn.Conv3d(nb_filter*2,1,(3,3,3),stride=1,padding=0,bias=False))
        self.out3 = nn.Sequential(nn.ReflectionPad3d(int(3/2)), 
                                  nn.Conv3d(nb_filter,1,(3,3,3),stride=1,padding=0,bias=False))

        self.out4 = nn.Sequential(nn.ReflectionPad3d(int(3/2)), 
                                  nn.Conv3d(nb_filter,1,(3,3,3),stride=1,padding=0,bias=False))
        
    def _make_block(self, nb_filter, nb_block):
        blocks = []
        inchannels = 1#nb_filter
        for i in range(nb_block):         
            outchannels = nb_filter * pow(2,i)
            blocks.append(dense_layer2(inchannels,outchannels))
            inchannels = outchannels + inchannels
        return nn.Sequential(*blocks), inchannels
    
    def _make_block_2(self, nb_filter, nb_block):
        blocks = []
        inchannels = nb_filter
        for i in range(nb_block):         
            outchannels = nb_filter
            blocks.append(dense_layer2_2(inchannels,outchannels))
            inchannels = outchannels + inchannels
        return nn.Sequential(*blocks), inchannels
       
    def forward(self, x, out_rec=False):
        #84 104 78
        #x = self.pre(x) #74 94 68
        x = self.block(x)
        # x = self.dropout1(x)
        x = self.block2(x)
        # x = self.dropout2(x)
        feature = self.out(x)         
        # feature = torch.tanh(feature)
        
        if out_rec: 
            x_1 =  self.up1(feature)
            x_2 =  self.up2(x_1)
            x_3 =  self.up3(x_2)
            #x_  = self.out4(x_3)
            #return feature, x_
            x_list = []
            x_list.append(self.out1(x_1))
            x_list.append(self.out2(x_2))
            x_list.append(self.out3(x_3))  
            return feature, x_list
        else:
            return feature

class ScaleDense_VAE2(nn.Module):
    def __init__(self,nb_filter=32, nb_block=3, nb_block2=2):
        '''
        Develop Scale Dense for brain age estimation

        Args:
            nb_filter (int): number of initial convolutional layer filter. Default: 8
            nb_block (int): number of Dense block. Default: 5
            use_gender (bool, optional): if use gender input. Default: True
        '''
        super(ScaleDense_VAE2,self).__init__()
        self.nb_block = nb_block
        # self.pre = nn.Sequential(
        #     nn.Conv3d(1,nb_filter,kernel_size=7,stride=1,padding=1,dilation=2),
        #     nn.ELU(),
        #     )
        self.block, last_channels = self._make_block(nb_filter,nb_block)
        # self.block2, last_channels2 = self._make_block_2(last_channels,nb_block2)
        self.out = nn.Sequential(
            nn.Conv3d(last_channels,512,(1,1,1),stride=1,padding=0,bias=False),
            nn.InstanceNorm3d(512)
            )
        self.dropout1 =  nn.Dropout3d(0.2)
        self.dropout2 =  nn.Dropout3d(0.2)
        #(83,104,79) (41,52,39) (20,26,19) (10,13,9)
        #(85,100,85) (42,50,42) (21,25,21) (10,12,10)
        #(77,97,79) (38,48,39) (19,24,19) (9,12,9)
        #(80,100,85) (40,50,42) (20,25,21) (10,12,10)
        #(84,101,87) (42,50,43) (21,25,21) (10,12,10)
        #(80,100,83)(40,50,41) (20,25,20) (10,12,10) 
        self.up1 = nn.Sequential(
            nn.Conv3d(512,nb_filter*16,(3,3,3),stride=1,padding=1,bias=False), 
            #nn.ConvTranspose3d(512, 64, kernel_size=3, stride=2, padding=1, output_padding=1),      
            nn.InstanceNorm3d(nb_filter*16),
            nn.ELU(),     
            nn.Upsample((20,25,20)),
            nn.Conv3d(nb_filter*16,nb_filter*16,(3,3,3),stride=1,padding=1,bias=False),     
            nn.InstanceNorm3d(nb_filter*16),
            nn.ELU(),
        )
        self.up2 = nn.Sequential(
            nn.Conv3d(nb_filter*16,nb_filter*4,(3,3,3),stride=1,padding=1,bias=False), 
            #nn.ConvTranspose3d(nb_filter*2, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  
            nn.InstanceNorm3d(nb_filter*4),
            nn.ELU(),
            nn.Upsample((40,50,41)),
            nn.Conv3d(nb_filter*4,nb_filter*4,(3,3,3),stride=1,padding=1,bias=False),     
            nn.InstanceNorm3d(nb_filter*4),
            nn.ELU(),
        )
        self.up3 = nn.Sequential(
            nn.Conv3d(nb_filter*4,nb_filter,(3,3,3),stride=1,padding=1,bias=False), 
            #nn.ConvTranspose3d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), 
            nn.InstanceNorm3d(nb_filter),
            nn.ELU(),
            nn.Upsample((80,100,83)),
            nn.Conv3d(nb_filter,nb_filter,(3,3,3),stride=1,padding=1,bias=False),     
            nn.InstanceNorm3d(nb_filter),
            nn.ELU(),
        )
        
        self.out1 = nn.Sequential(nn.ReflectionPad3d(int(3/2)), 
                                  nn.Conv3d(nb_filter*16,1,(3,3,3),stride=1,padding=0,bias=False))
        self.out2 = nn.Sequential(nn.ReflectionPad3d(int(3/2)), 
                                  nn.Conv3d(nb_filter*4,1,(3,3,3),stride=1,padding=0,bias=False))
        self.out3 = nn.Sequential(nn.ReflectionPad3d(int(3/2)), 
                                  nn.Conv3d(nb_filter,1,(3,3,3),stride=1,padding=0,bias=False))
        
    def _make_block(self, nb_filter, nb_block):
        blocks = []
        inchannels = 1#nb_filter
        for i in range(nb_block):  
            outchannels = nb_filter * pow(4,i)
            blocks.append(dense_layer2(inchannels,outchannels))
            inchannels = outchannels + inchannels
        return nn.Sequential(*blocks), inchannels
    
    def _make_block_2(self, nb_filter, nb_block):
        blocks = []
        inchannels = nb_filter
        for i in range(nb_block):         
            outchannels = nb_filter
            blocks.append(dense_layer2_2(inchannels,outchannels))
            inchannels = outchannels + inchannels
        return nn.Sequential(*blocks), inchannels
       
    def forward(self, x, out_rec=False):
        #84 104 78
        #x = self.pre(x) #74 94 68
        x = self.block(x)
        # x = self.block2(x)
        feature = self.out(x)
        # feature =  self.dropout1(feature)         
        feature = torch.tanh(feature)
        if out_rec: 
            x_1 =  self.up1(feature)
            x_2 =  self.up2(x_1)
            x_3 =  self.up3(x_2)
            x_list = []
            x_list.append(self.out1(x_1))
            x_list.append(self.out2(x_2))
            x_list.append(self.out3(x_3))  
            return feature, x_list
        else:
            return feature

class ScaleDense_VAE3(nn.Module):
    def __init__(self,nb_filter=32, nb_block=3, latent_dim=32):
        '''
        Develop Scale Dense for brain age estimation

        Args:
            nb_filter (int): number of initial convolutional layer filter. Default: 8
            nb_block (int): number of Dense block. Default: 5
            use_gender (bool, optional): if use gender input. Default: True
        '''
        super(ScaleDense_VAE3,self).__init__()
        self.nb_block = nb_block
        # self.pre = nn.Sequential(
        #     nn.Conv3d(1,nb_filter,kernel_size=7,stride=1,padding=1,dilation=2),
        #     nn.ELU(),
        #     )
        self.block, last_channels = self._make_block(nb_filter,nb_block)
        # self.block2, last_channels2 = self._make_block_2(last_channels,nb_block2)
        self.out = nn.Sequential(
            nn.Conv3d(last_channels,512,(1,1,1),stride=1,padding=0,bias=False),
            nn.InstanceNorm3d(512)
            )
        self.dropout1 =  nn.Dropout3d(0.2)
        self.dropout2 =  nn.Dropout3d(0.2)
        #(83,104,79) (41,52,39) (20,26,19) (10,13,9)
        #(85,100,85) (42,50,42) (21,25,21) (10,12,10)
        #(77,97,79) (38,48,39) (19,24,19) (9,12,9)
        #(80,100,85) (40,50,42) (20,25,21) (10,12,10)
        #(84,101,87) (42,50,43) (21,25,21) (10,12,10)
        #(80,100,83)(40,50,41) (20,25,20) (10,12,10) 
        self.up1 = nn.Sequential(
            nn.Conv3d(512,nb_filter*16,(3,3,3),stride=1,padding=1,bias=False), 
            #nn.ConvTranspose3d(512, 64, kernel_size=3, stride=2, padding=1, output_padding=1),      
            nn.InstanceNorm3d(nb_filter*16),
            nn.ELU(),     
            nn.Upsample((20,25,20)),
            nn.Conv3d(nb_filter*16,nb_filter*16,(3,3,3),stride=1,padding=1,bias=False),     
            nn.InstanceNorm3d(nb_filter*16),
            nn.ELU(),
        )
        self.up2 = nn.Sequential(
            nn.Conv3d(nb_filter*16,nb_filter*4,(3,3,3),stride=1,padding=1,bias=False), 
            #nn.ConvTranspose3d(nb_filter*2, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  
            nn.InstanceNorm3d(nb_filter*4),
            nn.ELU(),
            nn.Upsample((40,50,41)),
            nn.Conv3d(nb_filter*4,nb_filter*4,(3,3,3),stride=1,padding=1,bias=False),     
            nn.InstanceNorm3d(nb_filter*4),
            nn.ELU(),
        )
        self.up3 = nn.Sequential(
            nn.Conv3d(nb_filter*4,nb_filter,(3,3,3),stride=1,padding=1,bias=False), 
            #nn.ConvTranspose3d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), 
            nn.InstanceNorm3d(nb_filter),
            nn.ELU(),
            nn.Upsample((80,100,83)),
            nn.Conv3d(nb_filter,nb_filter,(3,3,3),stride=1,padding=1,bias=False),     
            nn.InstanceNorm3d(nb_filter),
            nn.ELU(),
        )
        
        self.out1 = nn.Sequential(nn.ReflectionPad3d(int(3/2)), 
                                  nn.Conv3d(nb_filter*16,1,(3,3,3),stride=1,padding=0,bias=False))
        self.out2 = nn.Sequential(nn.ReflectionPad3d(int(3/2)), 
                                  nn.Conv3d(nb_filter*4,1,(3,3,3),stride=1,padding=0,bias=False))
        self.out3 = nn.Sequential(nn.ReflectionPad3d(int(3/2)), 
                                  nn.Conv3d(nb_filter,1,(3,3,3),stride=1,padding=0,bias=False))
        
        self.mu = nn.Conv3d(512,latent_dim,1)
        self.logvar = nn.Conv3d(512,latent_dim,1)
        self.decoder_conv = nn.Conv3d(latent_dim,512,1)
        
    def _make_block(self, nb_filter, nb_block):
        blocks = []
        inchannels = 1#nb_filter
        for i in range(nb_block):  
            outchannels = nb_filter * pow(4,i)
            blocks.append(dense_layer2(inchannels,outchannels))
            inchannels = outchannels + inchannels
        return nn.Sequential(*blocks), inchannels
    
    def _make_block_2(self, nb_filter, nb_block):
        blocks = []
        inchannels = nb_filter
        for i in range(nb_block):         
            outchannels = nb_filter
            blocks.append(dense_layer2_2(inchannels,outchannels))
            inchannels = outchannels + inchannels
        return nn.Sequential(*blocks), inchannels
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def compute_kl_loss(self, mu, logvar):
        # 计算 KL 散度
        # kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        return kl.mean()
    
    def encoder(self, x):
        #84 104 78
        #x = self.pre(x) #74 94 68
        x = self.block(x)
        # x = self.block2(x)
        feature = self.out(x)
        mu = self.mu(feature)
        logvar = self.logvar(feature)
        return mu, logvar
    
    def decoder(self, z):
        x_0 = self.decoder_conv(z)
        x_1 =  self.up1(x_0)
        x_2 =  self.up2(x_1)
        x_3 =  self.up3(x_2)
        x_list = []
        x_list.append(self.out1(x_1))
        x_list.append(self.out2(x_2))
        x_list.append(self.out3(x_3))  
        return x_list
    
    def forward(self, x, out_rec=False):
        # feature =  self.dropout1(feature)         
        mu, logvar = self.encoder(x)
        loss = self.compute_kl_loss(mu, logvar)
        z = self.reparameterize(mu, logvar)
        if out_rec:
            x_list = self.decoder(z) 
            return z, x_list, loss
        else:
            return z

class Embeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    # Copied from transformers.models.bert.modeling_bert.BertEmbeddings.__init__
    def __init__(self, max_position_embeddings=1201, hidden_size=512):
        super().__init__()
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.position_embeddings2 = nn.Embedding(max_position_embeddings, hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.register_buffer("position_ids", torch.arange(max_position_embeddings).expand((1, -1)))

        #self.dropout_embedding = nn.Dropout2d(0.2)

    def forward(self,inputs_embeds=None,past_key_values_length=0 ):
        seq_length = inputs_embeds.shape[1]
        position_ids = self.position_ids[:, past_key_values_length:past_key_values_length+seq_length]
        position_embeddings = self.position_embeddings(position_ids)
        position_embeddings2 = self.position_embeddings2(position_ids)
        embeddings = inputs_embeds
        embeddings = position_embeddings * embeddings + position_embeddings2
        embeddings = self.LayerNorm(embeddings)
        return embeddings

class ScaleDense_MAE(nn.Module):
    def __init__(self,nb_filter=32, nb_block=3, nb_block2=2, in_channels=1, img_size=(84, 101, 87), patch_size=(8, 8, 8), emb_size=512,dropout=0.1):
        '''
        Develop Scale Dense for brain age estimation

        Args:
            nb_filter (int): number of initial convolutional layer filter. Default: 8
            nb_block (int): number of Dense block. Default: 5
            use_gender (bool, optional): if use gender input. Default: True
        '''
        super(ScaleDense_MAE,self).__init__()
        self.patch_size = patch_size
        self.patch_num = 11*13*11
        self.nb_block = nb_block

        self.block, last_channels = self._make_block(nb_filter,nb_block)
        self.block2, last_channels2 = self._make_block_2(last_channels,nb_block2)
        self.out = nn.Sequential(
            nn.Conv3d(last_channels2,512,(1,1,1),stride=1,padding=0,bias=False),
            nn.InstanceNorm3d(512),
            nn.ELU(),
            )
        self.dropout1 =  nn.Dropout3d(0.2)
        self.dropout2 =  nn.Dropout3d(0.2)
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.pos_embedding = nn.Parameter(torch.randn(1, self.patch_num + 1, emb_size))
        self.dropout = nn.Dropout(dropout)

        self.decoder_pred = nn.Linear(emb_size, patch_size[0]*patch_size[1]*patch_size[2]*in_channels, bias=True)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_size))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, emb_size))
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        self.TE = TransformerEncoder(seq_len=self.patch_num, d_model=emb_size, num_heads=8, 
                            num_layers=3, dim_feedforward=1024)
        self.pos_embed = Embeddings(self.patch_num+1,emb_size)
        self.pos_embed2 = Embeddings(self.patch_num+1,emb_size)

    def _make_block(self, nb_filter, nb_block):
        blocks = []
        inchannels = 1#nb_filter
        for i in range(nb_block):         
            outchannels = nb_filter * pow(2,i)
            blocks.append(dense_layer2(inchannels,outchannels))
            inchannels = outchannels + inchannels
        return nn.Sequential(*blocks), inchannels
    
    def _make_block_2(self, nb_filter, nb_block):
        blocks = []
        inchannels = nb_filter
        for i in range(nb_block):         
            outchannels = nb_filter
            blocks.append(dense_layer2_2(inchannels,outchannels))
            inchannels = outchannels + inchannels
        return nn.Sequential(*blocks), inchannels
    
    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore
 
    def unpatchify(self, x, feature_shape):
        """
        x: (N, L, patch_size**3*1)
        imgs: (N, 1, H, W,L)
        """
        p =  self.patch_size[0]
        _,_,w,h,l =feature_shape
        
        x = x.reshape(shape=(x.shape[0], w,h,l, p, p, p,1))
        x = torch.einsum('nwhlopqc->ncwohplq', x)
        imgs = x.reshape(shape=(x.shape[0], 1, w * p, h * p, l * p))
        return imgs
    
    def forward(self, x, out_rec=True, M_Ratio=0.75):
        #84 104 78
        x = F.pad(x,(1,0,2,1,2,2),'replicate')
        x = self.block(x)
        x = self.block2(x)
        feature = self.out(x)         
        feature = torch.tanh(feature)

        B,C,W,H,L = feature.shape
        feature = feature.view(B,C,-1).permute([0,2,1])
        
        feature_m, mask, ids_restore = self.random_masking(self.pos_embed(feature,1), M_Ratio)
        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(feature_m.shape[0], ids_restore.shape[1] + 1 - feature_m.shape[1], 1)
        feature_ = torch.cat([feature_m[:, :, :], mask_tokens], dim=1)  # no cls token
        feature_ = torch.gather(feature_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, feature_.shape[2]))  # unshuffle

        # add pos embed
        feature_ = self.pos_embed(feature_,0)
        feature_ = self.TE(feature_)
        feature_ = torch.tanh(feature_)

        if out_rec: 
            x_list = []
            x_ = self.decoder_pred(feature_)
            x_ = self.unpatchify(x_,(B,C,W,H,L))[:,:,2:-2,2:-1,1:]
            x_list.append(x_)
            feature_  = feature_.permute([0,2,1]).view(B,C,W,H,L)
            feature  = feature.permute([0,2,1]).view(B,C,W,H,L)
            return feature, mask, x_list
        
        else:
            feature_  = feature_.permute([0,2,1]).view(B,C,W,H,L)
            feature  = feature.permute([0,2,1]).view(B,C,W,H,L)
            return feature, mask
    
class ScaleDense_MAE2(nn.Module):
    def __init__(self,nb_filter=32, nb_block=3, nb_block2=2, in_channels=1, img_size=(84, 101, 87), patch_size=(8, 8, 8), emb_size=512,dropout=0.1):
        '''
        Develop Scale Dense for brain age estimation

        Args:
            nb_filter (int): number of initial convolutional layer filter. Default: 8
            nb_block (int): number of Dense block. Default: 5
            use_gender (bool, optional): if use gender input. Default: True
        '''
        super(ScaleDense_MAE2,self).__init__()
        self.patch_size = patch_size
        self.patch_num = 11*13*11
        self.nb_block = nb_block

        self.block, last_channels = self._make_block(nb_filter,nb_block)
        self.block2, last_channels2 = self._make_block_2(last_channels,nb_block2)
        self.out = nn.Sequential(
            nn.Conv3d(last_channels2,512,(1,1,1),stride=1,padding=0,bias=False),
            nn.InstanceNorm3d(512),
            nn.ELU(),
            )
        self.dropout1 =  nn.Dropout3d(0.2)
        self.dropout2 =  nn.Dropout3d(0.2)
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.pos_embedding = nn.Parameter(torch.randn(1, self.patch_num + 1, emb_size))
        self.dropout = nn.Dropout(dropout)

        self.decoder_pred = nn.Linear(emb_size, patch_size[0]*patch_size[1]*patch_size[2]*in_channels, bias=True)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_size))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, emb_size))
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        self.TE = TransformerEncoder(seq_len=self.patch_num+1, d_model=emb_size, num_heads=8, 
                            num_layers=3, dim_feedforward=1024)
        self.TE2 = TransformerEncoder(seq_len=self.patch_num+1, d_model=emb_size, num_heads=8, 
                            num_layers=3, dim_feedforward=1024)
        self.pos_embed = Embeddings(self.patch_num+1,emb_size)
        self.pos_embed2 = Embeddings(self.patch_num+1,emb_size)

    def _make_block(self, nb_filter, nb_block):
        blocks = []
        inchannels = 1#nb_filter
        for i in range(nb_block):         
            outchannels = nb_filter * pow(2,i)
            blocks.append(dense_layer2(inchannels,outchannels))
            inchannels = outchannels + inchannels
        return nn.Sequential(*blocks), inchannels
    
    def _make_block_2(self, nb_filter, nb_block):
        blocks = []
        inchannels = nb_filter
        for i in range(nb_block):         
            outchannels = nb_filter
            blocks.append(dense_layer2_2(inchannels,outchannels))
            inchannels = outchannels + inchannels
        return nn.Sequential(*blocks), inchannels
    
    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore
 
    def unpatchify(self, x, feature_shape):
        """
        x: (N, L, patch_size**3*1)
        imgs: (N, 1, H, W,L)
        """
        p =  self.patch_size[0]
        _,_,w,h,l =feature_shape
        
        x = x.reshape(shape=(x.shape[0], w,h,l, p, p, p,1))
        x = torch.einsum('nwhlopqc->ncwohplq', x)
        imgs = x.reshape(shape=(x.shape[0], 1, w * p, h * p, l * p))
        return imgs
    
    def forward(self, x, out_rec=True, M_Ratio=0.75):
        #84 104 78
        x = F.pad(x,(1,0,2,1,2,2),'replicate')
        x = self.block(x)
        x = self.block2(x)
        feature = self.out(x)         
        feature = torch.tanh(feature)

        B,C,W,H,L = feature.shape
        feature = feature.view(B,C,-1).permute([0,2,1])

        feature_m, mask, ids_restore = self.random_masking(self.pos_embed(feature,1), M_Ratio)
        # append cls token
        cls_token = self.pos_embed(self.cls_token,0)
        cls_tokens = cls_token.expand(feature_m.shape[0], -1, -1)
        feature_m = torch.cat((cls_tokens, feature_m), dim=1)
        feature_m = self.TE(feature_m)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(feature_m.shape[0], ids_restore.shape[1] + 1 - feature_m.shape[1], 1)
        feature_ = torch.cat([feature_m[:, 1:, :], mask_tokens], dim=1)  # no cls token
        feature_ = torch.gather(feature_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, feature_.shape[2]))  # unshuffle
        feature_ = torch.cat([feature_m[:, :1, :], feature_], dim=1)  # append cls token
        # add pos embed
        feature_ = self.pos_embed(feature_,0)
        feature_ = self.TE2(feature_)
        feature_ = feature_[:, 1:, :]
        feature_m = feature_m[:, 1:, :]
        if out_rec: 
            x_list = []
            x_ = self.decoder_pred(feature_)
            x_ = self.unpatchify(x_,(B,C,W,H,L))[:,:,2:-2,2:-1,1:]
            x_list.append(x_)
            feature_  = feature_.permute([0,2,1]).view(B,C,W,H,L)
            feature  = feature.permute([0,2,1]).view(B,C,W,H,L)
            return feature, mask, x_list
        else:
            feature_  = feature_.permute([0,2,1]).view(B,C,W,H,L)
            feature  = feature.permute([0,2,1]).view(B,C,W,H,L)
            return feature, mask

class ScaleDense_MAE3(nn.Module):
    def __init__(self,nb_filter=32, nb_block=3, nb_block2=2, in_channels=1, img_size=(84, 101, 87), patch_size=(8, 8, 8), emb_size=512,dropout=0.1):
        '''
        Develop Scale Dense for brain age estimation

        Args:
            nb_filter (int): number of initial convolutional layer filter. Default: 8
            nb_block (int): number of Dense block. Default: 5
            use_gender (bool, optional): if use gender input. Default: True
        '''
        super(ScaleDense_MAE3,self).__init__()
        self.patch_size = patch_size
        self.patch_num = 10*12*10
        self.nb_block = nb_block

        self.block, last_channels = self._make_block(nb_filter,nb_block)
        self.block2, last_channels2 = self._make_block_2(last_channels,nb_block2)
        self.out = nn.Sequential(
            nn.Conv3d(last_channels2,512,(1,1,1),stride=1,padding=0,bias=False),
            nn.InstanceNorm3d(512),
            nn.ELU(),
            )
        self.dropout1 =  nn.Dropout3d(0.2)
        self.dropout2 =  nn.Dropout3d(0.2)
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.pos_embedding = nn.Parameter(torch.randn(1, self.patch_num + 1, emb_size))
        self.dropout = nn.Dropout(dropout)

        self.decoder_pred = nn.Linear(emb_size, patch_size[0]*patch_size[1]*patch_size[2]*in_channels, bias=True)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_size))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, emb_size))
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        self.TE = TransformerEncoder(seq_len=self.patch_num, d_model=emb_size, num_heads=8, 
                            num_layers=3, dim_feedforward=1024)
        self.pos_embed = Embeddings(self.patch_num+1,emb_size)
        self.pos_embed2 = Embeddings(self.patch_num+1,emb_size)
        self.up1 = nn.Sequential(
            nn.Conv3d(512,nb_filter*4,(3,3,3),stride=1,padding=1,bias=False), 
            #nn.ConvTranspose3d(512, 64, kernel_size=3, stride=2, padding=1, output_padding=1),      
            nn.InstanceNorm3d(nb_filter*4),
            nn.ELU(),     
            nn.Upsample((21,25,21)),
            nn.Conv3d(nb_filter*4,nb_filter*4,(3,3,3),stride=1,padding=1,bias=False),     
            nn.InstanceNorm3d(nb_filter*4),
            nn.ELU(),
        )
        self.up2 = nn.Sequential(
            nn.Conv3d(nb_filter*4,nb_filter*2,(3,3,3),stride=1,padding=1,bias=False), 
            #nn.ConvTranspose3d(nb_filter*2, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  
            nn.InstanceNorm3d(nb_filter*2),
            nn.ELU(),
            nn.Upsample((42,50,43)),
            nn.Conv3d(nb_filter*2,nb_filter*2,(3,3,3),stride=1,padding=1,bias=False),     
            nn.InstanceNorm3d(nb_filter*2),
            nn.ELU(),
        )
        self.up3 = nn.Sequential(
            nn.Conv3d(nb_filter*2,nb_filter,(3,3,3),stride=1,padding=1,bias=False), 
            #nn.ConvTranspose3d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), 
            nn.InstanceNorm3d(nb_filter),
            nn.ELU(),
            nn.Upsample((84,101,87) ),
            nn.Conv3d(nb_filter,nb_filter,(3,3,3),stride=1,padding=1,bias=False),     
            nn.InstanceNorm3d(nb_filter),
            nn.ELU(),
        )
        
        self.out1 = nn.Sequential(nn.ReflectionPad3d(int(3/2)), 
                                  nn.Conv3d(nb_filter*4,1,(3,3,3),stride=1,padding=0,bias=False))
        self.out2 = nn.Sequential(nn.ReflectionPad3d(int(3/2)), 
                                  nn.Conv3d(nb_filter*2,1,(3,3,3),stride=1,padding=0,bias=False))
        self.out3 = nn.Sequential(nn.ReflectionPad3d(int(3/2)), 
                                  nn.Conv3d(nb_filter,1,(3,3,3),stride=1,padding=0,bias=False))


    def _make_block(self, nb_filter, nb_block):
        blocks = []
        inchannels = 1#nb_filter
        for i in range(nb_block):         
            outchannels = nb_filter * pow(2,i)
            blocks.append(dense_layer2(inchannels,outchannels))
            inchannels = outchannels + inchannels
        return nn.Sequential(*blocks), inchannels
    
    def _make_block_2(self, nb_filter, nb_block):
        blocks = []
        inchannels = nb_filter
        for i in range(nb_block):         
            outchannels = nb_filter
            blocks.append(dense_layer2_2(inchannels,outchannels))
            inchannels = outchannels + inchannels
        return nn.Sequential(*blocks), inchannels
    
    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore
 
    def unpatchify(self, x, feature_shape):
        """
        x: (N, L, patch_size**3*1)
        imgs: (N, 1, H, W,L)
        """
        p =  self.patch_size[0]
        _,_,w,h,l =feature_shape
        
        x = x.reshape(shape=(x.shape[0], w,h,l, p, p, p,1))
        x = torch.einsum('nwhlopqc->ncwohplq', x)
        imgs = x.reshape(shape=(x.shape[0], 1, w * p, h * p, l * p))
        return imgs
    
    def forward(self, x, out_rec=True, M_Ratio=0.75):
        #84 104 78
        x = self.block(x)
        x = self.block2(x)
        feature = self.out(x)         
        feature = torch.tanh(feature)

        B,C,W,H,L = feature.shape
        feature = feature.view(B,C,-1).permute([0,2,1])
        
        feature_m, mask, ids_restore = self.random_masking(self.pos_embed(feature,1), M_Ratio)
        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(feature_m.shape[0], ids_restore.shape[1] + 1 - feature_m.shape[1], 1)
        feature_ = torch.cat([feature_m[:, :, :], mask_tokens], dim=1)  # no cls token
        feature_ = torch.gather(feature_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, feature_.shape[2]))  # unshuffle

        # add pos embed
        feature_ = self.pos_embed(feature_,0)
        feature_ = self.TE(feature_)
        feature_ = torch.tanh(feature_)

        if out_rec: 
            feature_  = feature_.permute([0,2,1]).view(B,C,W,H,L)
            feature  = feature.permute([0,2,1]).view(B,C,W,H,L)
            x_1 =  self.up1(feature_)
            x_2 =  self.up2(x_1)
            x_3 =  self.up3(x_2)
            #x_  = self.out4(x_3)
            #return feature, x_
            x_list = []
            x_list.append(self.out1(x_1))
            x_list.append(self.out2(x_2))
            x_list.append(self.out3(x_3))  
            return feature, mask, x_list
        else:
            feature_  = feature_.permute([0,2,1]).view(B,C,W,H,L)
            feature  = feature.permute([0,2,1]).view(B,C,W,H,L)
            return feature, mask

class res_layer(nn.Module):
    def __init__(self,inchannels,outchannels):
        super(res_layer,self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(inchannels,outchannels,(3,3,3),stride=1,padding=1,bias=False),
            nn.InstanceNorm3d(outchannels),
            nn.ELU(),
            nn.Conv3d(outchannels,outchannels,(3,3,3),stride=1,padding=1,bias=False),     
            nn.InstanceNorm3d(outchannels),
            nn.ELU(),
            nn.MaxPool3d(2,2),
        )
        self.bypass = nn.Sequential(
            nn.Conv3d(inchannels,outchannels,(1,1,1),stride=1,padding=0,bias=False),
            nn.MaxPool3d(2,2),
        )
    def forward(self,x):
        x = self.block(x) + self.bypass(x)
        return x
    
class ScaleDense_Dis(nn.Module):
    def __init__(self,nb_filter=16, nb_block=3):
        '''
        Develop Scale Dense for brain age estimation

        Args:
            nb_filter (int): number of initial convolutional layer filter. Default: 8
            nb_block (int): number of Dense block. Default: 5
            use_gender (bool, optional): if use gender input. Default: True
        '''
        super(ScaleDense_Dis,self).__init__()
        self.nb_block = nb_block
        self.block, last_channels = self._make_block(nb_filter,nb_block)
        self.out = nn.Sequential(
            spectral_norm(nn.Conv3d(last_channels,last_channels,(1,1,1),stride=1,padding=0,bias=False)),
            # nn.InstanceNorm3d(last_channels),
            nn.ELU(),
            nn.Conv3d(last_channels,1,(1,1,1),stride=1,padding=0,bias=False),
            )

    def _make_block(self, nb_filter, nb_block):
        blocks = []
        inchannels = 1#nb_filter
        for i in range(nb_block):         
            outchannels = nb_filter * pow(2,i)
            blocks.append(dense_layer3(inchannels,outchannels))
            inchannels = outchannels + inchannels
        return nn.Sequential(*blocks), inchannels
    
    def forward(self, x):
        #84 104 78
        x = self.block(x)
        x = self.out(x)
        x = torch.sigmoid(x)
        return x

class FC(nn.Module):
    def __init__(self,):
        '''
        Develop Scale Dense for brain age estimation

        Args:
            nb_filter (int): number of initial convolutional layer filter. Default: 8
            nb_block (int): number of Dense block. Default: 5
            use_gender (bool, optional): if use gender input. Default: True
        '''
        super(FC,self).__init__()
        self.fc1 = nn.Linear(170,512)
        self.fc2 = nn.Linear(512,2)
        self.relu = nn.ReLU(0.2)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class dense_layer(nn.Module):
    def __init__(self,inchannels,outchannels):
        super(dense_layer,self).__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inchannels,outchannels,3,stride=1,padding=1,bias=False),
            nn.LayerNorm(outchannels),
            nn.ELU(),
            nn.Conv1d(outchannels,outchannels,3,stride=1,padding=1,bias=False),     
            nn.LayerNorm(outchannels),
            nn.ELU(),
            nn.MaxPool1d(2,2),
        )
    def forward(self,x):
        new_features = self.block(x)
        x = F.max_pool1d(x,2) 
        x = torch.cat([new_features,x], 1)
        return x
    
class GeneDesEncoder(nn.Module):
    def __init__(self, in_channels=3, base_channels=8):
        super(GeneDesEncoder, self).__init__()
        self.block, last_channels = self._make_block(base_channels,4)
        self.out = nn.Sequential(
            nn.Conv1d(last_channels,512,1,stride=1,padding=0,bias=False),
            nn.BatchNorm1d(512),
            nn.ELU(),
            )
        self.pos_embed = Embeddings(16,3)
    
    def _make_block(self, nb_filter, nb_block):
        blocks = []
        inchannels = 3
        for i in range(nb_block):         
            outchannels = nb_filter * pow(2,i+1) 
            blocks.append(dense_layer(inchannels,outchannels))
            inchannels = outchannels + inchannels
        return nn.Sequential(*blocks), inchannels
    
    def patchify(self,x,p=16):
        pad_len = math.ceil(x.shape[2]/ p)*p-x.shape[2]
        x = F.pad(x,(0,pad_len),'constant',0)
        l = x.shape[2] // p
        x = x.reshape(shape=(x.shape[0], 3, l, p))
        x = torch.einsum('nclp->nlcp', x)
        # x = x.reshape(shape=(x.shape[0],l, p*3))
        return x
    
    def forward(self, x):
        B,C,L = x.shape
        x = self.patchify(x)
        x = x.reshape(B*x.shape[1],C,x.shape[3])
        # x = self.pos_embed(x.permute([0,2,1])).permute([0,2,1])
        out = self.block(x)
        out =  self.out(out).view(B,L//16,-1).permute([0,2,1])
        return out
 
class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, mode="down"):
        super(ResidualBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.BatchNorm1d(out_channels) #
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.BatchNorm1d(out_channels) #
        # 如果输入和输出通道不匹配，需要在捷径路径中添加一个卷积层进行匹配
        if mode=="down":
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1),
                nn.BatchNorm1d(out_channels),
                nn.MaxPool1d(2,2)
            )
        else:
            self.shortcut = nn.Identity()
        self.nolin = nn.ReLU()

    def forward(self, x):
        out = self.nolin(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = F.max_pool1d(out,2) 
        out += self.shortcut(x)
        return self.nolin(out)

class GeneResEncoder(nn.Module):
    def __init__(self, input_channels=3, ):
        super(GeneResEncoder, self).__init__()
        self.encoder = nn.Sequential(
            ResidualBlock1D(input_channels, 8),
            ResidualBlock1D(8, 16),
            ResidualBlock1D(16, 32),
            ResidualBlock1D(32, 512),
        )

    def patchify(self,x,p=16):
        pad_len = math.ceil(x.shape[2]/ p)*p-x.shape[2]
        x = F.pad(x,(0,pad_len),'constant',0)
        l = x.shape[2] // p
        x = x.reshape(shape=(x.shape[0], 3, l, p))
        x = torch.einsum('nclp->nlcp', x)
        # x = x.reshape(shape=(x.shape[0],l, p*3))
        return x
    
    def forward(self, x=None, M_Ratio=0):
        # 编码
        if M_Ratio!= 0:
           x,_ = self.random_masking(x,M_Ratio)

        x = self.encoder(x).permute([0,2,1])
        # x = self.encoder(x)
        return x
    
    # def random_masking(self,x, mask_ratio):
    #     P = 16 
    #     N, D, L = x.shape
    #     mask = torch.ones_like(x)

    #     for start in range(0, L, P):
    #         end = min(start + P, L)
    #         block_size = end - start
    #         num_zeros = max(1, int(block_size * mask_ratio))  # 至少置零一个元素
    #         zero_indices = torch.randperm(block_size)[:num_zeros]
    #         mask[:, :, start:end][:, :, zero_indices] = 0

    #     return x * mask, mask

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, D, L], sequence
        """
        N, D, L = x.shape  # batch, dim, length
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(1, L, device=x.device).repeat(N,1)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        x_masked = x * (1.0-mask.unsqueeze(1))

        return x_masked, mask

class Embeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    # Copied from transformers.models.bert.modeling_bert.BertEmbeddings.__init__
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)

        # position = torch.arange(config.max_position_embeddings).unsqueeze(1)    
        # div_term = torch.exp(
        #     torch.arange(0, config.hidden_size, 2) * (-math.log(10000.0) / config.hidden_size)
        # )                                                
        # pe = torch.zeros(config.max_position_embeddings, config.hidden_size) 
        # pe[:, 0::2] = torch.sin(position * div_term)   
        # pe[:, 1::2] = torch.cos(position * div_term)          
        # self.position_embeddings = nn.Parameter(pe)  #pe.cuda()# 
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.position_embeddings2 = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.emb_dropout_prob)

        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        if version.parse(torch.__version__) > version.parse("1.6.0"):
            self.register_buffer(
                "token_type_ids",
                torch.zeros(self.position_ids.size(), dtype=torch.long, device=self.position_ids.device),
                persistent=False,
            )

        self.rescale_embeddings = config.rescale_embeddings
        self.hidden_size = config.hidden_size
    
    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        if input_ids is not None and inputs_embeds is not None:
            input_shape = (input_ids.size()[0],input_ids.size()[1]+inputs_embeds.size()[1])
        elif input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if input_ids is not None and inputs_embeds is not None:
            inputs_embeds = torch.cat([self.word_embeddings(input_ids),inputs_embeds],dim=1)
        elif input_ids is not None:
            inputs_embeds = self.word_embeddings(input_ids)

        if self.rescale_embeddings:
            inputs_embeds = inputs_embeds * (self.hidden_size ** 0.5)

        embeddings = inputs_embeds
        position_embeddings = self.position_embeddings(position_ids)
        position_embeddings2 = self.position_embeddings2(position_ids)
        embeddings = position_embeddings * embeddings + position_embeddings2    

        # embeddings = inputs_embeds
        # position_embeddings = self.position_embeddings[:seq_length].unsqueeze(0).to(embeddings.device)
        # position_embeddings2 = self.position_embeddings2(position_ids)
        # embeddings =  position_embeddings2*embeddings + position_embeddings

        embeddings = self.dropout(embeddings)
        embeddings = self.LayerNorm(embeddings)
        return embeddings,inputs_embeds

class GeneMLPEncoder(nn.Module):
    def __init__(self, config, input_channels=16*3, latent_feature=32):
        super(GeneMLPEncoder, self).__init__()
        self.config = copy.deepcopy(config)
        self.config.hidden_size = input_channels
        self.embeddings = Embeddings(self.config) 
        self.encoder = nn.Sequential(
            nn.Linear(input_channels, latent_feature),
            nn.ReLU(),
            nn.Linear(latent_feature, 512),
        )
    
    def patchify(self,x,p=16):
        pad_len = math.ceil(x.shape[2]/ p)*p-x.shape[2]
        x = F.pad(x,(0,pad_len),'constant',0)
        l = x.shape[2] // p
        x = x.reshape(shape=(x.shape[0], 3, l, p))
        x = torch.einsum('nclp->nlpc', x)
        x = x.reshape(shape=(x.shape[0],l, p*3))
        return x
    
    def forward(self, snp=None, M_Ratio=0, mask=None,p=16,use_embedding=True,gate=None,snp2=None):
        # 编码
        snp = snp.long()
        snp_oh = F.one_hot(snp.long().clamp(0,2),num_classes=3)
        snp_oh[snp==3] = 0

        if snp2 is not None:
            snp2 = snp2.long()
            snp2_oh = F.one_hot(snp2.long().clamp(0,2),num_classes=3)
            snp2_oh[snp2==3] = 0
            gate = gate.unsqueeze(2)
            snp_oh = gate*snp_oh + (1.0-gate)* snp2_oh#torch.flip(snp_oh,dims=[0])
            # snp = gate*snp + (1.0-gate)* torch.flip(snp,dims=[0])

        if snp2 is None and gate is not None:
            snp_oh = snp_oh* gate.unsqueeze(2)

        x = snp_oh.permute([0,2,1]).float()

        if mask != None:
            x = x*(1.0-mask.unsqueeze(1))
        elif M_Ratio!= 0:
           x,mask = self.random_masking(x,M_Ratio)
        else:
            mask = torch.zeros((snp.shape[0],snp.shape[1])).to(x.device)
        x = self.patchify(x,p=p)

        if use_embedding:
            x,_ = self.embeddings(inputs_embeds=x)
        x = self.encoder(x)
        return x,mask
   
    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, D, L], sequence
        """
        N, D, L = x.shape  # batch, dim, length
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(1, L, device=x.device).repeat(N,1)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        x_masked = x * (1.0-mask.unsqueeze(1))

        return x_masked, mask

class GeneMLPEncoder2(nn.Module):
    def __init__(self, input_channels=8, ):
        super(GeneMLPEncoder2, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_channels, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
        )
    
    def forward(self, x=None):
        # 编码
        x = self.encoder(x)
        return x

class GeneVAE(nn.Module):
    def __init__(self, input_channels=16*3,latent_dim=32 ):
        super(GeneVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_channels, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, input_channels),
            )
        self.mu = nn.Linear(latent_dim,latent_dim)
        self.logvar = nn.Linear(latent_dim,latent_dim)

    def unpatchify(self, x, p=16):
        """
        x: [B, L, output_channels]  where output_channels = 3 * patch_size
        """
        B, L, _ = x.shape
        x = x.reshape(B, L, 3, p)        # [B, L, 3, p]
        x = torch.einsum('nlcp->nclp', x)              # [B, 3, L, p]
        x = x.reshape(B, 3, L * p)       # [B, 3, T]
        return x
    
    def patchify(self,x,p=16):
        pad_len = math.ceil(x.shape[2]/ p)*p-x.shape[2]
        x = F.pad(x,(0,pad_len),'constant',0)
        l = x.shape[2] // p
        x = x.reshape(shape=(x.shape[0], 3, l, p))
        x = torch.einsum('nclp->nlpc', x)
        x = x.reshape(shape=(x.shape[0],l, p*3))
        return x
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def compute_kl_loss(self, mu, logvar):
        # 计算 KL 散度
        # kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=2)
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        return kl.mean()
    
    def forward(self, x=None, p=16, out_rec=False):
        # 编码
        B,_,L = x.shape
        x = self.patchify(x,p=p)
        feature = self.encoder(x)
        mu = self.mu(feature)
        logvar = self.logvar(feature)
        loss = self.compute_kl_loss(mu, logvar)
        z = self.reparameterize(mu, logvar)
        if out_rec:
            x_rec = self.decoder(z) 
            x_rec = self.unpatchify(x_rec,p=p)
            return z, x_rec[:,:,0:L], loss
        else:
            return z
    
class GeneVAE2(nn.Module):
    def __init__(self, input_channels=16*3,latent_dim=512 ):
        super(GeneVAE2, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_channels, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, input_channels),
            )

    def unpatchify(self, x, p=16):
        """
        x: [B, L, output_channels]  where output_channels = 3 * patch_size
        """
        B, L, _ = x.shape
        x = x.reshape(B, L, 3, p)        # [B, L, 3, p]
        x = torch.einsum('nlcp->nclp', x)              # [B, 3, L, p]
        x = x.reshape(B, 3, L * p)       # [B, 3, T]
        return x
    
    def patchify(self,x,p=16):
        pad_len = math.ceil(x.shape[2]/ p)*p-x.shape[2]
        x = F.pad(x,(0,pad_len),'constant',0)
        l = x.shape[2] // p
        x = x.reshape(shape=(x.shape[0], 3, l, p))
        x = torch.einsum('nclp->nlpc', x)
        x = x.reshape(shape=(x.shape[0],l, p*3))
        return x
    
    def forward(self, x=None, p=16, out_rec=False):
        # 编码
        B,_,L = x.shape
        x = self.patchify(x,p=p)
        feature = self.encoder(x)
        feature = F.sigmoid(feature)
        if out_rec:
            x_rec = self.decoder(feature) 
            x_rec = self.unpatchify(x_rec,p=p)
            return feature, x_rec[:,:,0:L]
        else:
            return feature

class GeneVAE3(nn.Module):
    def __init__(self, input_channels=16*3,latent_dim=32 ):
        super(GeneVAE3, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_channels, 512),
            nn.GELU(),
            nn.Linear(512, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.GELU(),
            nn.Linear(512, input_channels),
            )
        self.mu = nn.Linear(latent_dim,latent_dim)
        self.logvar = nn.Linear(latent_dim,latent_dim)
        self.cls = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, 4),
        )
        self.embed =  nn.Sequential(nn.Linear(4,latent_dim),
                                    nn.GELU(),
                                    nn.Linear(latent_dim, latent_dim))
        self.norm = nn.LayerNorm(latent_dim)
        self.mlp = nn.Sequential(
            nn.Linear(2, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim*2),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim,latent_dim),
        )
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def compute_kl_loss(self, mu, logvar):
        # 计算 KL 散度
        # kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=2)
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        return kl.mean()
    
    def forward(self, x=None, mri_f=None, age_sex=None, out_rec=False):
        # 编码
        B,C,L = x.shape
        # feature = self.encoder(x.reshape(B,C*L))

        style = self.mlp(age_sex)
        alpha, beta = torch.chunk(style, 2, dim=1)
        mri_f = self.mlp2(self.norm(mri_f) * alpha + beta) + mri_f

        c_emb = F.log_softmax(self.cls(mri_f),dim=-1)     #self.cls(mri_f)#
        feature =  self.embed(c_emb)

        mu = self.mu(feature)
        logvar = self.logvar(feature)
        loss = self.compute_kl_loss(mu, logvar)
        # z = self.reparameterize(mu, logvar)
        z = feature

        if out_rec:
            x_rec = self.decoder(z).reshape(B,C,L)
            return z, x_rec[:,:,0:L], loss
        else:
            return z
 