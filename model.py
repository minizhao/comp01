import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import sys
import torch
from numpy import random
import models


# 用resnet 编码图片特征类
class resnet(nn.Module):
    """docstring for ."""
    def __init__(self):
        super(resnet, self).__init__()
        self.resnet=models.resnet18(pretrained=True)
        self.lne=nn.Linear(51200, 128)
    def forward(self,inp):
        inp_restnet=inp.unsqueeze(1).repeat(1,3,1,1)
        out_resnet=self.resnet(inp_restnet)
        out=self.lne(out_resnet)
        return out

class Model(nn.Module):
    """docstring for ."""
    def __init__(self, batch_size=1,inp_dim=128,h_dim=256,e_layers=1,d_layers=1,e_bidire=False):
        super(Model, self).__init__()
        self.batch_size = batch_size
        self.e_layers=e_layers
        self.d_layers=d_layers
        self.h_dim=h_dim
        self.e_bidire=e_bidire
        self.inp_dim=inp_dim
        self.encoder=nn.GRU(inp_dim,h_dim, e_layers, dropout=0.1, bidirectional=e_bidire,batch_first=True)
        self.decoder=nn.GRU(h_dim,h_dim, e_layers, dropout=0.1, batch_first=True)
        self.o2p = nn.Linear(h_dim, 251001)
        self.resnet=resnet()
        self.dp = nn.Dropout(0.5)



    def forward(self,inp):
        out_feature=self.resnet(inp).unsqueeze(0)
        preds=torch.zeros(5,251001).cuda(1)
        h=self.init_hidden(self.e_layers,self.batch_size,self.h_dim,self.e_bidire)
        output, hidden = self.encoder(out_feature, h)
        #取最后的一个输入和状态做生成初始数据
        d_inp=output[:,-1,:].unsqueeze(0)

        for i in range(5):
            output, hidden=self.decoder(d_inp, hidden)
            preds[i]=self.dp(self.o2p(output.squeeze()))
        return preds


    def init_hidden(self,n_layers,batch_size,hidden_size,bidirectional=False):
        if bidirectional:
            return Variable(torch.zeros(n_layers*2,batch_size,hidden_size)).cuda(1)
        else:
            return Variable(torch.zeros(n_layers,batch_size,hidden_size)).cuda(1)
