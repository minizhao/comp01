import os
from scipy.misc import imread,imsave
import sys
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from model import Model
from torch.autograd import Variable
import torch.nn as nn

#滑动窗口切分数据
def split_data():
    # 选择了其中的一个文件夹，其中有61张图片
    path='./data/SRAD2018_TRAIN_010'
    files= sorted(os.listdir(path))

    for file in files:
        imgs=sorted(os.listdir(os.path.join(path,file)))
        q = []
        q.append(None)
        for img in imgs:
            img=imread(os.path.join(path,file,img))
            # 有三通道，数据只是单通道，三个通道值都一样,取出第一个通道
            img=img[:,:,0]
            img=(img>100)*1
            q.append(img)
            if len(q)>36:
                q.pop(0)
                yield q


#划分出输入和标签
def make_x_y(imgs_list):
    imgs=[]
    for img in imgs_list:
        img=np.array(img).astype(np.float)
        # min, max = 0, 255
        # img = (img-min)/(max-min)
        imgs.append(img)
    x=imgs[:31]
    y=imgs[31:]
    return np.array(x),np.array(y)

def train():
    model=Model().cuda(1)
    criterion = nn.MSELoss().cuda(1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    # nn.utils.clip_grad_norm(parameters, max_norm, norm_type=2)

    # 固定住resnet网络，不更新参数
    for param in model.resnet.parameters():
        param.requires_grad=False

    loss_list=[]
    for epoch  in range(20):
        count=0
        for imgs in split_data():
            inp,target=make_x_y(imgs)
            inp=Variable(torch.from_numpy(inp).type(torch.FloatTensor)).cuda(1)
            #每个文件夹下面会有26组训练数据
            target=Variable(torch.from_numpy(target).type(torch.FloatTensor)).view(5,-1).cuda(1)
            preds=model(inp)
            loss=criterion(preds,target)
            loss_list.append(loss.data.cpu().numpy())
            if len(loss_list)==26:
                count+=1
                if count%100==0:
                    torch.save(model,'model.pt')
                print('epoch:{},step{},mean_loss:{}'.format(epoch,count,np.mean(loss_list)))
                loss_list=[]
            optimizer.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm(model.parameters(), 1.0)
            optimizer.step()


if __name__ == '__main__':
    train()
