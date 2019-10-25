from __future__ import print_function, division
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import numpy as np
import torchvision
import time
import copy
import os
import os.path as osp

this_dir = osp.dirname(osp.realpath((__file__)))
import sys
os.environ["CUDA_VISIBLE_DEVICES"]="3"

sys.path.append('/home/s1985335/geomloss')
sys.path.append(os.path.join(this_dir, '..', 'geomloss'))

import argparse
from geomloss import SamplesLoss  # See also ImagesLoss, VolumesLoss
from tensorboardX import SummaryWriter
# from sinkhorn_ot import *
# from torch.autograd import Variable
import matplotlib.pyplot as plt
# import cv2
# from PIL import Image
# from skimage.transform import resize
# import pickle
# import topicmodel_utils as tm_utils
# import json
# from eval_coco_captions import bleu_eval
use_gpu = torch.cuda.is_available()
# torch.manual_seed(1)
# torch.cuda.manual_seed(1)
# np.random.seed(1)

num_topics = 512
n_instances = 36
input_dim = 2048
n_sub_labels = 25
attention_dim = 1024
num_classes = 90
lib_path = osp.join(this_dir,'..','..','src')
sys.path.insert(0,lib_path)
from coco_loader import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
## DATA DIR ##
data_folder = osp.join(this_dir,'..','image-captioning-bottom-up-top-down-master', 'dataset_2014')  # folder with data files saved by create_input_files.py
data_name = 'coco_5_cap_per_img_5_min_word_freq'  # base name shared by data files


model_dir = osp.join(this_dir,'..', 'MIML_Topic_Models','models_KLloss_with5topics')#'models_MIMLtopics_OTloss_rmsprop_clip0.1')
fake_dir = osp.join(this_dir, '..', 'fake_captions')

if not os.path.isdir(model_dir):
    os.mkdir(model_dir)
if not os.path.isdir(fake_dir):
    os.mkdir(fake_dir)


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


# loss_function = torch.nn.CrossEntropyLoss()
# loss_function = torch.nn.BCELoss()

# Define a Sinkhorn (~Wasserstein) loss between sampled measures
ot_loss_function = SamplesLoss(loss="sinkhorn", p=1, blur=.01)
kl_loss_function = torch.nn.KLDivLoss(reduction='batchmean')

class MIML(nn.Module):
    def __init__(self,input_dim, n_instances, n_labels):
        super(MIML,self).__init__()
        self.L = input_dim
        self.D = 1024
        self.K = n_labels
        self.M = torch.Tensor([1.])
        self.iclassifier = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.ReLU(),
            nn.Linear(self.D, self.K),            
        )
        
        # instance detector
        self.det = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.ReLU(),
            nn.Linear(self.D, self.K),            
        )       

     # The default is log_sum-exp        
    def pool_func(self,y):
        #self.M = torch.ones(1) * 1.0
        # self.M = self.M.to(y.device)
        # y = (1/self.M) * torch.log(torch.sum(torch.exp(self.M*y),dim=1))  
        # Y_prob = torch.sigmoid(y)      
        y = torch.sum(y, dim=1)         
        # Y_prob = torch.sigmoid(y) 
        return y
    
    # this is the attension function ( In our case it is the Gaussian.)
    def get_att_func(self, x, m, s):
        z = (x - m)/s
        x =  torch.exp( -(z**2))        
        return x
    
    def forward(self, input_emd):
        ## input_emd: [Batchsize, n_instances, input_dim]: [64, 36, 2048]
        ##GP0T0
    
        self.x = input_emd
        self.y = self.iclassifier(self.x)

        # m = torch.mean(self.y, dim=1).unsqueeze(1)
        # s = torch.std(self.y,dim=1).unsqueeze(1)
        
        # z = self.get_att_func(self.y,m,s)
        # self.y = torch.mul(z,self.y)    
        # Y_prob = self.pool_func(self.y)
        # Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        # Y_hat = torch.ge(Y_prob, 0.5).float()      
 
        # return Y_prob, Y_hat, self.x
        z = self.det(self.x)        
        z = torch.softmax(z,dim=1)        
        y = torch.sigmoid(self.y)
        
        y = torch.mul(z,y)   
        att_both = y       

        Y_prob = self.pool_func(y)
        
        Y_hat = torch.ge(Y_prob, 0.5).float()       

        return Y_prob, Y_hat, att_both        
        

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        # print(group['params'])
        for param in group['params']:
            print(param.grad)
            param.grad.data.clamp_(-grad_clip, grad_clip)


def load_model(model,epoch):
    model_file = osp.join(model_dir,"model_{}.pt".format(epoch))
    model.load_state_dict(torch.load(model_file))
    model.cuda()
    return model

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def mse(input, target):
    return F.mse_loss(input,target)
    # return torch.sum((input - target) ** 2)


def kl(input, target):
    out = F.kl_div(input, target)

    return out
def l1(input,target):
    out = torch.mean(torch.abs(input - target))
    return out

def one_hot_labels(labels,num_classes):
    y = torch.eye(num_classes)
    return y[labels]
prev_epoch = 0
def trainmodel(dataloader,testloader, model,optimizer,num_epochs,args):
    
        lr = args.lr
        writer = SummaryWriter()
        index = np.arange(512)
        losses = AverageMeter()
        # model = load_model(model, prev_epoch)
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)
            #model.to(device)
            model.train()  # Set model to training mode
            # lambda1 = 0.001
            lam = 1   ## 0 for OT loss, 1 for KL loss and 0.1 for OT+KL loss
            running_loss = 0.0
            iters = 0
             # Iterate over data.
            for i, (inputs, img_id, topics) in enumerate(dataloader):
                # print(inputs, img_id, topics)
                # raw_input('enter')
                inputs = inputs.view(-1,36,2048)
                topics = topics.view(-1, 512)
                ims = inputs.to(device)
                labels = topics.to(device)
                labels = labels.squeeze()
                batchsize = ims.shape[0]
                # zero the parameter gradients
                optimizer.zero_grad()
                outputs,_,_ = model(ims)
                
                try:
                    ot_loss = ot_loss_function(outputs, labels)
                except:
                    print(outputs)
                    print(outputs.shape)
                ###for regularization
                kl_loss = kl_loss_function(F.softmax(outputs,dim=1).log(), labels)   ### softmaxonly for KL loss only 
                loss = (1-lam)*ot_loss + lam*kl_loss
                loss.backward()


                torch.nn.utils.clip_grad_norm_(model.parameters(),0.1)

                # clip_gradient(optimizer, 0.1)
                optimizer.step()
                #print(preds)
                #raw_input('enter')
                losses.update(loss.item(), batchsize)

                running_loss+=loss.item()
                # Check how the generator is doing by saving G's output on fixed_noise
                if (iters % 100 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
                    count = iters+1
                    # print(
                    #     "[Epoch %d/%d] [Batch %d/%d] [CE Loss {loss.val:.4f}]"
                    #     % (epoch, num_epochs, i, len(dataloader),loss=losses)   # running_loss/count)
                    # )
                    print('Epoch: [{0}][{1}/{2}]\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(dataloader),
                                                                          loss=losses))
                    writer.add_scalar('OT_loss', running_loss/count, iters)
                iters += 1
            if((epoch % 2 == 0) and (epoch!=0)):
                # lr = lr*1.5 #1.2
                lr = lr
                # optimizer = optim.SGD(model.parameters(), lr=lr)
                optimizer = optim.RMSprop(model.parameters(), lr=args.lr)

                # optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0005)
                print('Updated Learning Rate', lr)

            output_file = os.path.join(model_dir, "model_{}.pt".format(epoch))
            print("Saving model to {}".format(output_file))
            torch.save(model.state_dict(), output_file)
            testmodel(args,testloader,writer,epoch,model)
        writer.close()
            


def testmodel(args,testloader,writer,epoch,model):
    model = load_model(model,epoch)
    model.eval()
    top1 = AverageMeter()
    running_mse = 0.0
    iters_val = 0
    with torch.no_grad():
        for ii, (inputs, img_id, topics) in enumerate(testloader):
            ims = inputs.to(device)
            topics = topics.view(-1,512)
            target = topics.to(device)
            outputs,_,_ = model(ims)  # compute output
            outputs = outputs.expand(5,512)
            mse_val = mse(outputs, target)
            top1.update(mse_val.item(), ims.size(0))
            
    print(' * Average MSE {0:.6f}'.format(top1.avg))
    writer.add_scalar('Val MSE', top1.avg, epoch)




def main(args):
    model = MIML(input_dim, n_instances, num_topics)
    # gpus = [0,1]
    # if len(gpus) > 1:
    #     model = torch.nn.DataParallel(model)

    model.to(device)

    dataset = Vision2MultipleTopics
    # optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0005)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr)
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
    num_epochs = args.epochs
    train_loader = torch.utils.data.DataLoader(
        dataset(data_folder, data_name, 'TRAIN'),
        batch_size=args.batch_size, shuffle=True, num_workers = 16, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        dataset(data_folder, data_name, 'VAL'),
        batch_size=1, shuffle=False, num_workers=16, pin_memory=True)


    print('train data loaded')

    trainmodel(train_loader, val_loader,model,optimizer,num_epochs,args)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train and Test Img Topic Models')
    parser.add_argument('--batch_size',type=int,default=20)
    parser.add_argument('--lr',type = float,default=0.0002) #Tried ##0.0001 0.0002  ##0.001
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--model_name', type=str, default='resnet')
    parser.add_argument('--fold', type=int, default=1)
    args = parser.parse_args()
    main(args)
