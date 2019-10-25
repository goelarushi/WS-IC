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
# from skimage import io
# from torchvision import datasets, models, transforms
# from torchvision.models import DenseNet
# from os.path import join as pjoin
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
input_dim = 2048
attention_dim = 1024
num_classes = 90
lib_path = osp.join(this_dir,'..','..','src')
sys.path.insert(0,lib_path)
from coco_loader import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
## DATA DIR ##
data_folder = osp.join(this_dir,'..','image-captioning-bottom-up-top-down-master', 'dataset_2014')  # folder with data files saved by create_input_files.py
data_name = 'coco_5_cap_per_img_5_min_word_freq'  # base name shared by data files


model_dir = osp.join(this_dir,'..', 'Attention_Topic_Models','models_KLlossNew_with5topics')
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
class Attention(nn.Module):
    def __init__(self,input_dim, attention_dim):
        super(Attention,self).__init__()
        self.input_dim = input_dim
        self.attention_dim = attention_dim
        self.attn = nn.Sequential(
                    nn.Linear(self.input_dim, self.attention_dim),
                    nn.Linear(self.attention_dim,1),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=0.5),

        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_emd):

        batch_size = input_emd.shape[0]
        input_emd = input_emd.contiguous()

        attn = self.attn(input_emd)  # attention over the sequence length
        #print(attn.shape)
        alpha = self.softmax(attn)  # gives the probability values for the time steps in the sequence (weights to each time step)

        #print(lstm_emd.shape,lstm_emd)
        attn_feature_map = input_emd * alpha  # gives attention weighted embedding
        attn_feature_map = torch.sum(attn_feature_map, dim=1)  # computes the weighted sum

        return attn_feature_map


class AttnTopicModel(nn.Module):
    def __init__(self, input_dim, attention_dim, num_topics):
        super(AttnTopicModel, self).__init__()
        self.input_dim = input_dim
        self.topics = num_topics
        self.attention_dim =attention_dim
        self.attention = Attention(self.input_dim, self.attention_dim)
        self.topic_model = nn.Sequential(
                           nn.Linear(self.input_dim, 1024),
                           nn.ReLU(True),
                           nn.Linear(1024, self.topics))

    def forward(self, input):
        input_attn = self.attention(input)
        output = self.topic_model(input_attn)
        return output

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        # print(group['params'])
        for param in group['params']:
            print(param.grad.data.norm(2))
            input('enter')
            param.grad.data.clamp_(-grad_clip, grad_clip)
            print(param.grad.data.norm(2))
            input('enter')


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


def trainmodel(dataloader,testloader, model,optimizer,num_epochs,args):

        lr = args.lr
        writer = SummaryWriter()
        index = np.arange(512)
        losses = AverageMeter()
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
                outputs = model(ims)
                
                outputs = F.relu(outputs)
                # outputs = outputs/outputs.sum()
                outputs = F.softmax(outputs, dim=1)
                # print(torch.sum(outputs, dim=1))
                # print(outputs)
                # Code snippet to plot topics
                # print(labels.shape, outputs.shape, outputs.dim())
                # input('enter')
                # plt.subplot(1, 2, 1)
                # plt.bar(index,outputs[0].detach().cpu().numpy().squeeze())
                # plt.title('Predicted Topics')
                # plt.subplot(1, 2, 2)
                # plt.bar(index,labels[0].detach().cpu().numpy().squeeze())
                # plt.title('Ground-Truth Topics')
                
                # plt.savefig(osp.join(this_dir,'..', str(img_id[0])+'.png'))
                # plt.show()
                # input('enter')

                ###Code snippet for sinkhorn ot approximation loss
                # cost = ((outputs[:,None,:] - labels[:,:,None])**2)
                # cost= cost/cost.max()
                # # cost = cost.to(device=device, dtype=torch.double)
                # # labels_ = labels.detach().requires_grad_()
                # loss = SinkhornOT.apply(labels, outputs, cost, 1.0, 10)

                # loss = loss.mean()

                ot_loss = ot_loss_function(outputs, labels)
                ###for regularization
                kl_loss = kl_loss_function(outputs.log(), labels)
                loss = (1-lam)*ot_loss + lam*kl_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
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
            outputs = model(ims)  # compute output
            outputs = outputs.expand(5,512)
            mse_val = mse(outputs, target)
            top1.update(mse_val.item(), ims.size(0))
            
    print(' * Average MSE {0:.6f}'.format(top1.avg))
    writer.add_scalar('Val MSE', top1.avg, epoch)




def main(args):
    model = AttnTopicModel(input_dim, attention_dim, num_topics)
    # model = WSTopicModel(input_dim, attention_dim, num_topics)
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
    parser.add_argument('--lr',type = float,default=0.0002) #Tried ##0.0001   ##0.001
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--model_name', type=str, default='resnet')
    parser.add_argument('--fold', type=int, default=1)
    args = parser.parse_args()
    main(args)
