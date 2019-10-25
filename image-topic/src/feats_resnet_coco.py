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
import sys
import os.path as osp
import argparse
from skimage import io
from torchvision import datasets, models, transforms
from torchvision.models import DenseNet
from os.path import join as pjoin
from tensorboardX import SummaryWriter
import resnet
from torch.autograd import Variable
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from skimage.transform import resize
import pickle
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2"
use_gpu = torch.cuda.is_available()
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
this_dir = osp.dirname(osp.realpath((__file__)))
num_topics = 25

lib_path = osp.join(this_dir,'..','..','src')
sys.path.insert(0,lib_path)
from coco_loader import MSCocoDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
## DATA DIR ##

# data_dir = osp.join(this_dir,'..','..', 'coco/images')
data_dir = '/home/arushi/Desktop/data/coco/images'    ### loading data from home rather than raid
topic_dir = osp.join(this_dir, '..', 'lda_512topics')
model_dir = osp.join(this_dir,'..', 'topic_models')
feat_dir = osp.join(this_dir, '..', 'new_resnet50')

if not os.path.isdir(model_dir):
    os.mkdir(model_dir)
if not os.path.isdir(feat_dir):
    os.mkdir(feat_dir)



def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def model_def(model_name):
   print(model_name)
   model_ft = resnet.resnet50(pretrained=True)
   set_parameter_requires_grad(model_ft, True)
   print(model_ft)
   return model_ft


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        #print(group['params'])
        for param in group['params']:
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
    return torch.sum((input - target) ** 2)

def one_hot_labels(labels,num_classes):
    y = torch.eye(num_classes)
    return y[labels]

def featmodel(dataloader, model,args):
    output_dir = osp.join(feat_dir,'train2017')
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    model.to(device)
    model.eval()  # Set model to training mode
    # Iterate over data.
    for i, (inputs, labels, img_id) in enumerate(dataloader):
        with torch.no_grad():
            inputs = inputs.to(device)
            labels = labels.to(device)

            output_file = osp.join(output_dir, str(img_id[0])+'.npy')

            # zero the parameter gradients
            feat_outputs = model(inputs)
            features = feat_outputs.cpu()
            feats = features.detach().numpy()
            feats = feats.squeeze()
            np.save(output_file, feats)
            print(output_file)

def main(args):
    feat_extract = True
    model = model_def(args.model_name)
    dataset = MSCocoDataset

    image_size = 224 ## for resnet
    transform_train = transforms.Compose([transforms.Resize([image_size, image_size]),transforms.ToTensor()])

    train_data = dataset(data_dir, topic_dir, mode='train',transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_data,batch_size=1, shuffle=False, num_workers=16, pin_memory=True)
    print('train data loaded')

    featmodel(train_loader, model,args)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train and Test Img Topic Models')
    parser.add_argument('--batch_size',type=int,default=32)
    parser.add_argument('--lr',type = float,default=0.00001)
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--model_name', type=str, default='resnet')
    parser.add_argument('--fold', type=int, default=1)
    args = parser.parse_args()
    main(args)
