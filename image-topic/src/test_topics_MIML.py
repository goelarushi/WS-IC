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
# from tensorboardX import SummaryWriter

from torch.autograd import Variable
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from skimage.transform import resize
import pickle
import topicmodel_utils as tm_utils
import json
this_dir = osp.dirname(osp.realpath((__file__)))
print(this_dir)

# sys.path.append('/home/s1985335/geomloss')
sys.path.append(osp.join(this_dir,'coco-caption-master/pycocoevalcap'))

import os.path as osp
import argparse
# from geomloss import SamplesLoss  # See also ImagesLoss, VolumesLoss
from eval_coco_captions import bleu_eval
os.environ["CUDA_VISIBLE_DEVICES"]="3"
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


model_dir = osp.join(this_dir,'..', 'MIML_Topic_Models','models_OT+KLloss_with5topics')#models_MIMLtopics_OT+KLloss_rmsprop_clip0.1
# fake_dir = osp.join(this_dir, '..', 'generated_topic_captions')
fake_dir = model_dir
if not os.path.isdir(fake_dir):
    os.mkdir(fake_dir)


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

# loss_function = torch.nn.BCEWithLogitsLoss()
# loss_function = torch.nn.KLDivLoss()
# loss_function = torch.nn.L1Loss()
# loss_function = torch.nn.CrossEntropyLoss()
# loss_function = SamplesLoss(loss="sinkhorn", p=1, blur=.01)

# class MIML(nn.Module):
#     def __init__(self,input_dim, n_instances, n_labels, n_sub_labels):
#         super(MIML,self).__init__()
#         #L: n_labels
#         #K: n_sub_labels 
#         self.input_dim = input_dim
#         self.n_instances = n_instances
#         self.n_labels = n_labels
#         self.n_sub_labels = n_sub_labels
#         self.with_batchnorm = True
#         self.with_softmax = True
#         self.sub_concept_layer = nn.modules.Conv2d(self.input_dim, self.n_labels * self.n_sub_labels, 1, 1) #shape: [Batch_Size, (L*K), n_instances]
#         if self.with_batchnorm:
#             model = [self.sub_concept_layer, nn.BatchNorm2d(self.n_labels * self.n_sub_labels), nn.ReLU(True)]
#         else:
#             model = [self.sub_concept_layer, nn.ReLU(True)]
#         self.concept_model = nn.Sequential(*model)

#         self.sub_concept_pooling = nn.modules.MaxPool2d((self.n_sub_labels, 1), stride=(1,1))
#         self.instance_pooling = nn.modules.MaxPool2d((self.n_instances,1), stride=(1,1))
#         self.softmax = nn.Softmax(dim=-1)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, input_emd):
#         ## input_emd: [Batchsize, n_instances, input_dim]: [64, 36, 2048]
#         # K: 50, L: 512
#         #reshape input into: [64, 2048, 36, 1]
#         self.input_emd = input_emd.permute(0,2,1).unsqueeze(3)
#         self.conv_output = self.concept_model(self.input_emd) #[64, (50*512), 36]
#         ##reshape conv output
#         self.conv_output = self.conv_output.view(-1, self.n_labels, self.n_sub_labels, self.n_instances)
#         self.sub_concept_pooling_output = self.sub_concept_pooling(self.conv_output)  
#         self.sub_concept_pooling_output = self.sub_concept_pooling_output.view(-1, self.n_labels, self.n_instances).permute(0,2,1)

#         if self.with_softmax:
#             softmax_normalization_output = self.softmax(self.sub_concept_pooling_output)
#             self.output = self.instance_pooling(softmax_normalization_output).view(-1, self.n_labels)
#         else:
#             self.output = self.instance_pooling(self.sub_concept_pooling_output).view(-1, self.n_labels)

#         return self.output

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
        #print(group['params'])
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)


def load_model(model,epoch):
    # gpus = [2, 3]
    # if len(gpus) > 1:
    #     model = torch.nn.DataParallel(model)

    model.to(device)
    model_file = osp.join(model_dir,"model_{}.pt".format(epoch))
    model.load_state_dict(torch.load(model_file))

    # model.cuda()
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

def kl(input, target):
    input = input.log()
    input[input == -inf] = -0.00
    print(input)
    out = F.kl_div(input, target)


    return out

def one_hot_labels(labels,num_classes):
    y = torch.eye(num_classes)
    return y[labels]

def testmodel(args,testloader,epoch,model):
    model = load_model(model,epoch)
    # model.cuda()
    model.eval()
    top1 = AverageMeter()
    top2 = AverageMeter()
    index = np.arange(512)
    wfinal_data =[]
    tfinal_data =[]
    topic_data =[]

    prob_json_file = osp.join(fake_dir, str(epoch)+'ot_test_topics_2014.json')
    wjson_file = osp.join(fake_dir, str(epoch) + '_ot_test_topword_captions_2014.json')
    tjson_file = osp.join(fake_dir, str(epoch) + '_ot_test_threshold_captions_2014.json')
    with torch.no_grad():
        for ii, (inputs, img_id, topics) in enumerate(testloader):
            ims = inputs.to(device)
            target = topics.to(device)
            target = target/target.sum()
            outputs,_,_ = model(ims)  # compute output
            # outputs = torch.Tensor(np.random.rand(ims.size(0), num_topics))

            # outputs = F.relu(outputs)
            # outputs = F.softmax(outputs, dim=1)
            prob_topics = outputs.cpu().detach().numpy().squeeze()
            # prob_topics = target.cpu().numpy().squeeze()
            # sigmoid_outputs = outputs
            # outputs[outputs>0.04] = 1.
            # outputs[outputs < 0.04] = 0.

            # outputs = outputs/outputs.sum()
            # print(outputs)
            # print(ims)
            # outputs = F.sigmoid(outputs)
            # plt.subplot(1,2,1)
            # plt.bar(index,outputs.cpu().numpy().squeeze())
            # plt.title('Predicted Topics')
            # plt.subplot(1, 2, 2)
            # plt.bar(index,target.cpu().numpy().squeeze())
            # plt.title('Ground-Truth Topics')
            
            # plt.savefig(osp.join(this_dir,'..', str(img_id[0])+'.png'))
            # plt.show()
            # raw_input('enter')
            # l1_val = l1(outputs,target)
            # mse_val = mse(outputs, target)
            # # outputs[outputs == 0] = 1
            # kl_div = kl(outputs.log(), target)
            # top1.update(mse_val.item(), ims.size(0))
            # top2.update(kl_div.item(), ims.size(0))
            # top3.update(l1_val.item(), ims.size(0))


            threshold_prob_topics = outputs.cpu().detach().numpy().squeeze()
            # threshold_prob_topics = target.cpu().numpy().squeeze()
            data = {"image_id": int(img_id[0]), "topic": prob_topics.tolist()}
            #
            topic_data.append(data)
            # print('Generating captions for top words')

            gen_captions = tm_utils.get_relevant_words(threshold_prob_topics)
            data = {"image_id": int(img_id[0]), "caption": str(gen_captions)}
            wfinal_data.append(data)
            # print('Generating captions for threshold')

            gen_captions = tm_utils.get_threshold_words(threshold_prob_topics)
            data = {"image_id": int(img_id[0]), "caption": str(gen_captions)}
            tfinal_data.append(data)
        with open(prob_json_file, 'w') as outfile:
            json.dump(topic_data, outfile, indent=3)
        print(prob_json_file)
        with open(wjson_file, 'w') as outfile:
            json.dump(wfinal_data, outfile, indent=3)
        bleu_eval(wjson_file)
        with open(tjson_file, 'w') as outfile:
            json.dump(tfinal_data, outfile, indent=3)
        print(tjson_file)
        bleu_eval(tjson_file)
    # print(' * Average MSE {0:.3f}'.format(top1.avg))
    # print(' * Average KL {0:.3f}'.format(top2.avg))
    # print(' * Average L1 {0:.3f}'.format(top3.avg))
    return top1.avg

def main(args):
    model = MIML(input_dim, n_instances, num_topics)
    # gpus = [2,3]
    # if len(gpus) > 1:
    #     model = torch.nn.DataParallel(model)
    # #
    # model.to(device)

    dataset = Vision2MultipleTopics

    val_loader = torch.utils.data.DataLoader(
        dataset(data_folder, data_name, 'TEST'),
        batch_size=args.batch_size, shuffle=False, num_workers=16, pin_memory=True)


    print('val data loaded')

    testmodel(args, val_loader, args.best_epoch, model)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train and Test Img Topic Models')
    parser.add_argument('--batch_size',type=int,default=1)
    parser.add_argument('--best_epoch', type=int, default=30)
    parser.add_argument('--model_name', type=str, default='resnet')
    args = parser.parse_args()
    main(args)
