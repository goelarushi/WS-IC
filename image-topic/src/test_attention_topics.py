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

from torch.autograd import Variable
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from skimage.transform import resize
import pickle
import topicmodel_utils as tm_utils
import json
from eval_coco_captions import bleu_eval
os.environ["CUDA_VISIBLE_DEVICES"]="3"
use_gpu = torch.cuda.is_available()
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
this_dir = osp.dirname(osp.realpath((__file__)))
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


model_dir = osp.join(this_dir,'..', 'bottomupfeat+attn_topic_models_bceloss_2014')
fake_dir = osp.join(this_dir, '..', 'generated_topic_captions')

if not os.path.isdir(fake_dir):
    os.mkdir(fake_dir)


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

# loss_function = torch.nn.BCEWithLogitsLoss()
# loss_function = torch.nn.KLDivLoss()
# loss_function = torch.nn.L1Loss()
loss_function = torch.nn.CrossEntropyLoss()

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
        #print(group['params'])
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)


def load_model(model,epoch):
    # gpus = [2, 3]
    # if len(gpus) > 1:
    #     model = torch.nn.DataParallel(model)

    # model.to(device)
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

    prob_json_file = osp.join(fake_dir, 'bce_test_topics_2014.json')
    wjson_file = osp.join(fake_dir, str(epoch) + '_test_topword_captions_2014.json')
    tjson_file = osp.join(fake_dir, str(epoch) + '_test_threshold_captions_2014.json')
    with torch.no_grad():
        for ii, (inputs, img_id, topics) in enumerate(testloader):
            ims = inputs.cuda()
            target = topics.cuda(async=True)
            # target = target/target.sum()
            outputs = model(ims)  # compute output
            # outputs = np.random.rand(ims.size(0), num_topics)
            # outputs = torch.Tensor(outputs).cuda()
                    # print(outputs)
            # outputs = F.relu(outputs)
            # print(outputs)
            # outputs = F.softmax(outputs, dim=1)
            outputs = F.sigmoid(outputs)
            prob_topics = outputs.cpu().detach().numpy().squeeze()

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
            #
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
    model = AttnTopicModel(input_dim, attention_dim, num_topics)
    # gpus = [2,3]
    # if len(gpus) > 1:
    #     model = torch.nn.DataParallel(model)

    model.to(device)

    dataset = Vision2Topics

    val_loader = torch.utils.data.DataLoader(
        dataset(data_folder, data_name, 'TRAIN'),
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
