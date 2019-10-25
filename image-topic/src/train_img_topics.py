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
os.environ["CUDA_VISIBLE_DEVICES"]="2,3"
use_gpu = torch.cuda.is_available()
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
this_dir = osp.dirname(osp.realpath((__file__)))
num_topics = 512

lib_path = osp.join(this_dir,'..','..','src')
sys.path.insert(0,lib_path)
from coco_loader import MSCocoDataset, MSCocoFeatures

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
## DATA DIR ##

data_dir = osp.join(this_dir,'..','..', 'coco/images')
feat_dir = osp.join(this_dir, '..', 'resnet50')
topic_dir = osp.join(this_dir, '..', 'lda_512topics')
model_dir = osp.join(this_dir,'..', 'celoss+newmodelall512_topic_models')
fake_dir = osp.join(this_dir, '..', 'fake_captions')

if not os.path.isdir(model_dir):
    os.mkdir(model_dir)
if not os.path.isdir(fake_dir):
    os.mkdir(fake_dir)

#### Topic model things
from gensim import models
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from gensim.utils import simple_preprocess
from gensim.corpora.dictionary import Dictionary
import numpy as np
import os
import os.path as osp
# '''
# directories
# '''
# model_path = osp.join(this_dir, '..', 'lda_models', 'lda_25.gensim')
# dictionary_path = osp.join(this_dir, '..', 'lda_25topics', 'train2017_dictionary.dict')
#
# '''
# load data
# '''
# lda_model = models.LdaModel.load(model_path)
# lda_dictionary = Dictionary().load(dictionary_path)
# print('loaded topic model')
# #####

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

# loss_function = torch.nn.BCEWithLogitsLoss()
# loss_function = torch.nn.KLDivLoss()
# loss_function = torch.nn.L1Loss()
loss_function = torch.nn.CrossEntropyLoss()

def model_def(model_name):
       print(model_name)
       #model_ft = models.resnet50(pretrained=True)
       #set_parameter_requires_grad(model_ft, False)
       #num_ftrs = model_ft.fc.in_features

       # num_ftrs = 2048
       # model_ft = nn.Sequential(
       #     nn.Dropout(0.7),
       #     nn.ReLU(True),
       #     nn.Linear(num_ftrs, 1024),
       #     nn.Dropout(0.7),
       #     nn.ReLU(True),
       #     nn.Linear(1024, num_topics))
       ##### new model
       num_ftrs = 2048
       model_ft = nn.Sequential(
           nn.Dropout(0.5),
           nn.Linear(num_ftrs, 1024),
           nn.ReLU(True),
           nn.Linear(1024, num_topics))
       print(model_ft)
       return model_ft


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        #print(group['params'])
        for param in group['params']:
            # print(param.grad.data)
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

def kl(input, target):
    out = F.kl_div(input, target)

    return out
def l1(input,target):
    out = torch.mean(torch.abs(input - target))
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
    top3 = AverageMeter()
    final_data =[]
    json_file = osp.join(fake_dir, str(epoch)+'_captions_val2017_fakecap_results.json')

    with torch.no_grad():
        for ii, (inputs,labels,img_id) in enumerate(testloader):
            ims = inputs.cuda()
            target = labels.cuda(async=True)
            # target = target/target.sum()
            outputs = model(ims)  # compute output
            # outputs = F.relu(outputs)
            # outputs = F.softmax(outputs,dim=1)    #### comment with l1 loss
            # outputs = outputs / outputs.sum()
            # l1_val = l1(outputs,target)
            # mse_val = mse(outputs, target)
            # # outputs[outputs == 0] = 1
            # kl_div = kl(outputs.log(), target)
            # top1.update(mse_val.item(), ims.size(0))
            # top2.update(kl_div.item(), ims.size(0))
            # top3.update(l1_val.item(), ims.size(0))

            prob_topics = outputs.cpu().detach().numpy().squeeze()
            gen_captions = tm_utils.get_relevant_words(prob_topics)
            # print(gen_captions, img_id)
            data = {"image_id": int(img_id[0]), "caption": str(gen_captions)}
            final_data.append(data)
        with open(json_file, 'w') as outfile:
            json.dump(final_data, outfile, indent=3)
    print('done')
    bleu_eval(json_file)
    # print(' * Average MSE {0:.3f}'.format(top1.avg))
    # print(' * Average KL {0:.3f}'.format(top2.avg))
    # print(' * Average L1 {0:.3f}'.format(top3.avg))
    return top1.avg

def trainmodel(dataloader, testloader, model,optimizer,num_epochs,args):

        iters = 0
        writer = SummaryWriter()
        index = np.arange(512)
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)
            #model.to(device)
            model.train()  # Set model to training mode
            lambda1 = 0.001
            running_loss = 0.0

             # Iterate over data.
            for i, (inputs, labels,img_id) in enumerate(dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()
                outputs = model(inputs)
                # outputs = F.relu(outputs)
                # outputs = F.softmax(outputs,dim=1)
                # l1_norm = torch.sum(torch.abs(outputs))
                outputs = F.log_softmax(outputs,dim=1)
                # outputs = outputs/outputs.sum()
                # outputs = F.sigmoid(outputs)
                ce_loss = -1*torch.sum(labels*outputs)
                # kl_loss = loss_function(outputs.log(),labels)
                # l1_loss = loss_function(outputs, labels)

                # loss = ce_loss+lambda1*l1_norm
                loss = ce_loss
                loss.backward()
                clip_gradient(optimizer, 0.1)
                optimizer.step()
                #print(preds)
                #raw_input('enter')


                running_loss+=loss.item()
                # Check how the generator is doing by saving G's output on fixed_noise
                if (iters % 200 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
                    count = iters+1
                    print(
                        "[Epoch %d/%d] [Batch %d/%d] [L1 loss: %f]"
                        % (epoch, num_epochs, i, len(dataloader), running_loss/count)
                    )
                    writer.add_scalar('L1_loss', running_loss/count, iters)
                iters += 1

            output_file = os.path.join(model_dir, "model_{}.pt".format(epoch))
            print("Saving model to {}".format(output_file))
            torch.save(model.state_dict(), output_file)

            #test_acc = testmodel(args,testloader,epoch,model)

def main(args):
    feat_extract = False
    model = model_def(args.model_name)
    gpus = [2,3]
    if len(gpus) > 1:
        model = torch.nn.DataParallel(model)

    model.to(device)
    # dataset = MSCocoDataset
    feat_dataset = MSCocoFeatures
    num_epochs = args.epochs

    image_size = 224 ## for resnet
    # transform_train = transforms.Compose([transforms.Resize([image_size, image_size]),transforms.ToTensor()])
    # transform_test =  transforms.Compose([transforms.Resize([image_size, image_size]),transforms.ToTensor()])
    transform_feat = transforms.Compose([transforms.ToTensor()])
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0005)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr)
    # train_data = dataset(data_dir, topic_dir, mode='train',transform=transform_train)   #### For MSCOCODataset
    train_data = feat_dataset(feat_dir, topic_dir, mode='train',transform = transform_feat)  #### For MSCOCODataset
    train_loader = torch.utils.data.DataLoader(train_data,batch_size=args.batch_size, shuffle=True, num_workers=32, pin_memory=True)
    print('train data loaded')
    # # test_data =dataset(data_dir, topic_dir, mode='val',transform=transform_test)
    # #len_data = test_data.test_len
    test_data = feat_dataset(feat_dir, topic_dir, mode='val',transform = transform_feat)  #### For MSCOCODataset
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=32,
                                              pin_memory=True)
    print('test data loaded')
    # test_acc = testmodel(test_loader, 60)
    trainmodel(train_loader, test_loader, model,optimizer,num_epochs,args)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train and Test Img Topic Models')
    parser.add_argument('--batch_size',type=int,default=64)
    parser.add_argument('--lr',type = float,default=0.00001)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--model_name', type=str, default='resnet')
    parser.add_argument('--fold', type=int, default=1)
    args = parser.parse_args()
    main(args)
