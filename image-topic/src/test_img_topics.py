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
from sklearn.metrics import average_precision_score
import torch.nn.functional as F
from numpy import inf
from sklearn.metrics import precision_recall_curve
from torchvision.models import DenseNet
from os.path import join as pjoin
from tensorboardX import SummaryWriter
from eval_coco_captions import bleu_eval

from torch.autograd import Variable
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from skimage.transform import resize
import pickle
import topicmodel_utils as tm_utils
os.environ["CUDA_VISIBLE_DEVICES"]="2,3"
use_gpu = torch.cuda.is_available()
# torch.manual_seed(1)
# torch.cuda.manual_seed(1)
# np.random.seed(1)
this_dir = osp.dirname(osp.realpath((__file__)))
num_topics =512

lib_path = osp.join(this_dir,'..','..','src')
sys.path.insert(0,lib_path)
from coco_loader import MSCocoDataset, MSCocoFeatures

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
## DATA DIR ##

data_dir = osp.join(this_dir,'..','..', 'coco/images')
topic_dir = osp.join(this_dir, '..', 'lda_512topics')
feat_dir = osp.join(this_dir, '..', 'resnet50')

model_dir = osp.join(this_dir,'..', 'celoss+newmodelall512_topic_models')
fake_dir = osp.join(this_dir, '..', '512fake_captions')
# if not os.path.isdir(model_dir):
#     os.mkdir(model_dir)

if not os.path.isdir(fake_dir):
    os.mkdir(fake_dir)
import json
# json_file= osp.join(fake_dir, 'random_captions_val2017_fakecap_results.json')
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

loss_function = torch.nn.BCEWithLogitsLoss()
loss_function = torch.nn.KLDivLoss()
#### Topic model things
from gensim import models
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from gensim.utils import simple_preprocess
from gensim.corpora.dictionary import Dictionary
import numpy as np
import os
import os.path as osp
'''
directories
'''
model_path = osp.join(this_dir, '..', 'lda_models', 'lda_512.gensim')
dictionary_path = osp.join(this_dir, '..', 'lda_512topics', 'lda_dictionary_nobelow2.dict')

'''
load data
'''
lda_model = models.LdaModel.load(model_path)
lda_dictionary = Dictionary().load(dictionary_path)
print(len(lda_model.id2word), len(lda_dictionary))
id = [20,40,76,89,10000]
for n in id:
    print(lda_model.id2word[n])
    print(lda_dictionary[n])
print('loaded topic model')
####

def model_def(model_name):
    print(model_name)
    # model_ft = models.resnet50(pretrained=True)
    # set_parameter_requires_grad(model_ft, False)
    # num_ftrs = model_ft.fc.in_features
    #
    # num_ftrs = 2048
    # model_ft = nn.Sequential(
    #     nn.Dropout(0.7),
    #     nn.ReLU(True),
    #     nn.Linear(num_ftrs, 1024),
    #     nn.Dropout(0.7),
    #     nn.ReLU(True),
    #     nn.Linear(1024, num_topics))
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
            param.grad.data.clamp_(-grad_clip, grad_clip)


def load_model(model,epoch):
    gpus = [2, 3]
    if len(gpus) > 1:
        model = torch.nn.DataParallel(model)

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
    # json_file = osp.join(fake_dir, '_captions_val2017_fakecap_results.json')
    prob_json_file = osp.join(fake_dir, '512topics_train2017.json')
    with torch.no_grad():
        for ii, (inputs, labels, img_id) in enumerate(testloader):
            ims = inputs.cuda()
            target = labels.cuda(async=True)
            # target = target/target.sum()
            outputs = model(ims)  # compute output
            # outputs = np.random.rand(ims.size(0), num_topics)
            # outputs = torch.Tensor(outputs).cuda()
                    # print(outputs)
            # outputs = F.relu(outputs)
            outputs = F.softmax(outputs, dim=1)

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

            prob_topics = outputs.cpu().detach().numpy().squeeze()
            data = {"image_id": int(img_id[0]), "topic": prob_topics.tolist()}
            #
            topic_data.append(data)
            # print('Generating captions for top words')
            # wjson_file = osp.join(fake_dir, str(epoch) + '_topword_captions_val2017_fakecap_results.json')
            # gen_captions = tm_utils.get_relevant_words(prob_topics)
            # data = {"image_id": int(img_id[0]), "caption": str(gen_captions)}
            # wfinal_data.append(data)
            # print('Generating captions for threshold')
            # tjson_file = osp.join(fake_dir, str(epoch) + '_threshold_captions_val2017_fakecap_results.json')
            # gen_captions = tm_utils.get_threshold_words(prob_topics)
            # data = {"image_id": int(img_id[0]), "caption": str(gen_captions)}
            # tfinal_data.append(data)
        with open(prob_json_file, 'w') as outfile:
            json.dump(topic_data, outfile, indent=3)
        print(prob_json_file)
        # bleu_eval(wjson_file)
        # with open(tjson_file, 'w') as outfile:
        #     json.dump(tfinal_data, outfile, indent=3)
        # print(tjson_file)
        # bleu_eval(tjson_file)
    # print(' * Average MSE {0:.3f}'.format(top1.avg))
    # print(' * Average KL {0:.3f}'.format(top2.avg))
    # print(' * Average L1 {0:.3f}'.format(top3.avg))
    return top1.avg

def main(args):
    feat_extract = False
    model = model_def(args.model_name)
    dataset = MSCocoDataset
    feat_dataset = MSCocoFeatures
    image_size = 224 ## for resnet
    transform_test =  transforms.Compose([transforms.Resize([image_size, image_size]),transforms.ToTensor()])
    transform_feat = transforms.Compose([transforms.ToTensor()])
    # test_data =dataset(data_dir, topic_dir, mode='val',transform=transform_test)
    # #len_data = test_data.test_len
    # test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4,
    #                                           pin_memory=True)
    # print('test data loaded')
    test_data = feat_dataset(feat_dir, topic_dir, mode='train', transform=transform_feat)  #### For MSCOCODataset
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=16,
                                              pin_memory=True)
    print('test data loaded')
    test_acc = testmodel(args, test_loader, args.best_epoch, model)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train and Test Img Topic Models')
    parser.add_argument('--batch_size',type=int,default=1)
    parser.add_argument('--best_epoch', type=int, default=23)
    parser.add_argument('--model_name', type=str, default='resnet')
    args = parser.parse_args()
    main(args)
