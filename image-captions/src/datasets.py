import torch
from torch.utils.data import Dataset
import h5py
import json
import os
import numpy as np
import pickle
#

class CaptionAttentionDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, data_name, split, transform=None):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        self.split = split
        self.remove_captions = True
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        # Open hdf5 file where images are stored
        self.image_features = torch.load(os.path.join(data_folder, self.split+'_image_features.pth'))
        self.image_ids = torch.load(os.path.join(data_folder, self.split+'_image_ids.pth'))
        #self.all_topics = torch.load(os.path.join(data_folder, self.split+'_topics.pth'))
        # Captions per image
        self.cpi = 5
        with open(os.path.join(data_folder, self.split+'_caption_weights_48.pkl'),'r') as j:
            self.caption_weights = pickle.load(j)
        self.caption_weights = np.concatenate(self.caption_weights)


        with open(os.path.join(data_folder, self.split + '2014_lda512_10passes' + '.pkl'), 'r') as j:
            self.all_topics = pickle.load(j)
        # Load encoded captions
        with open(os.path.join(data_folder, self.split + '_CAPTIONS_' + data_name + '.json'), 'r') as j:
            self.captions = json.load(j)
        # Load caption lengths
        with open(os.path.join(data_folder, self.split + '_CAPLENS_' + data_name + '.json'), 'r') as j:
            self.caplens = json.load(j)

        # Total number of datapoints
        # self.dataset_size = len(self.captions)
        self.dataset_size = len(self.image_ids)
        print(self.dataset_size)
        print('files loaded')

    def __getitem__(self, i):
        self.ids = self.image_ids[i]
        self.features = self.image_features[i]
        self.topics = torch.Tensor(self.all_topics[self.ids])
        self.id_wght = next(item for item in self.caption_weights if item['image_id'] == self.ids)
        self.weight_index = np.where(self.id_wght['caption_weights']==max(self.id_wght['caption_weights']))[0]
        # print(len(self.weight_index))
        self.all_captions = torch.LongTensor(
            self.captions[((i) * self.cpi):(((i) * self.cpi) + self.cpi)])
        self.all_caplens = torch.LongTensor(
            self.caplens[((i) * self.cpi):(((i) * self.cpi) + self.cpi)])


        if(self.remove_captions and self.split is 'TRAIN'):
            self.weighted_captions =(np.delete(self.all_captions, self.weight_index[0], 0))
            self.weighted_caplens = (np.delete(self.all_caplens, self.weight_index[0], 0))
            # raw_input('enter')
            return self.features, self.all_captions, self.all_caplens, self.weighted_captions, self.weighted_caplens,  self.ids, self.topics
        else:
            # self.weighted_captions = (np.delete(self.all_captions, self.weight_index[0], 0))
            # self.weighted_caplens = (np.delete(self.all_caplens, self.weight_index[0], 0))
            return self.features, self.all_captions, self.all_caplens, self.ids, self.topics
        #
        # if self.split is 'TRAIN':
        #     return self.features, self.all_captions,self.all_caplens, self.ids, self.topics
        # else:
        #     return self.features,self.all_captions,self.all_caplens, self.ids, self.topics

    def __len__(self):
        return self.dataset_size


class CaptionFixedWeightDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, data_name, split, transform=None):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        # Open hdf5 file where images are stored
        self.image_features = torch.load(os.path.join(data_folder, self.split+'_image_features.pth'))
        self.image_ids = torch.load(os.path.join(data_folder, self.split+'_image_ids.pth'))
        #self.all_topics = torch.load(os.path.join(data_folder, self.split+'_topics.pth'))
        # Captions per image
        self.cpi = 5
        with open(os.path.join(data_folder, self.split+'_caption_weights_48.pkl'),'r') as j:
            self.caption_weights = pickle.load(j)
        self.caption_weights = np.concatenate(self.caption_weights)

        with open(os.path.join(data_folder, 'caption4_weights_39.pkl'),'r') as j:
            self.caption4_weights = pickle.load(j)
        self.caption4_weights = np.concatenate(self.caption4_weights)

        with open(os.path.join(data_folder, self.split + '2014_lda512_10passes' + '.pkl'), 'r') as j:
            self.all_topics = pickle.load(j)
        # Load encoded captions
        with open(os.path.join(data_folder, self.split + '_CAPTIONS_' + data_name + '.json'), 'r') as j:
            self.captions = json.load(j)
        # Load caption lengths
        with open(os.path.join(data_folder, self.split + '_CAPLENS_' + data_name + '.json'), 'r') as j:
            self.caplens = json.load(j)

        # Total number of datapoints
        # self.dataset_size = len(self.captions)
        self.dataset_size = len(self.image_ids)
        print(self.dataset_size)
        print('files loaded')

    def __getitem__(self, i):
        self.ids = self.image_ids[i]
        self.features = self.image_features[i]
        self.topics = torch.Tensor(self.all_topics[self.ids])
        self.id_wght = next(item for item in self.caption_weights if item['image_id'] == self.ids)
        self.weight_index = np.where(self.id_wght['caption_weights']==max(self.id_wght['caption_weights']))[0]

        self.id4_wght = next(item for item in self.caption4_weights if item['image_id'] == self.ids)['caption_weights']
        self.assign_weights = np.insert(self.id4_wght, self.weight_index[0], 1.)/2   ### to assign approx 0.5  weight to two relevant captions
        # print(len(self.weight_index))
        self.all_captions = torch.LongTensor(
            self.captions[((i) * self.cpi):(((i) * self.cpi) + self.cpi)])
        self.all_caplens = torch.LongTensor(
            self.caplens[((i) * self.cpi):(((i) * self.cpi) + self.cpi)])


        if(self.split is 'TRAIN'):

            return self.features, self.all_captions, self.all_caplens, self.assign_weights,  self.ids, self.topics
        else:

            return self.features, self.all_captions, self.all_caplens, self.ids, self.topics
        #
        # if self.split is 'TRAIN':
        #     return self.features, self.all_captions,self.all_caplens, self.ids, self.topics
        # else:
        #     return self.features,self.all_captions,self.all_caplens, self.ids, self.topics

    def __len__(self):
        return self.dataset_size

class CaptionDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, data_name, split, transform=None):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        # Open hdf5 file where images are stored
        self.image_features = torch.load(os.path.join(data_folder, self.split + '_image_features.pth'))
        self.image_ids = torch.load(os.path.join(data_folder, self.split + '_image_ids.pth'))
        # self.all_topics = torch.load(os.path.join(data_folder, self.split+'_topics.pth'))

        # Captions per image
        self.cpi = 5
        with open(os.path.join(data_folder, self.split + '2014_lda512_10passes' + '.pkl'), 'r') as j:
            self.all_topics = pickle.load(j)
        # Load encoded captions
        with open(os.path.join(data_folder, self.split + '_CAPTIONS_' + data_name + '.json'), 'r') as j:
            self.captions = json.load(j)
        # Load caption lengths
        with open(os.path.join(data_folder, self.split + '_CAPLENS_' + data_name + '.json'), 'r') as j:
            self.caplens = json.load(j)

        # Total number of datapoints
        self.dataset_size = len(self.captions)
        print('files loaded')

    def __getitem__(self, i):
        self.ids = self.image_ids[i // self.cpi]
        self.features = self.image_features[i // self.cpi]
        self.topics = torch.Tensor(self.all_topics[self.ids])
        self.caption = torch.LongTensor(self.captions[i])
        #
        self.caplen = torch.LongTensor([self.caplens[i]])

        if self.split is 'VAL' or self.split is 'TEST':
            self.all_captions = torch.LongTensor(
                self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])

        if self.split is 'TRAIN':

            return self.features, self.caption, self.caplen, self.ids, self.topics
        else:
            return self.features, self.caption, self.caplen, self.all_captions, self.ids, self.topics

    def __len__(self):
        return self.dataset_size
import pandas as pd
import os.path as os
from get_topic_for_caption import *
this_dir = osp.dirname(osp.realpath((__file__)))

class CaptionNewDictDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, data_name, split, transform=None):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        # Open hdf5 file where images are stored
        self.image_features = torch.load(os.path.join(data_folder, self.split + '_image_features.pth'))
        self.image_ids = torch.load(os.path.join(data_folder, self.split + '_image_ids.pth'))
        self.coco_ann_data = os.path.join(data_folder, '..', '..', '..', 'coco', 'annotations', '2014')


        # Captions per image
        self.cpi = 5
        # with open(os.path.join(data_folder, self.split + '2014_lda512_10passes' + '.pkl'), 'r') as j:
        #     self.all_topics = pickle.load(j)
        # self.topic_matrix = np.load(os.path.join(data_folder,'lda_512_matrix_capt_vocab.npy'))

        with open(os.path.join(data_folder, 'capt_topicdist_matrix' + '.pkl'), 'rb') as j:
            self.all_topics = pickle.load(j)
        # Load encoded captions
        with open(os.path.join(data_folder, 'captions_with_13k_dict'+ '.pkl'), 'rb') as j:
            self.captions = pickle.load(j)
        # Load caption lengths
        with open(os.path.join(data_folder, 'caption_lengths_with_13k_dict'+ '.pkl'), 'rb') as j:
            self.caplens = pickle.load(j)

        with open(os.path.join(self.coco_ann_data, self.split.lower() + '2014' + '.json'), 'r') as j:
            self.cocodata = json.load(j)
        self.coco_annotations = pd.DataFrame(self.cocodata['annotations'])

        self.pred_topic_file = json.load(open(os.path.join(this_dir,'..', 'MIML_Topic_Models','models_OT+KLloss_with5topics','10ot_'+self.split.lower()+'_topics_2014.json')))
        # Total number of datapoints
        self.dataset_size = len(self.image_ids)
        #self.dataset_size  =50
        print('files loaded')

    ##### For training denoising seq2seq model

    def add_noise(self,seq_len,word_keep,x):
        total_len = len(x)
        pad_len = total_len - seq_len
        x_wopad = x[:seq_len]
        x_wopad = x_wopad * np.random.choice(2, size=seq_len, p=[1-word_keep, word_keep])
        noisy_x_wpad = np.pad(x_wopad,(0,pad_len))

        return noisy_x_wpad

    def __getitem__(self, i):
        self.ids = self.image_ids[i]
    
        self.features = self.image_features[i]
        id_annotations = self.coco_annotations[self.coco_annotations['image_id'] == self.ids].to_dict('records')
        self.id_captions=[]
        self.id_caplens =[]
        self.id_top_words = []
        self.topics = []
        self.pred_topics = next(item for item in self.pred_topic_file if item['image_id']== self.ids)
        self.pred_topics = self.pred_topics['topic']
        self.pred_topics = torch.Tensor(self.pred_topics)
        # self.pred_topics[self.pred_topics < 0.060] = 0
        self.pred_topics = self.pred_topics.expand(5,512)
        if (len(id_annotations) > 5):
            id_annotations = id_annotations[0:5]
        for ann in id_annotations:
            caption_id = ann['id']
            if (caption_id != 626307 and caption_id != 598973):   ### bad topics
                self.topics.append(self.all_topics[caption_id])
            
        
            # self.topics = self.all_topics[caption_id]
            # top_words = get_top_words_for_caption(self.topic_matrix, self.topics)
            if (caption_id != 592559 and caption_id != 767357):   ### bad caption length
                # noisy_captions = self.add_noise(self.caplens[caption_id],0.5,self.captions[caption_id])
                # self.id_captions.append(noisy_captions)
                self.id_captions.append(self.captions[caption_id])
                self.id_caplens.append(self.caplens[caption_id])
            else:
                # noisy_captions = self.add_noise(self.caplens[caption_id]-1,0.5,self.captions[caption_id][0:52])
                # self.id_captions.append(noisy_captions[0:52])
                self.id_captions.append(self.captions[caption_id][0:52])
                self.id_caplens.append(self.caplens[caption_id]-1)
            # self.id_top_words.append(top_words)
        
        self.topics = np.stack(self.topics)
        if(self.topics.shape[0]==4):
            self.topics = np.vstack((self.topics, self.topics[1]))
        self.topics= torch.Tensor(self.topics)
        

        self.ungroup_captions = torch.LongTensor(np.stack(self.id_captions))
        self.ungroup_caplens =  torch.LongTensor(self.id_caplens)
        # self.ungroup_top_words = torch.LongTensor(np.stack(self.id_top_words))

        if self.split is 'TRAIN':# or self.split is 'TEST':
            # if (i%2 == 0):
            self.final_topics = self.topics
            # else:
                # self.final_topics = self.pred_topics

            return self.features, self.ungroup_captions, self.ungroup_caplens, self.ids, self.final_topics #self.ungroup_top_words
        else:
            self.final_topics = self.pred_topics
            return self.features, self.ungroup_captions, self.ungroup_caplens,self.ids, self.final_topics

    def __len__(self):
        return self.dataset_size
