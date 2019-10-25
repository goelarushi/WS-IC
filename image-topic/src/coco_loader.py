import os
import torch
import pickle
import numpy as np
from PIL import Image
import json
import h5py


class MSCocoDataset(torch.utils.data.Dataset):

    def __init__(self, data_dir, topic_dir, mode='train', transform=None):
        self.data_dir = data_dir
        self.csv_dir = topic_dir
        self.mode = mode
        self._transform = transform
        self.image_data = []
        self.topic_label = []
        self.image_id = []
        self.img_dir = os.path.join(self.data_dir, self.mode + '2017')  # +'_subset')
        self.csv_dir = os.path.join(self.csv_dir, '512topics_' + self.mode + '2017' + '.json')

        csv_file = json.load(open(self.csv_dir))

        for id in range(len(csv_file)):
            cur_ann = csv_file[id]
            img_id = str((cur_ann['img_id']))
            img_id = img_id.zfill(12)
            topic = cur_ann['topic_dist']
            topics = [float(i) for i in topic]
            topics = torch.Tensor(topics)
            # print(im_info[0], img_id)
            imgname = os.path.join(self.img_dir, str(img_id) + '.jpg')
            if os.path.isfile(imgname):
                raw_img = Image.open(imgname)
                if (len(raw_img.mode) != 3):
                    rgb_img = raw_img.convert('RGB')
                    if self._transform is not None:
                        proc_img = self._transform(rgb_img)
                    # print(proc_img, proc_img.shape)
                    self.image_data.append(proc_img)
                    self.topic_label.append(topics)
                    self.image_id.append(img_id)

    def __len__(self):
        ## total number of images
        return len(self.image_data)

    def __getitem__(self, index):
        return self.image_data[index], self.topic_label[index], self.image_id[index]


class MSCocoFeatures(torch.utils.data.Dataset):

    def __init__(self, data_dir, topic_dir, mode='train', transform=None):
        self.data_dir = data_dir
        self.csv_dir = topic_dir
        self.mode = mode
        self.transform = transform
        self.image_data = []
        self.topic_label = []
        self.image_id = []
        self.img_dir = os.path.join(self.data_dir, self.mode + '2017')  # +'_subset')
        self.csv_dir = os.path.join(self.csv_dir, '512topics_' + self.mode + '2017' + '.json')

        csv_file = json.load(open(self.csv_dir))

        for id in range(len(csv_file)):
            cur_ann = csv_file[id]
            img_id = str((cur_ann['img_id']))
            img_id = img_id.zfill(12)
            topic = cur_ann['topic_dist']
            topics = [float(i) for i in topic]
            topics = torch.Tensor(topics)
            # print(im_info[0], img_id)
            imgfile = os.path.join(self.img_dir, str(img_id) + '.npy')
            if os.path.isfile(imgfile):
                v_feats = np.load(imgfile)
                ### l1 norm features
                # v_feats = v_feats/v_feats.sum()
                v_feats = torch.Tensor(v_feats)
                self.image_data.append(v_feats)
                self.topic_label.append(topics)
                self.image_id.append(img_id)

    def __len__(self):
        ## total number of images
        return len(self.image_data)

    def __getitem__(self, index):
        return self.image_data[index], self.topic_label[index], self.image_id[index]


# cat_dict={'person':0, 'bicycle':1, 'car':2, 'motorcycle':3, 'airplane':4, 'bus':5, 'train':6, 'truck':7, 'boat':8, 'traffic light':9, 'fire hydrant':10, 'stop sign':11, 'parking meter':12,
#           'bench':13, 'bird':14, 'cat':15, 'dog':16, 'horse':17, 'sheep':18, 'cow':19, 'elephant':20, 'bear':21, 'zebra':22, 'giraffe':23, 'backpack':24, 'handbag':25, 'tie':26, 'suitcase':27,
#           'frisbee':28, 'skis':29, 'snowboard':30, 'sports ball':31, 'kite':32, 'baseball bat':33, 'baseball glove':34, 'skateboard':35, 'surfboard':36, 'tennis racket':37, 'bottle':38, 'wine glass':39, 'cup':40,
#           'fork':41, 'knife':42, 'spoon':43, 'bowl':44, 'banana':45, 'apple':46, 'sandwich':47, 'orange':48, 'broccoli':49, 'carrot':50, 'hot dog':51, 'pizza':52, 'donut':53, 'cake':54, 'chair':55, 'couch':56,
#           'potted plant':57, 'bed':58, 'dining table':59, 'toilet':60, 'tv':61, 'laptop':62, 'mouse':63, 'remote':64, 'keyboard':65, 'cell phone':66, 'microwave':67, 'oven':68,
#           'toaster':69, 'sink':70, 'refrigerator':71, 'book':72, 'clock':73, 'vase':74, 'scissors':75, 'teddy bear':76, 'hair drier':77, 'toothbrush':78,'umbrella':79}
class MSCocoObjectDataset(torch.utils.data.Dataset):

    def __init__(self, data_dir, csv_dir, topic_dir, mode='train', transform=None):
        self.data_dir = data_dir
        self.csv_dir = csv_dir
        self.topic_dir = topic_dir
        self.mode = mode
        self._transform = transform
        self.image_data = []
        self.image_id = []
        self.topic_label = []
        self.cat_label = []
        self.img_dir = os.path.join(self.data_dir, self.mode + '2017')  # +'_subset')
        self.csv_dir = os.path.join(self.csv_dir, 'instances_' + self.mode + '2017' + '.json')
        self.topic_dir = os.path.join(self.topic_dir, '512topics_' + self.mode + '2017' + '.json')
        num_classes = 90
        csv_file = json.load(open(self.csv_dir))
        topic_file = json.load(open(self.topic_dir))
        print(csv_file.keys())
        annotations = csv_file['annotations']
        img_objs = {}
        for ann in annotations:
            img_id = ann['image_id']
            obj = ann['category_id'] - 1

            if img_id not in img_objs:
                img_objs[img_id] = []
            img_objs[img_id].append(obj)

        im_info = csv_file['images']
        category = csv_file['categories']
        for id in range(len(topic_file)):
            cur_ann = topic_file[id]
            img_id = str((cur_ann['img_id']))
            try:
                label = img_objs[int(img_id)]
                # print('found')
            except:
                # print('objects missing for image id', img_id)
                continue
            img_id = img_id.zfill(12)
            topic = cur_ann['topic_dist']
            topics = [float(i) for i in topic]
            topics = torch.Tensor(topics)
            # print(im_info[0], img_id)
            imgfile = os.path.join(self.img_dir, str(img_id) + '.npy')
            one_hot_label = np.zeros(num_classes)

            one_hot_label[label] = 1
            one_hot_label = torch.Tensor(one_hot_label)
            if os.path.isfile(imgfile):
                v_feats = np.load(imgfile)
                ### l1 norm features
                # v_feats = v_feats/v_feats.sum()
                v_feats = torch.Tensor(v_feats)
                self.image_data.append(v_feats)
                self.topic_label.append(topics)
                self.image_id.append(img_id)
                self.cat_label.append(one_hot_label)

        # print(self.image_id)

    def __len__(self):
        ## total number of images
        return len(self.image_data)

    def __getitem__(self, index):
        return self.image_data[index], self.topic_label[index], self.image_id[index], self.cat_label[index]


class MSCocoBottomUpDataset(torch.utils.data.Dataset):
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
        # assert self.split in {'TRAIN', 'VAL'}

        # Open hdf5 file where images are stored
        feat_file = os.path.join(data_folder, self.split + '36.hdf5')
        self.train_hf = h5py.File(feat_file)
        self.train_features = self.train_hf['image_features']
        # self.val_hf = h5py.File(data_folder + '/val36.hdf5', 'r')
        # self.val_features = self.val_hf['image_features']
        self.train_topics = json.load(open(os.path.join(data_folder, '512topics_' + self.split + '2017' + '.json')))
        # self.val_topics = json.load(open(os.path.join(data_folder, '512topics_' + 'val' + '2017' + '.json')))
        self.image_features = []
        self.all_topics = []
        self.image_ids = []

        # Load bottom up image features distribution
        with open(os.path.join(data_folder, self.split + '_GENOME_DETS_' + data_name + '.json'), 'r') as j:
            self.objdet = json.load(j)
        for id in range(len(self.objdet)):
            objdet = self.objdet[id]
            img = torch.FloatTensor(self.train_features[objdet[1]])
            img_id = objdet[2]
            topic_id = next(f for f in self.train_topics if f['img_id'] == img_id)
            topic = topic_id['topic_dist']
            topics = torch.Tensor([float(t) for t in topic])
            self.image_features.append(img)
            self.image_ids.append(img_id)
            self.all_topics.append(topics)

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform
        self.dataset_size = len(self.objdet)

    def __getitem__(self, i):
        return self.image_features[i], self.image_ids[i], self.all_topics[i]

    def __len__(self):
        return self.dataset_size


import pandas as pd
import random


class Vision2Topics(torch.utils.data.Dataset):
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

        with open(os.path.join(data_folder, 'capt_topicdist_matrix' + '.pkl'), 'rb') as j:
            self.all_topics = pickle.load(j)
        # Load original coco annotations for splits
        with open(os.path.join(self.coco_ann_data, self.split.lower() + '2014' + '.json'), 'r') as j:
            self.cocodata = json.load(j)
        self.coco_annotations = pd.DataFrame(self.cocodata['annotations'])

        # Total number of datapoints
        self.dataset_size = len(self.image_ids)
        print('files loaded')

    def __getitem__(self, i):
        self.ids = self.image_ids[i]
        self.features = self.image_features[i]
        self.topics = []
        id_annotations = self.coco_annotations[self.coco_annotations['image_id'] == self.ids].to_dict('records')
        for ann in id_annotations:
            caption_id = ann['id']
            if (caption_id != 626307 and caption_id != 598973):
                self.topics.append(self.all_topics[caption_id])

        self.topics = np.stack(self.topics)
        # self.random_topic = torch.Tensor(random.sample(self.topics,1)[0])
        # self.random_topic = torch.Tensor(self.topics[0])
        self.threshold_topic = self.topics[0]
        # self.threshold_topic[self.threshold_topic > 0.04] = 1
        # self.threshold_topic[self.threshold_topic < 0.04] = 0

        self.random_topic = torch.Tensor(self.threshold_topic)

        return self.features, self.ids, self.random_topic

    def __len__(self):
        return self.dataset_size


class Vision2MultipleTopics(torch.utils.data.Dataset):
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
        self.cpi = 5

        # Open hdf5 file where images are stored
        self.image_features = torch.load(os.path.join(data_folder, self.split + '_image_features.pth'))
        self.image_ids = torch.load(os.path.join(data_folder, self.split + '_image_ids.pth'))
        self.coco_ann_data = os.path.join(data_folder, '..', '..', '..', 'coco', 'annotations', '2014')

        with open(os.path.join(data_folder, 'capt_topicdist_matrix' + '.pkl'), 'rb') as j:
            self.all_topics = pickle.load(j)
        # Load original coco annotations for splits
        with open(os.path.join(self.coco_ann_data, self.split.lower() + '2014' + '.json'), 'r') as j:
            self.cocodata = json.load(j)
        self.coco_annotations = pd.DataFrame(self.cocodata['annotations'])

        # Total number of datapoints
        self.dataset_size = len(self.image_ids)
        print('files loaded')

    def __getitem__(self, i):
        self.ids = self.image_ids[i]
        self.features = self.image_features[i]
        self.topics = []
        id_annotations = self.coco_annotations[self.coco_annotations['image_id'] == self.ids].to_dict('records')
        if (len(id_annotations) > 5):
            id_annotations = id_annotations[0:5]
        for ann in id_annotations:
            caption_id = ann['id']
            if (caption_id != 626307 and caption_id != 598973):
                self.topics.append(self.all_topics[caption_id])

        self.topics = np.stack(self.topics)
        if(self.topics.shape[0]==4):
            self.topics = np.vstack((self.topics, self.topics[1]))
        self.topics= torch.Tensor(self.topics)

        if(self.split == 'TRAIN'):# or self.split is 'VAL'):
            self.train_features = self.features.expand(self.cpi, 36, 2048)
            return self.train_features, self.ids, self.topics
        else:
            return self.features, self.ids, self.topics

    def __len__(self):
        return self.dataset_size
