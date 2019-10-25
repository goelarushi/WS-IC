import numpy as np
import pickle
import os
import os.path as osp
import json
import torch
this_dir = osp.dirname(osp.realpath((__file__)))

data_folder = osp.join(this_dir, 'dataset_2014')  # folder with data files saved by create_input_files.py
data_name = 'coco_5_cap_per_img_5_min_word_freq'  # base name shared by data files


# Load word map (word2ix)
word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
with open(word_map_file, 'r') as j:
    word_map = json.load(j)
rev_word_map = {v: k for k, v in word_map.items()}
vocab_size = len(word_map)

test_image_id = 112410
img_id = torch.load(osp.join(data_folder,'TRAIN_image_ids.pth'))
img_index = np.where(np.array(img_id) == test_image_id)[0]

captions_file = json.load(open(osp.join(data_folder, 'TRAIN_CAPTIONS_coco_5_cap_per_img_5_min_word_freq.json')))

captions = captions_file[(img_index*5):((img_index*5)+5)]

img_captions = list(
            map(lambda c: [rev_word_map[w] for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],
                captions))  # remove <start> and pads
img_caps = [' '.join(c) for c in img_captions]
img_caps = np.expand_dims(img_caps,0)

print(img_caps)