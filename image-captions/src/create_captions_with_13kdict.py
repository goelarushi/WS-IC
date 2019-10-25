import os
import os.path as osp

import pickle
from gensim.corpora.dictionary import Dictionary


this_dir = osp.dirname(osp.realpath(__file__))
caption_path = '/raid/IC-GAN/coco/Son/data/coco2014_filtered_tokens_indices.pkl'
output_folder = osp.join(this_dir,'dataset_2014')
dictionary_path = '/raid/IC-GAN/coco/Son/data/LDA/captions_dictionary_nobelow2.dict'
dictionary = Dictionary().load(dictionary_path)
dictionary.add_documents(['<start>'.split()])
dictionary.add_documents(['<pad>'.split()])
id_dict = dictionary.token2id
new_captions = {}
caption_lengths = {}
with open(caption_path, 'rb') as j:
    caption_data = pickle.load(j)
max_len=52

for id in caption_data:
    caption_id = id
    caption = caption_data[caption_id]
    caption.insert(0, id_dict['<start>'])   ### insert(location, value)
    caption.insert(len(caption), id_dict['<eoc>'])
    caption_length = len(caption)
    for i in range(len(caption), max_len):
        caption.insert(i,id_dict['<pad>'])
    new_captions[caption_id] = caption
    caption_lengths[caption_id] = caption_length

with open(osp.join(output_folder,'captions_with_13k_dict.pkl'), 'wb') as outfile:
    pickle.dump(new_captions,outfile, protocol=2)

with open(osp.join(output_folder,'caption_lengths_with_13k_dict.pkl'), 'wb') as outfile:
    pickle.dump(caption_lengths,outfile, protocol=2)

