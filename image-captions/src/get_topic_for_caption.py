from __future__ import division
import numpy as np
import pickle
import os
import os.path as osp
import json
import torch
from matplotlib import pyplot as plt
from gensim.corpora.dictionary import Dictionary

this_dir = osp.dirname(osp.realpath((__file__)))


#### Uncomment this portion for debugging
# topic_matrix = np.load(osp.join(this_dir, 'dataset_2014', 'lda_512_matrix_capt_vocab.npy'))
# dictionary_path = osp.join(this_dir, 'dataset_2014', 'captions_dictionary_nobelow2.dict')
# data_dir=  osp.join(this_dir,'dataset_2014')
# # caption_file = '/raid/IC-GAN/coco/Son/data/coco2014_filtered_tokens_indices.pkl'
# # dictionary_path = '/raid/IC-GAN/coco/Son/data/LDA/captions_dictionary_nobelow2.dict'
# dictionary = Dictionary().load(dictionary_path)

# # dictionary.add_documents(['<start>'.split()])
# # dictionary.add_documents(['<pad>'.split()])
# word_map = dictionary.token2id
# rev_word_map = {v: k for k, v in word_map.items()}
# vocab_size = len(word_map)

# # captions = pickle.load(open(caption_file,'rb'))

# # topic_matrix = np.load('/raid/IC-GAN/coco/Son/data/lda_models/lda_512_matrix_capt_vocab.npy')
# assert topic_matrix.shape[1] == vocab_size
# topic_dist = pickle.load(open(osp.join(data_dir,'capt_topicdist_matrix.pkl'),'rb'))
def get_top_words_for_caption(topic_matrix, topics):
    topics_terms = topic_matrix

    topic_dist = topics
    word_dist = np.multiply(np.transpose(topics_terms), topic_dist)
    word_dist = np.transpose(word_dist)

    word_prob = np.sum(word_dist, axis=0)
    # print(word_prob.shape)
    # print(word_prob[171])
    # word_sorted = np.argsort(word_prob)  # ascending order
    num_words = 52
    word_sorted = word_prob.argsort()[::-1][:num_words]
    # print(word_sorted)
    # input('enter')
    relevant_words = []
    for idx in word_sorted:
        # print(model.id2word[idx])
        # relevant_words.append(str(model.id2word[idx]))
        # relevant_words.append(str(dictionary[idx]))
        relevant_words.append(idx)
    generated_caption = str(' '.join(relevant_words))
    # print(generated_caption)
    return relevant_words
# top_topic_words={}
# count=0
# for id in topic_dist:
#     caption_id = id
#     count+=1
#     print(count)
#     topics = topic_dist[caption_id]
#     top_words = get_top_words_for_caption(topic_matrix, topics)
#     print(top_words)
#     input('enter')
#     top_topic_words[caption_id] = top_words

# with open(osp.join(data_dir,'top_words_for_topics.pkl'), 'wb') as outfile:
#     pickle.dump(top_topic_words, outfile, protocol=2)


#
# caption_id = 123
# test_caption = captions[caption_id]
# caplen = len(test_caption)
#
# one_hot_caption = np.zeros(vocab_size)
#
# for idx in test_caption:
#     one_hot_caption[idx] += idx/caplen
#
# topic_for_caption = np.multiply(topic_matrix, one_hot_caption)
# topic_for_caption = np.sum(topic_for_caption,axis=1)
# index = np.arange(512)
# print(test_caption)
# plt.bar(index,topic_for_caption)
# plt.show()

def get_topics(test_caption, dictionary, topic_matrix):
    word_map = dictionary.token2id
    # rev_word_map = {v: k for k, v in word_map.items()}
    vocab_size = len(word_map) - 2
    assert topic_matrix.shape[1] == vocab_size

    caplen = len(test_caption)

    one_hot_caption = np.zeros(vocab_size)

    for idx in test_caption:
        one_hot_caption[idx] += idx / caplen

    topic_for_caption = np.multiply(topic_matrix, one_hot_caption)
    topic_for_caption = np.sum(topic_for_caption, axis=1)

    return topic_for_caption



