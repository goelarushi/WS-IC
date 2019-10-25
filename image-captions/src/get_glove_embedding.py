import os
import os.path as osp
import numpy as np
import pickle
this_file = osp.dirname(osp.realpath(__file__))
embedding_file = osp.join(this_file, 'dataset_2014','glove.840B.300d.txt')
import bcolz
import json
words = []
idx = 0
word2idx = {}
vectors = bcolz.carray(np.zeros(1), rootdir=osp.join(this_file, 'dataset_2014','840B.300d.dat'), mode='w')

def txt2vec():
    with open(embedding_file, 'rb') as f:
        for l in f:
            line = l.split()
            word = line[0]
            words.append(word)
            word2idx[word] = idx
            idx += 1
            vect = np.array(line[1:]).astype(np.float)
            vectors.append(vect)
        print(idx)

    vectors = bcolz.carray(vectors[1:].reshape((-1, 300)),rootdir=osp.join(this_file, 'dataset_2014','840B.300d.dat'), mode='w')
    vectors.flush()
    pickle.dump(words, open(osp.join(this_file, 'dataset_2014','840B.300d_words.pkl'), 'wb'))
    pickle.dump(word2idx, open(osp.join(this_file, 'dataset_2014','840B.300d_idx.pkl'), 'wb'))

def glove_dict():
    vectors = bcolz.open(osp.join(this_file, 'dataset_2014','840B.300d.dat'))[:]
    words = pickle.load(open(osp.join(this_file, 'dataset_2014','840B.300d_words.pkl'), 'rb'))
    word2idx = pickle.load(open(osp.join(this_file, 'dataset_2014','840B.300d_idx.pkl'), 'rb'))
    glove = {w: vectors[word2idx[w]] for w in words}

    pickle.dump(glove, open(osp.join(this_file, 'dataset_2014', 'glove.pkl'),'wb'))

# def glove_weights():
glove = pickle.load(open(osp.join(this_file, 'dataset_2014', 'glove.pkl'),'rb'))
wordmap = json.load(open(osp.join(this_file, 'dataset_2014', 'WORDMAP_coco_5_cap_per_img_5_min_word_freq.json')))
vocab_size = len(wordmap)
weights_matrix = np.zeros((vocab_size, 300))
words_found = 0

for i, word in enumerate(wordmap):
    idx = wordmap[str(word)]
    try:
        weights_matrix[idx] = glove[str(word)]
        words_found+=1
    except KeyError:
        weights_matrix[idx] = np.random.normal(scale=0.6, size=(300,))
        print(idx)
pickle.dump(weights_matrix, open(osp.join(this_file, 'dataset_2014','glove_weights_matrix.pkl'), 'wb'))
print(words_found)