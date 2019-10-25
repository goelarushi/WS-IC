from gensim import models
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from gensim.utils import simple_preprocess
from gensim.corpora.dictionary import Dictionary
import numpy as np
import os
import os.path as osp

def print_topics(lda_model, num_topics=-1, num_words=10):
    for idx, topic in lda_model.print_topics(num_topics=num_topics, num_words=num_words):
        print('Topic: {} \tWords: {}'.format(idx, topic))


def print_topic_dist(lda_results, lda_model):
    for index, score in lda_results:
        print("Score: {}\t Topic: {}\t({})".format(score, index, lda_model.print_topic(index, 10)))


def preprocess(doc, stemmer=None):
    tokens = []
    tmp = simple_preprocess(doc)
    for token in tmp:
        if token not in en_stopwords:
            if stemmer is not None:
                tokens.append(stemmer.stem(lemmatizer.lemmatize(token)))
            else:
                tokens.append(lemmatizer.lemmatize(token))
    return tokens


def get_relevant_topics_dist(lda_model, doc, dictionary, print_result=False):
    if isinstance(doc, str):
        doc = dictionary.doc2bow(preprocess(doc))
    lda_results = lda_model[doc]
    if isinstance(lda_results, tuple):
        lda_results = lda_results[0]
    sorted_results = sorted(lda_results, key=lambda tup: -1 * tup[1])
    if print_result:
        print_topic_dist(sorted_results, lda_model)

    return sorted_results


def get_topics_dist(lda_model, doc, dictionary):
    topic_dist = get_relevant_topics_dist(lda_model, doc, dictionary, print_result=False)
    n_topics = lda_model.num_topics

    tmp = list_to_dict(topic_dist)
    all_topics = []
    for i in range(n_topics):
        all_topics.append(tmp.get(i, 0))
    return all_topics


def list_to_dict(list_of_tuples):
    tmp = {}
    for idx, value in list_of_tuples:
        tmp[idx] = value
    return tmp


def get_word_prob(lda_model, axis=1):
    # topics_terms = lda_model.state.get_lambda()
    topics_terms = model
    topics_terms_proba = np.apply_along_axis(lambda x: x / x.sum(), axis, topics_terms)
    return topics_terms_proba

'''
init
'''
lemmatizer = WordNetLemmatizer()
en_stopwords = stopwords.words('english')
this_dir = osp.dirname(osp.realpath((__file__)))

'''
directories
'''
data_folder = osp.join(this_dir,'..','image-captioning-bottom-up-top-down-master', 'dataset_2014')
# model_path = osp.join(this_dir, '..', 'lda_models', 'lda_512.gensim')
# dictionary_path = osp.join(this_dir, '..', 'lda_512topics', 'lda_dictionary_nobelow2.dict')
dictionary_path = osp.join(data_folder, 'captions_dictionary_nobelow2.dict')
model_path =  osp.join(data_folder, 'lda_512_matrix_capt_vocab.npy')
'''
load data
'''
# model = models.LdaModel.load(model_path)
model = np.load(model_path)
dictionary = Dictionary().load(dictionary_path)
print('loaded topic model')
#####


def get_word_prob_given_topics(topic_dist, topics_terms):
    word_dist = np.multiply(np.transpose(topics_terms), topic_dist)
    word_dist = np.transpose(word_dist)
    word_prob = np.sum(word_dist, axis=0)
    word_sorted = word_prob.argsort()[::-1]

    return word_prob, word_sorted

def get_words_threshold(lda_model, word_prob, word_sorted, threshold):
    words = []
    idx = 0
    while True:
        word_idx = word_sorted[idx]
        prob = word_prob[word_idx]
        if prob >= threshold:
            # words.append(lda_model.id2word[word_idx])
            words.append(dictionary[word_idx])
            idx += 1
        else:
            break
    return ' '.join(words)



def get_threshold_words(all_topics):


    # word probability (matrix [#topics, #words])

    # topics_terms = get_word_prob(model)  # [#topics, #words]
    topics_terms = model
    topic_dist = all_topics
    word_prob,word_sorted = get_word_prob_given_topics(topic_dist,topics_terms)

    threshold = 0.02

    # print("Processing for threshold ", threshold)

    capt = get_words_threshold(model, word_prob, word_sorted, threshold)
    return capt


def get_relevant_words(all_topics):
    '''
    running examples
    '''
    # print topics
    # print_topics(model, num_topics=-1, num_words=10)

    # get topic distribution (relevant topics only)
    # doc = 'a guy is waterboarding in the ocean on a windy day.'
    # doc = 'A Honda motorcycle parked in front of a garage.'
    # doc = 'a man in a jacket and hat looks at the camera.'
    # topics = get_relevant_topics_dist(model, doc, dictionary, print_result=True)
    # get topic distribution (all topics)
    # all_topics = get_topics_dist(model, doc, dictionary)
    # print(all_topics)
    # a = np.array(all_topics)
    # print(a.shape)

    # word probability (matrix [#topics, #words])
    # topics_terms = get_word_prob(model)   # [#topics, #words]


    topics_terms = model
    # words probability with a given topic distribution
    # topic_dist = np.array([0.1139166, 0, 0, 0.3706041, 0.13249765, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.34891573, 0])
    topic_dist = all_topics
    word_dist = np.multiply(np.transpose(topics_terms), topic_dist)
    word_dist = np.transpose(word_dist)

    word_prob = np.sum(word_dist, axis=0)
    # print(word_prob.shape)
    # print(word_prob[171])
    # word_sorted = np.argsort(word_prob)  # ascending order
    num_words = 10
    word_sorted = word_prob.argsort()[::-1][:num_words]
    relevant_words = []
    for idx in word_sorted:
        # print(model.id2word[idx])
        # relevant_words.append(str(model.id2word[idx]))
        relevant_words.append(str(dictionary[idx]))

    generated_caption = str(' '.join(relevant_words))

    return generated_caption
