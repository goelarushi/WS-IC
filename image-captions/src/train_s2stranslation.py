#from __future__ import unicode_literals, print_function, division
#
#import time
#import torch
#import os
#print(torch.__version__)
#import torch.backends.cudnn as cudnn
#import torch.optim
#import torch.utils.data
#import torch.nn.functional as F
#import sys
#import torchvision.transforms as transforms
#from torch import nn
#from torch.nn.utils.rnn import pack_padded_sequence
##from models_s2stranslation import EncoderRNN, DecoderRNN
#from datasets import *
#from utils import *
#from nltk.translate.bleu_score import corpus_bleu
#import os.path as osp
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"
#import os
#import pickle
#from gensim.corpora import Dictionary
#from get_topic_for_caption import get_topics
#import random
#sys.path.append('/home/s1985335/geomloss')
#sys.path.append(osp.join(this_dir,'nlg-eval-master/'))
#
#
#from tqdm import tqdm
#from nlgeval import NLGEval
#from geomloss import SamplesLoss  # See also ImagesLoss, VolumesLoss
#this_dir = osp.dirname(osp.realpath((__file__)))
#nlgeval = NLGEval()
#use_gpu = torch.cuda.is_available()
#torch.manual_seed(1)
#torch.cuda.manual_seed(1)
#np.random.seed(1)
## Data parameters
#data_folder = osp.join(this_dir, 'dataset_2014')  # folder with data files saved by create_input_files.py
#data_name = 'coco_5_cap_per_img_5_min_word_freq'  # base name shared by data files
#
## Model parameters
#emb_dim = 1024  # dimension of word embeddings
## emb_dim = 300  # dimension of glove word embeddings
#num_topics = 512
#attention_dim = 1024  # dimension of attention linear layers
#decoder_dim = 1024  # dimension of decoder RNN
#dropout = 0.5
#lr = 2e-3
## Training parameters
#start_epoch = 0
#epochs = 70  # number of epochs to train for (if early stopping is not triggered)
#epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU
#batch_size = 20
#workers = 24  # for data-loading; right now, only 1 works with h5py
#best_bleu4 = 0.  # BLEU-4 score right now
#print_freq = 100  # print training/validation stats every __ batches
#checkpoint = None  # path to checkpoint, None if none
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(device, list(range(torch.cuda.device_count())))
## torch.cuda.set_device(3)
#cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead
#
#glove_file = osp.join(this_dir, 'dataset_2014','glove_weights_matrix.pkl')
#glove = pickle.load(open(glove_file,'rb'), encoding='latin1')
#
#topic_matrix = np.load(osp.join(this_dir, 'dataset_2014', 'lda_512_matrix_capt_vocab.npy'))
#dictionary_path = osp.join(this_dir, 'dataset_2014', 'captions_dictionary_nobelow2.dict')
#dictionary = Dictionary().load(dictionary_path)
#raw_dict = dictionary
#
#print(os.environ["CUDA_VISIBLE_DEVICES"])
#model_dir = os.path.join(this_dir, 's2stranslation_models')  
#if not os.path.isdir(model_dir):
#    os.mkdir(model_dir)
#
#dictionary.add_documents(['<start>'.split()])
#dictionary.add_documents(['<pad>'.split()])
#word_map = dictionary.token2id
#rev_word_map = {v: k for k, v in word_map.items()}
#vocab_size = len(word_map)
#import time
#import math
#import matplotlib.pyplot as plt
#plt.switch_backend('agg')
#import matplotlib.ticker as ticker
#import numpy as np
#
#def asMinutes(s):
#    m = math.floor(s / 60)
#    s -= m * 60
#    return '%dm %ds' % (m, s)
#
#
#def timeSince(since, percent):
#    now = time.time()
#    s = now - since
#    es = s / (percent)
#    rs = es - s
#    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))
#
#def showPlot(points):
#    plt.figure()
#    fig, ax = plt.subplots()
#    # this locator puts ticks at regular intervals
#    loc = ticker.MultipleLocator(base=0.2)
#    ax.yaxis.set_major_locator(loc)
#    plt.plot(points)
#
#
#def repeat(tensor, dims):
#    if len(dims) != len(tensor.shape):
#        raise ValueError("The length of the second argument must equal the number of dimensions of the first.")
#    for index, dim in enumerate(dims):
#        repetition_vector = [1] * (len(dims) + 1)
#        repetition_vector[index + 1] = dim
#        new_tensor_shape = list(tensor.shape)
#        new_tensor_shape[index] *= dim
#        tensor = tensor.unsqueeze(index + 1).repeat(repetition_vector).reshape(new_tensor_shape)
#    return tensor
#   
#class EncoderRNN(nn.Module):
#    def __init__(self, input_size, hidden_size):
#        super(EncoderRNN, self).__init__()
#        self.hidden_size = hidden_size
#        self.embed_dim = 1024
#        self.embedding = nn.Embedding(input_size, self.embed_dim)
#        self.gru = nn.GRU(self.embed_dim, hidden_size)
#
#    def forward(self, input, hidden):
#        embedded = self.embedding(input).view(1, 1, -1)
#        output = embedded
#        output, hidden = self.gru(output, hidden)
#        return output, hidden
#
#    def initHidden(self):
#        return torch.zeros(1, 1, self.hidden_size, device=device)
#
#class DecoderRNN(nn.Module):
#    def __init__(self, hidden_size, output_size):
#        super(DecoderRNN, self).__init__()
#        self.hidden_size = hidden_size
#        self.embed_dim = 1024
#        self.embedding = nn.Embedding(output_size, self.embed_dim)
#        self.gru = nn.GRU(self.embed_dim, hidden_size)
#        self.out = nn.Linear(hidden_size, output_size)
#        self.softmax = nn.LogSoftmax(dim=1)
#
#    def forward(self, input, hidden):
#        output = self.embedding(input).view(1, 1, -1)
#        output = F.relu(output)
#        output, hidden = self.gru(output, hidden)
#        output = self.softmax(self.out(output[0]))
#        return output, hidden
#
#    def initHidden(self):
#        return torch.zeros(1, 1, self.hidden_size, device=device)
#
#def one_hot_labels(labels,num_classes):
#    y = torch.eye(num_classes)
#    return y[labels]
#
#def clip_gradient(decoder_optimizer_topic, grad_clip):
#    for group in decoder_optimizer_topic.param_groups:
#        #print(group['params'])
#        for param in group['params']:
#            # print(param.grad.data)
#            param.grad.data.clamp_(-grad_clip, grad_clip)
#
#teacher_forcing_ratio = 0.5
#
#
#def train(captions, caption_lengths, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
#    encoder_hidden = encoder.initHidden()
#    batch_size = captions.size(0)
#    max_length = captions.size(1)
#    encoder_optimizer.zero_grad()
#    decoder_optimizer.zero_grad()
#
#    
#    
#    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
#    encoder_inputs = [captions[batch][:length] for batch,length in  enumerate(caption_lengths)]
#    
#    loss = 0.
#    batch_encoder_outputs = []
#    batch_encoder_hiddens = []
#    for input_tensor in encoder_inputs:
#        input_length = input_tensor.size(0)
#    
#        for ei in range(input_length):
#            encoder_output, encoder_hidden = encoder(
#                    input_tensor[ei], encoder_hidden)
#            encoder_outputs[ei] = encoder_output[0, 0]
#            batch_encoder_outputs.append(encoder_outputs)
#            batch_encoder_hiddens.append(encoder_hidden)
#            
#    decoder_input = torch.tensor([word_map['<start>']], device=device)
#
#    batch_decoder_hiddens = batch_encoder_hiddens
#
#    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
#    
#    targets = encoder_inputs
#    if use_teacher_forcing:
#        # Teacher forcing: Feed the target as the next input
#        for b,target_tensor in enumerate(targets):
#            target_length = target_tensor.size(0)
#          
#            decoder_hidden = batch_decoder_hiddens[b]
#            encoder_outputs = batch_encoder_outputs[b]
#            for di in range(target_length):
#                decoder_output, decoder_hidden = decoder(
#                    decoder_input, decoder_hidden)
#              
#                loss += criterion(decoder_output, target_tensor[di].unsqueeze(0))
#                decoder_input = target_tensor[di]  # Teacher forcing
#
#    else:
#        # Without teacher forcing: use its own predictions as the next input
#        for b,target_tensor in enumerate(targets):
#            target_length = target_tensor.size(0)
#           
#            decoder_hidden = batch_decoder_hiddens[b]
#            encoder_outputs = batch_encoder_outputs[b]
#            for di in range(target_length):
#                decoder_output, decoder_hidden = decoder(
#                    decoder_input, decoder_hidden)
#               
#                topv, topi = decoder_output.topk(1)
#                decoder_input = topi.squeeze().detach()  # detach from history as input
#    
#                loss += criterion(decoder_output, target_tensor[di].unsqueeze(0))
#                if decoder_input.item() == word_map['<eoc>']:
#                    break
#                
#
#    loss.backward()
#    
#    encoder_optimizer.step()
#    decoder_optimizer.step()
#
#    return loss.item() / sum(caption_lengths)
#
#
#def trainIters(train_loader,epoch, encoder, decoder, print_every=100, plot_every=100, learning_rate=0.01):
#    start = time.time()
#    plot_losses = []
#    print_loss_total = 0.  # Reset every print_every
#    plot_loss_total = 0.  # Reset every plot_every
#
#    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
#    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)
#
#    criterion =nn.NLLLoss() # nn.CrossEntropyLoss() #nn.NLLLoss()
#    for iter, (imgs, caps, caplens,img_id,topics) in enumerate(train_loader):
#       
#        # Move to GPU, if available
#        imgs = imgs.to(device)
#        caps = caps.to(device)
#        caplens = caplens.to(device)
#        caps = caps.view(-1,52)
#        caplens = caplens.view(-1)
#  
#        loss = train(caps, caplens, encoder,
#                     decoder, encoder_optimizer, decoder_optimizer, criterion)
#        print_loss_total += loss
#        plot_loss_total += loss
##        print(print_loss_total)
#        if iter % print_every == 0:
#            print_loss_avg = print_loss_total / print_every
#            print_loss_total = 0.
#            print('Epoch: [{0}][{1}/{2}]\t'
#                  'CE Loss {loss:.4f}\t' .format(epoch, iter, len(train_loader),loss=print_loss_avg))
#            #print('%s (%d %d%%) %.4f' % (timeSince(start, iter / len(train_loader)),
#             #                            iter, iter / len(train_loader) * 100, print_loss_avg))
#
#        if iter % plot_every == 0:
#            plot_loss_avg = plot_loss_total / plot_every
#            plot_losses.append(plot_loss_avg)
#            plot_loss_total = 0.
#    e_output_file = os.path.join(model_dir, "encoder_model_{}.pt".format(epoch))
#    d_output_file = os.path.join(model_dir, "decoder_model_{}.pt".format(epoch))
#    torch.save(encoder.state_dict(), e_output_file)
#    torch.save(decoder.state_dict(), d_output_file)
#    print("Saving models to {}".format(e_output_file))
##    showPlot(plot_losses)
#
#
#def main():
#    train_loader = torch.utils.data.DataLoader(
#        CaptionNewDictDataset(data_folder, data_name, 'VAL'),
#        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
#    print('train data loaded')
#    val_loader = torch.utils.data.DataLoader(
#        CaptionNewDictDataset(data_folder, data_name, 'VAL'),
#        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
#    print('val data loaded')
#    
#    hidden_size = 256
#    encoder1 = EncoderRNN(vocab_size, hidden_size).to(device)
#    decoder1 = DecoderRNN(hidden_size,vocab_size).to(device)
#    for epoch in range(start_epoch, epochs):
#        trainIters(train_loader, epoch,encoder1, decoder1)
#
#
#
#if __name__ == '__main__':
#    main()
    
    
    
import time
import torch
import os
print(torch.__version__)
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.nn.functional as F
import sys
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
import os.path as osp
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import os
import pickle
from gensim.corpora import Dictionary
from get_topic_for_caption import get_topics
from models_s2stranslation import TranslationModel

sys.path.append('/home/s1985335/geomloss')
sys.path.append(osp.join(this_dir,'nlg-eval-master/'))


from tqdm import tqdm
from nlgeval import NLGEval
from geomloss import SamplesLoss  # See also ImagesLoss, VolumesLoss
this_dir = osp.dirname(osp.realpath((__file__)))
nlgeval = NLGEval()
use_gpu = torch.cuda.is_available()
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
# Data parameters
data_folder = osp.join(this_dir, 'dataset_2014')  # folder with data files saved by create_input_files.py
data_name = 'coco_5_cap_per_img_5_min_word_freq'  # base name shared by data files

# Model parameters
emb_dim = 1024  # dimension of word embeddings
# emb_dim = 300  # dimension of glove word embeddings
num_topics = 512
attention_dim = 1024  # dimension of attention linear layers
decoder_dim = 1024  # dimension of decoder RNN
dropout = 0.5
lr = 2e-3
# Training parameters
start_epoch = 0
epochs = 70  # number of epochs to train for (if early stopping is not triggered)
epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU
batch_size = 20
workers = 24  # for data-loading; right now, only 1 works with h5py
best_bleu4 = 0.  # BLEU-4 score right now
print_freq = 100  # print training/validation stats every __ batches
checkpoint = None  # path to checkpoint, None if none
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device, list(range(torch.cuda.device_count())))
# torch.cuda.set_device(3)
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

glove_file = osp.join(this_dir, 'dataset_2014','glove_weights_matrix.pkl')
glove = pickle.load(open(glove_file,'rb'), encoding='latin1')

topic_matrix = np.load(osp.join(this_dir, 'dataset_2014', 'lda_512_matrix_capt_vocab.npy'))
dictionary_path = osp.join(this_dir, 'dataset_2014', 'captions_dictionary_nobelow2.dict')
dictionary = Dictionary().load(dictionary_path)
raw_dict = dictionary

### [H,T] : GT Topics concatenated with hidden
### [E,T] : LSTM input concatenated with GT Topics
### [H,ET]50% : Hidden concatenated with Topic Embeddings (50% GT, 50% Predicted)
### alignattn[I,H]: Image feATURES CONCATENATED WITH Hidden vector
### alignattn: Image feATURES CONCATENATED WITH Word Embeddings
### caption: only CE lss
### caption[CE+OT]: CE plus OT loss both
print(os.environ["CUDA_VISIBLE_DEVICES"])
model_dir = os.path.join(this_dir, 's2stranslation_attndecoder[1024-withSS]_models')   ## 1024-withSS - Scheduled Sampling with 1024 for both hidden and embedding
if not os.path.isdir(model_dir):
    os.mkdir(model_dir)


def main():
    """
    Training and validation.
    """

    global best_bleu4, epochs_since_improvement, checkpoint, start_epoch, data_name, word_map, rev_word_map

    # Read word map

    dictionary.add_documents(['<start>'.split()])
    dictionary.add_documents(['<pad>'.split()])
    word_map = dictionary.token2id
    rev_word_map = {v: k for k, v in word_map.items()}

    # Initialize / load checkpoint
    if checkpoint is None:
        decoder = TranslationModel(attention_dim=attention_dim,
                                       embed_dim=emb_dim,
                                       hidden_dim=decoder_dim,
                                       vocab_size=len(word_map),
                                       word_map=word_map,
                                       weights_matrix=glove,
                                       dropout=dropout)

        decoder_optimizer_ce = torch.optim.Adamax(params=filter(lambda p: p.requires_grad, decoder.parameters()))
        # decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr, weight_decay=0.0005)
    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_bleu4 = checkpoint['bleu-4']
        decoder = checkpoint['decoder']
        decoder_optimizer_ce = checkpoint['decoder_optimizer_ce']
        #decoder_optimizer_topic = checkpoint['decoder_optimizer_topic']

    # Move to GPU, if available
    # gpus = [1, 2]
    # if len(gpus) > 1:
    #    decoder = torch.nn.DataParallel(decoder)

    decoder = decoder.to(device)
    # decoder = decoder.cuda()
    # Loss functions
    criterion_ce = nn.CrossEntropyLoss()  # .to(device)
    criterion_ot = SamplesLoss(loss="sinkhorn", p=1, blur=.01)
    # Custom dataloaders
    train_loader = torch.utils.data.DataLoader(
        CaptionNewDictDataset(data_folder, data_name, 'TRAIN'),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    print('train data loaded')
    val_loader = torch.utils.data.DataLoader(
        CaptionNewDictDataset(data_folder, data_name, 'VAL'),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    print('val data loaded')
    # Epochs
    for epoch in range(start_epoch, epochs):

        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_improvement == 20:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(decoder_optimizer_ce, 0.8)

        # One epoch's training
        train(train_loader=train_loader,
              decoder=decoder,
              criterion_ce=criterion_ce,
              decoder_optimizer_ce=decoder_optimizer_ce,
              epoch=epoch)

        # One epoch's validation
        recent_bleu4 = validate(val_loader=val_loader,
                                decoder=decoder,
                                criterion_ce=criterion_ce)


        # Check if there was an improvement
        is_best = recent_bleu4 > best_bleu4
        best_bleu4 = max(recent_bleu4, best_bleu4)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(model_dir, data_name, epoch, epochs_since_improvement, decoder, decoder_optimizer_ce, recent_bleu4,
                        is_best)


def repeat(tensor, dims):
    if len(dims) != len(tensor.shape):
        raise ValueError("The length of the second argument must equal the number of dimensions of the first.")
    for index, dim in enumerate(dims):
        repetition_vector = [1] * (len(dims) + 1)
        repetition_vector[index + 1] = dim
        new_tensor_shape = list(tensor.shape)
        new_tensor_shape[index] *= dim
        tensor = tensor.unsqueeze(index + 1).repeat(repetition_vector).reshape(new_tensor_shape)
    return tensor
    
def one_hot_labels(labels,num_classes):
    y = torch.eye(num_classes)
    return y[labels]

def clip_gradient(decoder_optimizer_topic, grad_clip):
    for group in decoder_optimizer_topic.param_groups:
        #print(group['params'])
        for param in group['params']:
            # print(param.grad.data)
            param.grad.data.clamp_(-grad_clip, grad_clip)


def get_topic_distribution(scores,decode_lengths):
    predicted_topics =[]
    _, preds = torch.max(scores, dim=2)
    preds = preds.tolist()
    for j, p in enumerate(preds):
        gen_captions = preds[j][:decode_lengths[j]]
        gen_topic = get_topics(gen_captions, raw_dict, topic_matrix)
        predicted_topics.append(gen_topic)
    predicted_topics = torch.Tensor(np.stack(predicted_topics)).to(device)
    # print(predicted_topics.shape)
    # raw_input('enter')
    return predicted_topics

def train(train_loader, decoder, criterion_ce,decoder_optimizer_ce, epoch):
    """
    Performs one epoch's training.
    :param train_loader: DataLoader for training data
    :param decoder: decoder model
    :param criterion_ce: cross entropy loss layer
    :param criterion_dis : discriminative loss layer
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number
    """

    decoder.train()  # train mode (dropout and batchnorm is used)

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading timex`
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy
    start = time.time()
    split = 'TRAIN'
    # Batches
    for i, (imgs, caps, caplens,img_id,topics) in enumerate(train_loader):
        data_time.update(time.time() - start)
        # Move to GPU, if available
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)
        # top_words = top_words.to(device)
        caps = caps.view(-1,52)
        # top_words = top_words.view(-1,52)
        
        caplens = caplens.view(-1)
        # Forward prop.
        # print(imgs.shape, caps.shape, caplens.shape, img_id.shape)
        scores, caps_sorted, decode_lengths, sort_ind= decoder(caps, caplens,split)

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]
        # sorted_topics = topics[sort_ind]
        # print(sorted_topics.shape)
        # raw_input('enter')
        scores_copy=scores
        # predicted_topics = get_topic_distribution(scores_copy, decode_lengths)   ##### batch_size, topic dim



        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores= pack_padded_sequence(scores, decode_lengths, batch_first=True)
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)
        # top_words = pack_padded_sequence(top_words, decode_lengths, batch_first=True)
        # Calculate loss
        scores = scores.data
        targets = targets.data
        # top_words = top_words.data
        decoder_optimizer_ce.zero_grad()
        loss_g = criterion_ce(scores, targets)

        loss = loss_g
        # loss = loss_g + loss_t #+ lambda1*topic_loss

        ### Topic loss
        # Back prop.
        

        loss.backward()
        # loss_t.backward()
        # Clip gradients when they are getting too large
        torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, decoder.parameters()), 0.25)
        # torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, decoder.parameters()), 0.1)
        # clip_gradient(decoder_optimizer_topic, 0.1)
        # Update weights
        decoder_optimizer_ce.step()

        # Update weights
        # decoder_optimizer_topic.step()
        # Keep track of metrics
        top5 = accuracy(scores, targets, 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss_CE+OT {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses,
                                                                          top5=top5accs))





def validate(val_loader, decoder, criterion_ce):
    """
    Performs one epoch's validation.
    :param val_loader: DataLoader for validation data.
    :param decoder: decoder model
    :param criterion_ce: cross entropy loss layer
    :param criterion_dis : discriminative loss layer
    :return: BLEU-4 score
    """
    decoder.eval()  # eval mode (no dropout or batchnorm)

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)
    split = 'VAL'
    # Batches
    with torch.no_grad():
        for i, (imgs, allcaps, allcaplens, img_id, topics) in enumerate(val_loader):

            # Move to device, if available
            imgs = imgs.to(device)
            allcaps = allcaps.to(device)
            allcaplens = allcaplens.to(device)
            caps = allcaps.view(-1, 52)
            caplens = allcaplens.view(-1)
            scores, caps_sorted, decode_lengths, sort_ind= decoder(caps, caplens, split)

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores_copy = scores
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)
            # Calculate loss
            # loss_d = criterion_dis(scores_d, targets_d.long())
            scores = scores.data
            targets = targets.data
            loss_g = criterion_ce(scores, targets)

           
            # loss = loss_g + lambda1*loss_topics
            loss = loss_g

            # Keep track of metrics
            losses.update(loss.item(), sum(decode_lengths))
            top5 = accuracy(scores, targets, 5)
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            start = time.time()

            if i % print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader),
                                                                                batch_time=batch_time,
                                                                                loss=losses, top5=top5accs))

            # Store references (true captions), and hypothesis (prediction) for each image
            # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
            # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

            # References
            allcaps = repeat(allcaps,[5,1,1])

            allcaps = allcaps[sort_ind]  # because images were sorted in the decoder
            for j in range(allcaps.shape[0]):
                img_caps = allcaps[j].tolist()
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                        img_caps))  # remove <start> and pads
                references.append(img_captions)

            # Hypotheses
            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
            preds = temp_preds
            hypotheses.extend(preds)

            assert len(references) == len(hypotheses)

    # Calculate BLEU-4 scores
    bleu4 = corpus_bleu(references, hypotheses)
    bleu4 = round(bleu4, 4)
    # metrics = nlgeval.compute_metrics(references, hypotheses)
    # bleu4 = metrics['Bleu_4']
    print(
        '\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}\n'.format(
            loss=losses,
            top5=top5accs,
            bleu=bleu4))
    print(model_dir)
    return bleu4


if __name__ == '__main__':
    main()

