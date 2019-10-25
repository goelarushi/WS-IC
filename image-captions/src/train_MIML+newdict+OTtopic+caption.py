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
from models_MIMLnewdictOTtopiccaption import DecoderWithTopicAttention
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
import os.path as osp
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import os
import pickle
from gensim.corpora import Dictionary
from get_topic_for_caption import get_topics

sys.path.append('/home/s1985335/geomloss')

from geomloss import SamplesLoss  # See also ImagesLoss, VolumesLoss
this_dir = osp.dirname(osp.realpath((__file__)))
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
epochs = 45  # number of epochs to train for (if early stopping is not triggered)
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
latent_classes = [64,128, 256, 512]
print(os.environ["CUDA_VISIBLE_DEVICES"])
model_dir = os.path.join(this_dir, 'e2e_newdict+Notopic+caption[CE]+mimlattn[I,H]Softmax[256]_2014_models')  
if not os.path.isdir(model_dir):
    os.mkdir(model_dir)


def main():
    """
    Training and validation.
    """

    global best_bleu4, epochs_since_improvement, checkpoint, start_epoch, data_name, word_map

    # Read word map

    dictionary.add_documents(['<start>'.split()])
    dictionary.add_documents(['<pad>'.split()])
    word_map = dictionary.token2id
    # Initialize / load checkpoint
    if checkpoint is None:
        decoder = DecoderWithTopicAttention(attention_dim=attention_dim,
                                       embed_dim=emb_dim,
                                       decoder_dim=decoder_dim,
                                       vocab_size=len(word_map),
                                       weights_matrix=glove,
                                       dropout=dropout)

        decoder_optimizer_topic = torch.optim.RMSprop(decoder.parameters(), lr=lr)
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
    criterion_dis = nn.MultiLabelMarginLoss()  # .to(device)
    criterion_ot = SamplesLoss(loss="sinkhorn", p=2, blur=.01)
    # Custom dataloaders
    train_loader = torch.utils.data.DataLoader(
        CaptionNewDictDataset(data_folder, data_name, 'TRAIN'),
        batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
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
              criterion_ot=criterion_ot,
              decoder_optimizer_ce=decoder_optimizer_ce,
              decoder_optimizer_topic=decoder_optimizer_topic,
              epoch=epoch)

        # One epoch's validation
        recent_bleu4 = validate(val_loader=val_loader,
                                decoder=decoder,
                                criterion_ce=criterion_ce,
                                criterion_ot=criterion_ot)


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

def train(train_loader, decoder, criterion_ce, criterion_ot,decoder_optimizer_ce, decoder_optimizer_topic, epoch):
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
    losses_t = AverageMeter() 
    top5accs = AverageMeter()  # top5 accuracy
    topic_dim=512
    start = time.time()
    split = 'TRAIN'
    # Batches
    for i, (imgs, caps, caplens,img_id,topics) in enumerate(train_loader):
        data_time.update(time.time() - start)
        # Move to GPU, if available
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)
        topics = topics.to(device)
        # top_words = top_words.to(device)
        caps = caps.view(-1,52)
        # top_words = top_words.view(-1,52)
        caplens = caplens.view(-1)
        topics = topics.view(-1,topic_dim)
        # Forward prop.
        # print(imgs.shape, caps.shape, caplens.shape, img_id.shape)
        scores, caps_sorted, decode_lengths, sort_ind, sorted_topics= decoder(imgs, caps, caplens,topics,split)

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
        # decoder_optimizer_topic.zero_grad()
        loss_g = criterion_ce(scores, targets)

        # one_hot_targets = one_hot_labels(targets, len(word_map))
        # loss_ot = criterion_ot(F.softmax(scores,dim=1), one_hot_targets.to(device))

        # predicted_topics = F.relu(predicted_topics)
        # predicted_topics = F.softmax(predicted_topics,dim=1)
        # topic_loss = -1*torch.sum(sorted_topics*predicted_topics)
        # loss_t = criterion_ce(scores, top_words)
        # loss_t = criterion_ot(predicted_topics, sorted_topics)
        # lam = 0.9 #0.5 #0.1
        # loss  = (1-lam)*loss_ot + lam * loss_g
        #loss = loss_g + loss_t #+ lambda1*topic_loss
        loss = loss_g
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
        # losses_t.update(loss_ot.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss_CE {loss.val:.4f} ({loss.avg:.4f})\t'
                #   'Loss_OT {loss_t.val:.4f} ({loss_t.avg:.4f})\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses,
                                                                        #   loss_t=losses_t,
                                                                          top5=top5accs))





def validate(val_loader, decoder, criterion_ce, criterion_ot):
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
            topics = topics.to(device)
            topics = topics.view(-1,512)
            caplens = allcaplens.view(-1)
            scores, caps_sorted, decode_lengths, sort_ind,sorted_topics= decoder(imgs, caps, caplens,topics, split)



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

            # one_hot_targets = one_hot_labels(targets, len(word_map))
            # loss_ot = criterion_ot((F.softmax(scores,dim=1)), one_hot_targets.to(device))
        
            # lam =0.9 #0.5# 0.1
            # loss  = (1-lam)*loss_ot + lam * loss_g
            # loss = loss_g + (10 * loss_d)
            # lambda1 = 0.01
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

    print(
        '\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}\n'.format(
            loss=losses,
            top5=top5accs,
            bleu=bleu4))
    print(model_dir)
    return bleu4


if __name__ == '__main__':
    main()
