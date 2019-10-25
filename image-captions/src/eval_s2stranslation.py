import torch
import time
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch.nn.utils.rnn import pack_padded_sequence

from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
import torch.nn.functional as F
import sys
print_freq = 50
import os.path as osp
this_dir = osp.dirname(osp.realpath((__file__)))
print(this_dir)
sys.path.append(osp.join(this_dir,'..','src','coco-caption-master/pycocoevalcap/'))
sys.path.append(osp.join(this_dir,'nlg-eval-master/'))
# sys.path.append('/data1/s1985335/raid/IC-GAN/img_captions/image-captioning-bottom-up-top-down-master/nlg-eval-master')

from tqdm import tqdm
from nlgeval import NLGEval
from gensim.corpora.dictionary import Dictionary

import os.path as osp
from matplotlib import pyplot as plt
this_dir = osp.dirname(osp.realpath((__file__)))
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

use_gpu = torch.cuda.is_available()
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
# Data parameters
data_folder = osp.join(this_dir, 'dataset_2014')  # folder with data files saved by create_input_files.py
data_name = 'coco_5_cap_per_img_5_min_word_freq'  # base name shared by data files

# checkpoint_file = os.path.join(this_dir, 's2stranslation_attndecoder[1024-withSS]_models', 'BEST_1checkpoint_coco_5_cap_per_img_5_min_word_freq.pth') # model checkpoint
#checkpoint_file = os.path.join(this_dir, 'cider_optim_e2e_newdict+[H+ET]topic+caption+alignattn_2014_models', 'BEST_33checkpoint_coco_5_cap_per_img_5_min_word_freq.pth')#'BEST_1checkpoint_coco_5_cap_per_img_5_min_word_freq.pth')  # model checkpoint
checkpoint_folder = os.path.join(this_dir, 's2stranslation_attndecoder[1024-withSS]_models') # model checkpoint
chkpt_files = [os.path.join(checkpoint_folder, fn) for fn in sorted(os.listdir(checkpoint_folder))]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# Load model
torch.nn.Module.dump_patches = True
# checkpoint = torch.load(checkpoint_file,map_location = device)
# decoder = checkpoint['decoder']
# decoder = decoder.to(device)
# decoder.eval()
glove_file = osp.join(this_dir, 'dataset_2014','glove_weights_matrix.pkl')
glove = pickle.load(open(glove_file,'rb'),encoding='latin1')
nlgeval = NLGEval()  # loads the evaluator


dictionary_path = osp.join(this_dir, 'dataset_2014', 'captions_dictionary_nobelow2.dict')
dictionary = Dictionary().load(dictionary_path)
dictionary.add_documents(['<start>'.split()])
dictionary.add_documents(['<pad>'.split()])
word_map = dictionary.token2id
rev_word_map = {v: k for k, v in word_map.items()}
# print(rev_word_map[13321])
# input('enter')
vocab_size = len(word_map)
pred_topic_file = json.load(open(os.path.join(this_dir,'..','MIML_Topic_Models','models_OT+KLloss_with5topics','10ot_test_topics_2014.json')))
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


import matplotlib.pyplot as plt
# plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    ax.xaxis.set_major_locator(loc)
    plt.plot(points)
    plt.savefig(os.path.join(checkpoint_folder, 'plot_loss.jpg'))
    plt.show()



def validate(criterion_ce, decoder, loader):
    """
    Performs one epoch's validation.
    :param val_loader: DataLoader for validation data.
    :param decoder: decoder model
    :param criterion_ce: cross entropy loss layer
    :param criterion_dis : discriminative loss layer
    :return: BLEU-4 score
    """
    # decoder.eval()  # eval mode (no dropout or batchnorm)
    # loader = torch.utils.data.DataLoader(
    #     CaptionNewDictDataset(data_folder, data_name, 'TEST'),
    #     batch_size=20, shuffle=True, num_workers=16, pin_memory=torch.cuda.is_available())
    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()
    plot_losses =[]
    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)
    split = 'VAL'
    # Batches
    with torch.no_grad():
        for i, (imgs, allcaps, allcaplens, img_id, topics) in enumerate(loader):

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
            plot_losses.append(losses.avg)
            if i % print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(loader),
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
                map(lambda c: [rev_word_map[w] for w in c if w not in {word_map['<start>'], word_map['<eoc>'], word_map['<pad>']}],
                    img_caps))  # remove <start> and pads
                img_caps = [' '.join(c) for c in img_captions]
                #print(img_caps)
                references.append(img_caps)
                #references.append(img_captions)

            # Hypotheses
            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
                seq = preds[j]
                seq = [seq[i] for i in range(len(seq)-1) if seq[i]!=seq[i+1]]
                hypothesis = ([rev_word_map[w] for w in seq if w not in {word_map['<start>'],word_map['<eoc>'], word_map['<pad>']}])
                hypothesis = ' '.join(hypothesis)
                hypotheses.append(hypothesis)
           
            assert len(references) == len(hypotheses)

    # Calculate BLEU-4 scores
    # bleu4 = corpus_bleu(references, hypotheses)
    # bleu4 = round(bleu4, 4)
    metrics = nlgeval.compute_metrics(references, hypotheses)
    bleu4 = metrics['Bleu_4']
    print(
        '\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}\n'.format(
            loss=losses,
            top5=top5accs,
            bleu=bleu4))
    print(metrics)
    # showPlot(plot_losses)
    return losses.avg, bleu4


if __name__ == '__main__':
    criterion_ce = nn.CrossEntropyLoss()
    results_dir = os.path.join(checkpoint_folder, 'results_test.npy')
    results = {}
    plot_loss = []
    loader = torch.utils.data.DataLoader(
        CaptionNewDictDataset(data_folder, data_name, 'TEST'),
        batch_size=20, shuffle=True, num_workers=16, pin_memory=torch.cuda.is_available())
    for ch in range(len(chkpt_files)):
        checkpoint_file = chkpt_files[ch]
        checkpoint = torch.load(checkpoint_file,map_location = device)
        decoder = checkpoint['decoder']
        decoder = decoder.to(device)
        decoder.eval()
        epoch = checkpoint['epoch']
        loss,bleu4 = validate(criterion_ce,decoder, loader)
        print(epoch)
        results[epoch] = [loss,bleu4]
        plot_loss.append(loss)
    np.save(results_dir, results)
    showPlot(plot_loss)
    
