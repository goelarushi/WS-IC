import time
import torch

print(torch.__version__)
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.nn.functional as F

import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from models_captattn import CaptionLSTM
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
import os.path as osp
import os
import pickle
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
# torch.cuda.max_memory_allocated(device=0)
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
feature_dim=2048
lr = 2e-3 #2e-3
# Training parameters
start_epoch = 0
epochs = 50  # number of epochs to train for (if early stopping is not triggered)
epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU
batch_size = 20 #100
workers = 18  # for data-loading; right now, only 1 works with h5py
best_bleu4 = 0.  # BLEU-4 score right now
print_freq = 100  # print training/validation stats every __ batches
checkpoint = None  # path to checkpoint, None if none

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cuda'
print(device, list(range(torch.cuda.device_count())))
# torch.cuda.set_device(3)
#cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead
glove_file = osp.join(this_dir, 'dataset_2014','glove_weights_matrix.pkl')
glove = pickle.load(open(glove_file,'rb'))

print(os.environ["CUDA_VISIBLE_DEVICES"])
model_dir = os.path.join(this_dir, 'e2e_ncaptattn_2014_models')
if not os.path.isdir(model_dir):
    os.mkdir(model_dir)
# checkpoint_file = os.path.join(this_dir, 'e2e_captattn_2014_models', 'BEST_48checkpoint_coco_5_cap_per_img_5_min_word_freq.pth')#'BEST_1checkpoint_coco_5_cap_per_img_5_min_word_freq.pth')  # model checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# Load model
# torch.nn.Module.dump_patches = True
# checkpoint = torch.load(checkpoint_file,map_location = device)
# decoder = checkpoint['decoder']
# decoder = decoder.to(device)

def main():
    """
    Training and validation.
    """

    global best_bleu4, epochs_since_improvement, checkpoint, start_epoch, data_name, word_map

    # Read word map
    word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

    # Initialize / load checkpoint
    if checkpoint is None:
        decoder = CaptionLSTM(attention_dim=attention_dim,
                                       embed_dim=emb_dim,
                                       decoder_dim=decoder_dim,
                                       vocab_size=len(word_map),
                                       weights_matrix=glove,
                                       dropout=dropout)

        decoder_optimizer = torch.optim.Adamax(params=filter(lambda p: p.requires_grad, decoder.parameters()))
    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_bleu4 = checkpoint['bleu-4']
        decoder = checkpoint['decoder']
        decoder_optimizer = checkpoint['decoder_optimizer']

    # Move to GPU, if available
    # gpus = [1, 2]
    # if len(gpus) > 1:
    #    decoder = torch.nn.DataParallel(decoder)

    decoder = decoder.to(device)
    #decoder = decoder.cuda()
    # Loss functions
    criterion_ce = nn.CrossEntropyLoss()  # .to(device)

    # Custom dataloaders
    train_loader = torch.utils.data.DataLoader(
        CaptionAttentionDataset(data_folder, data_name, 'TRAIN'),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    print('train data loaded')
    val_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'VAL'),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    print('val data loaded')
    # Epochs
    for epoch in range(start_epoch, epochs):

        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        #if epochs_since_improvement == 20:
        #    break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(decoder_optimizer, 0.8)

        # One epoch's training
        caption_weights = train(train_loader=train_loader,
              decoder=decoder,
              criterion_ce=criterion_ce,
              decoder_optimizer=decoder_optimizer,
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
            cap_json_file = osp.join(model_dir, 'caption_weights' + str(epoch) + '.pkl')
            with open(cap_json_file, 'wb') as outfile:
                pickle.dump(caption_weights, outfile)
        # # Save checkpoint
        save_checkpoint(model_dir, data_name, epoch, epochs_since_improvement, decoder, decoder_optimizer, recent_bleu4,
                        is_best)


def train(train_loader, decoder, criterion_ce, decoder_optimizer, epoch):
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
    print(len(train_loader))
    start = time.time()
    split = 'TRAIN'
    # Batches
    cpi = 5
    all_caption_weights = []
    for i, (imgs, allcaps, allcaplens,wght_caps, wght_caplens, img_id, topics) in enumerate(train_loader):
    # for i, (imgs, allcaps, allcaplens, fixed_capt_weights, img_id, topics) in enumerate(train_loader):
        data_time.update(time.time() - start)
        cur_batch_size = imgs.shape[0]
        # Move to GPU, if available
        caption_weights=[]
        # fixed_capt_weights =fixed_capt_weights.to(device)
        imgs = imgs.to(device)
        allcaplens = allcaplens.to(device)
        # wght_caplens = wght_caplens.to(device)
        topics = topics.to(device)
        allcaps = allcaps.to(device)
        # wght_caps = wght_caps.to(device)


        # Forward prop.
        #### This one for all 5 captions
        scores, caps_sorted,capt_weights,decode_lengths, sort_ind, org_idx,topic_output = decoder(imgs, allcaps, allcaplens, topics,split,cpi)
        ##### This one for 4 captions
        # scores, caps_sorted, capt_weights, decode_lengths, sort_ind, org_idx, topic_output = decoder(imgs, wght_caps,
        #                                                                                              wght_caplens,
        #                                                                                              topics, split,cpi)
        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>

        targets = caps_sorted
        targets = targets[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        # print(scores.shape)
        unpacked_scores = [score[0:sid] for score, sid in zip(scores, decode_lengths)]
        unpacked_targets = [target[0:tid] for target, tid in zip(targets, decode_lengths)]
        unpacked_scores = torch.cat(unpacked_scores,dim=0)
        unpacked_targets = torch.cat(unpacked_targets, dim=0)
        # normal_loss = criterion_ce(unpacked_scores, unpacked_targets)

        batch_loss = []
        scores = scores.view(cur_batch_size,cpi,-1,9490)
        decode_lengths = np.array(decode_lengths).reshape(cur_batch_size,cpi)
        targets = targets.view(cur_batch_size,cpi,-1)


        for bs in range(cur_batch_size):
            bscores = scores[bs]
            btargets = targets[bs]
            bdecode_lengths = decode_lengths[bs]

            # scores = unpacked_scores[bs*5: (bs*5)+5]
            # targets = unpacked_targets[bs*5: (bs*5)+5]
            caption_losses= []
            image_id = int(img_id[bs].cpu().numpy())

            cap_wght = torch.squeeze(capt_weights[bs], 1)
            # cap_wght = (fixed_capt_weights[bs])
            for cap in range(cpi):
                cscores = bscores[cap][0:bdecode_lengths[cap]]
                ctargets = btargets[cap][0:bdecode_lengths[cap]]
                loss = cap_wght[cap] * criterion_ce(cscores, ctargets)
                # loss = 0.20 * criterion_ce(cscores, ctargets)
                # loss = criterion_ce(scores[cap], targets[cap])
                caption_losses.append(loss)
            ### Weighted caption loss

            data= {'image_id': image_id, 'caption_weights': cap_wght.detach().cpu().numpy()}
            # mul_caption_losses = [cap_wght[k]*caption_losses[k] for k in range(len(caption_losses))]
            # mul_caption_losses = [1 * caption_losses[k] for k in range(len(caption_losses))]
            weighted_caption_loss = torch.stack(caption_losses, dim=0).sum(dim=0)    #### use mean when weight is 1 and sum when learning weights
            # max_loss = torch.argmax(torch.Tensor(caption_losses))
            # batch_loss.append(caption_losses[max_loss.item()])
            # final_loss+=caption_losses[min_loss.item()]
            batch_loss.append(weighted_caption_loss)
            caption_weights.append(data)

        final_loss = torch.stack(batch_loss, dim=0).mean(dim=0)
        all_caption_weights.append(caption_weights)


        # Back prop.
        decoder_optimizer.zero_grad()
        final_loss.backward()
        #
        # # Clip gradients when they are getting too large
        torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, decoder.parameters()), 0.25)
        #
        # Update weights
        decoder_optimizer.step()
        # Keep track of metrics
        #top5 = accuracy(scores, targets, 5)
        losses.update(final_loss.item(), unpacked_scores.shape[0])
        #top5accs.update(top5, sum(decode_lengths_mean))
        batch_time.update(time.time() - start)
        start = time.time()
        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses))
    final_caption_weights = np.asarray(all_caption_weights)
    # cap_json_file = osp.join(model_dir, 'caption_weights_val' + str(48) + '.pkl')
    # with open(cap_json_file, 'wb') as outfile:
    #     pickle.dump(final_caption_weights, outfile)
    return final_caption_weights


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
    cpi=1
    # Batches
    caption_weights = []
    with torch.no_grad():
        for i, (imgs, caps, caplens, allcaps,  img_id, topics) in enumerate(val_loader):

            # Move to device, if available
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)
            topics = topics.to(device)
            # Forward prop.
            scores, caps,capt_weights, decode_lengths, sort_ind,topic_output= decoder(imgs, caps, caplens,topics, split,cpi)
            scores_copy=scores
            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>

            targets = caps
            targets = targets[:, 1:]

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            # print(scores.shape)
            scores= pack_padded_sequence(scores, decode_lengths, batch_first=True)
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)

            scores = scores.data
            targets = targets.data
            loss_g = criterion_ce(scores, targets)
            # loss_topics = -1 * torch.sum(topics * topic_output)
            # # loss = loss_g + (10 * loss_d)
            # lambda1 = 0.01
            # loss = loss_g + lambda1 * loss_topics
            loss = loss_g

            # Keep track of metrics
            losses.update(loss.item(), sum(decode_lengths))
            # Keep track of metrics
            top5 = accuracy(scores, targets, 5)
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            start = time.time()

            if i % print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(i, len(val_loader),
                                                                                batch_time=batch_time,
                                                                                loss=losses))

            # Store references (true captions), and hypothesis (prediction) for each image
            # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
            # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

            # References
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
        '\n * LOSS - {loss.avg:.3f}, BLEU-4 - {bleu}\n'.format(
            loss=losses,
            bleu=bleu4))

    return bleu4


if __name__ == '__main__':
    main()

