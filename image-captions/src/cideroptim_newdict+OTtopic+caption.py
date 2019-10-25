import time
import torch

print(torch.__version__)
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.nn.functional as F
import sys
from torch import distributions
import numpy as np
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from models_cideroptimnewdictOTtopiccaption import *
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
import os.path as osp
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import pickle
from gensim.corpora.dictionary import Dictionary
from get_topic_for_caption import get_topics
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
import torch.nn.functional as F
import itertools
import sys
import os.path as osp

this_dir = osp.dirname(osp.realpath((__file__)))
print(this_dir)

# sys.path.append('/home/s1985335/geomloss')
sys.path.append(osp.join(this_dir,'..','src','coco-caption-master/pycocoevalcap/'))
sys.path.append(osp.join(this_dir,'nlg-eval-master/'))
# sys.path.append('/data1/s1985335/raid/IC-GAN/img_captions/image-captioning-bottom-up-top-down-master/nlg-eval-master/')
# sys.path.append('/data1/s1985335/raid/IC-GAN/img_captions/src/coco-caption-master/pycocoevalcap/')

from tqdm import tqdm
from nlgeval import NLGEval
from gensim.corpora.dictionary import Dictionary
from tokenizer import ptbtokenizer
from cider import cider as cider_train
import os.path as osp
from matplotlib import pyplot as plt
this_dir = osp.dirname(osp.realpath((__file__)))
emb_dim = 1024  # dimension of word embeddings
# emb_dim = 300  # dimension of glove word embeddings
num_topics = 512
attention_dim = 1024  # dimension of attention linear layers
decoder_dim = 1024  # dimension of decoder RNN
dropout = 0.5
use_gpu = torch.cuda.is_available()
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
# Data parameters
data_folder = osp.join(this_dir, 'dataset_2014')  # folder with data files saved by create_input_files.py
data_name = 'coco_5_cap_per_img_5_min_word_freq'  # base name shared by data files
print_freq = 50
total_epochs = 60
### checkpoint from CE training 
checkpoint_file = os.path.join(this_dir, 'e2e_newdict+Notopic+caption+alignattn[I,H]_2014_models', 'BEST_26checkpoint_coco_5_cap_per_img_5_min_word_freq.pth')#'BEST_1checkpoint_coco_5_cap_per_img_5_min_word_freq.pth')  # model checkpoint
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead
glove_file = osp.join(this_dir, 'dataset_2014','glove_weights_matrix.pkl')
glove = pickle.load(open(glove_file,'rb'),encoding='latin1')
nlgeval = NLGEval()  # loads the evaluator
#beam_size = 5
dictionary_path = osp.join(this_dir, 'dataset_2014', 'captions_dictionary_nobelow2.dict')
dictionary = Dictionary().load(dictionary_path)
dictionary.add_documents(['<start>'.split()])
dictionary.add_documents(['<pad>'.split()])
word_map = dictionary.token2id
print(word_map['<eoc>'],word_map['<pad>'], word_map['<start>'])
rev_word_map = {v: k for k, v in word_map.items()}
vocab_size = len(word_map)



# Load model
# torch.nn.Module.dump_patches = True
checkpoint = torch.load(checkpoint_file,map_location = device)

ce_decoder = checkpoint['decoder']
start_epoch = checkpoint['epoch']+1
epochs_since_improvement = checkpoint['epochs_since_improvement']
optimizer = checkpoint['decoder_optimizer']
# optimizer = torch.optim.Adam(decoder.parameters(), lr=0.1)





pred_topic_file = json.load(open(os.path.join(this_dir,'..','MIML_Topic_Models','models_OT+KLloss_with5topics','10ot_train_topics_2014.json'),'rb'))
best_cider = 0.
batch_size =20
model_dir = os.path.join(this_dir, 'cider_optim_e2e_newdict+Notopic+caption+alignattn[I,H]_2014_models')  
if not os.path.isdir(model_dir):
    os.mkdir(model_dir)
def beam_search(decoder, image_features, caps,gt_topics,img_id,beam_size, type):
   
    # Lists to store references (true captions), and hypothesis (prediction) for each image
    # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
    # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]
    k = beam_size
    
    topics =  gt_topics.squeeze().to(device)
    # pred_topics = next(item for item in pred_topic_file if item['image_id']== int(img_id))
    # pred_topics = pred_topics['topic']
    # pred_topics = torch.Tensor(pred_topics).to(device)
    data = {'image_id': int(0), 'caption': str(0)}
    # Move to GPU device, if available
    image_features = image_features.to(device)  # (1, 3, 256, 256)
    image_features = image_features.expand(k, 36, 2048)

    # pred_topics = pred_topics.expand(k, 512)
    # Tensor to store top k previous words at each step; now they're just <start>
    k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)

    # Tensor to store top k sequences; now they're just <start>
    seqs = k_prev_words  # (k, 1)

    # Tensor to store top k sequences' scores; now they're just 0
    top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

    # Lists to store completed sequences and scores
    complete_seqs = list()
    complete_seqs_scores = list()

    # Start decoding
    step = 1
    h1, c1 = decoder.init_hidden_state(k)  # (batch_size, decoder_dim)
    concat = 'early'
    matchattn = 'True'
    attn = 'hidden' ## 'word' '3words'   # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
    while True:

        embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

        # embeddings = decoder.load_embeddings(glove, k_prev_words.cpu()).squeeze(1)
        # embeddings= embeddings.float()
        # predicted_topic_output = decoder.topic_attention(image_features)
        # predicted_topic_output = F.relu(predicted_topic_output)
        # predicted_topic_output  =F.softmax(predicted_topic_output, dim=1)
        if attn == 'hidden':
            matchattn_inputs=decoder.match_attention(image_features,h1) 
        elif attn == '3words':
            if (embeddings.shape[0]!=5):
                # embeddings = torch.cat((embeddings, torch.zeros(embeddings.shape[1]).to(device).unsqueeze(0)),dim=0)
                concat_embeddings = embeddings
            else:
                stacked_embeddings.append(embeddings)
                if (step ==0 or step ==1 or step ==2):
                    concat_embeddings = torch.mean(torch.stack(stacked_embeddings), dim=0)
                else:
                    # print(step)
                    # print(stacked_embeddings)
                    # input('enter')
                    prev_embeddings = stacked_embeddings[(step-3+1) : step]
                    concat_embeddings = torch.mean(torch.stack(prev_embeddings), dim=0)
            matchattn_inputs=decoder.match_attention(image_features,concat_embeddings)
        else: ##word
            matchattn_inputs=decoder.match_attention(image_features,embeddings)#
        # topic_output = F.softmax(topic_output, dim=1)
        if (concat=='early'):
            if (matchattn=='False'):
                h1,c1 = decoder.early_language_model(
                    torch.cat([attention_weighted_encoding, embeddings], dim=1),(h1,c1))
                scores = decoder.fc(h1)
            else:
                h1, c1 = decoder.early_language_model(
                    torch.cat([matchattn_inputs,embeddings], dim=1), (h1, c1))
                # scores = decoder.fc(torch.cat([h1,topics],dim=1))
                scores = decoder.fc(h1)
        else:
            h1, c1 = decoder.late_language_model(embeddings, (h1, c1))
            h2 = torch.cat([h1, attention_weighted_encoding, topic_output], dim=1)
            scores = decoder.fc2(h2)  # (s, vocab_size)

        scores = F.log_softmax(scores, dim=1)
        # Add
        scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)
       
        # For the first step, all k points will have the same scores (since same k previous words, h, c)
        if (type == 'test'):
    
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
            else:
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)
                
        else:
            if step==1:
   
                distr = distributions.Categorical(logits=scores)
                top_k_words = distr.sample()
                top_k_scores = distr.log_prob(top_k_words)
            else:
                distr = distributions.Categorical(logits=scores)
                top_k_words = distr.sample()
                top_k_scores = distr.log_prob(top_k_words)
			

        # Convert unrolled indices to actual indices of scores
        prev_word_inds = top_k_words / vocab_size  # (s)
        next_word_inds = top_k_words % vocab_size  # (s)
        #print(prev_word_inds, next_word_inds)
        # Add new words to sequences
        seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

        # Which sequences are incomplete (didn't reach <end>)?
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                            next_word != word_map['<eoc>']]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        # Set aside complete sequences
        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
        k -= len(complete_inds)  # reduce beam length accordingly

        # Proceed with incomplete sequences
        if k == 0:
            break
        seqs = seqs[incomplete_inds]
        h1 = h1[prev_word_inds[incomplete_inds]]
        c1 = c1[prev_word_inds[incomplete_inds]]

        image_features = image_features[prev_word_inds[incomplete_inds]]
        # pred_topics = pred_topics[prev_word_inds[incomplete_inds]]
        topics = topics[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)
        flag = False
        # Break if things have been going on too long
        if step > 50:
            flag = True
            break
        step += 1
        # print(complete_seqs_scores)
    if flag is not True:
        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]
        seq_scores = complete_seqs_scores[i]
        hypothesis = ([rev_word_map[w] for w in seq if w not in {word_map['<start>'], word_map['<eoc>'], word_map['<pad>']}])
        hypothesis = ' '.join(hypothesis)
        img_caps = caps[0].tolist()
    
        img_captions = list(
            map(lambda c: [rev_word_map[w] for w in c if w not in {word_map['<start>'], word_map['<eoc>'], word_map['<pad>']}],
                img_caps))  # remove <startimg_caps
        img_caps = [' '.join(c) for c in img_captions]
        # print(hypothesis)
        # References
    else:
        seq = seqs[0][:15].tolist()
        seq.append(word_map['<eoc>'])
        # print(seq)
        # print(top_k_scores)
        seq_scores = top_k_scores[0][:15]
        hypothesis = ([rev_word_map[w] for w in seq if w not in {word_map['<start>'], word_map['<pad>']}])
        # print(hypothesis)
        hypothesis = ' '.join(hypothesis)
        img_caps = caps[0].tolist()
    
        img_captions = list(
            map(lambda c: [rev_word_map[w] for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                img_caps))  # remove <startimg_caps
        img_caps = [' '.join(c) for c in img_captions]
            # print(hypothesis)

    
    # print(img_caps)
    # print(seq)
    # Hypotheses
    
    return img_caps, hypothesis, seq_scores


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



def cider_optim_train(trainloader,val_loader,optimizer, epoch,criterion_ce,cider_decoder):
    #### for baseline results
    #decoder.eval()
    #baselines = []
    #beam_size =5
    #with torch.no_grad():
    #    for j, (image_features, caps, caplens,img_id, topics) in enumerate(
    #            tqdm(trainloader, desc="EVALUATING AT BEAM SIZE " + str(beam_size))):
    #        type = 'test'
    #        pred_topics = next(item for item in pred_topic_file if item['image_id']== int(img_id))
    #        pred_topics = pred_topics['topic']
    #        pred_topics = torch.Tensor(pred_topics).to(device).expand(beam_size,512)
    #        gt_caps, baseline_caps, _ = beam_search(decoder, image_features, caps,pred_topics, img_id,beam_size,type)
    #        baselines.append(baseline_caps)
    #    np.save(os.path.join(model_dir,'baselines_26.npy'),baselines)
    baselines = np.load(os.path.join(model_dir,'baselines_26.npy'))
    cider_decoder.train()
    losses = AverageMeter() 
    rewards = AverageMeter()
    ciders = AverageMeter()
    running_loss = .0
    running_reward = .0
    running_cider =.0
    beam_size=5
    split = 'TRAIN'
    type = 'sample'
    iters = 0
    for it, (imgs, allcaps, allcaplens,img_id, topics) in enumerate(trainloader):
            #tqdm(trainloader, desc="EVALUATING AT BEAM SIZE " + str(beam_size))):
        
#        gt_caps, gen_caps, log_scores = beam_search(decoder, image_features, caps,topics, img_id, beam_size,type)

        references = list()
        baselines_final = []
        hypotheses = list()
        # optimizer.zero_grad()

#        references.append(gt_caps)
#        hypotheses.append(gen_caps)
#        baselines_final.append(baselines[it])
#        
        baselines_final.append(baselines[(it*(batch_size)):(it*(batch_size)+(batch_size))])
        repeat_baselines = [y for x in baselines_final[0] for y in (x,)*5]
        #ref_baselines_final.append(ref_baselines[(it*(batch_size-1)):(it*(batch_size-1)+(batch_size-1))])
        # # Move to device, if available
        imgs = imgs.to(device)
        allcaps = allcaps.to(device)
        allcaplens = allcaplens.to(device)
        caps = allcaps.view(-1, 52)
        topics = topics.to(device)
        topics = topics.view(-1,512)
        caplens = allcaplens.view(-1)


        scores, sample_words, sample_probs, caps_sorted, decode_lengths, sort_ind,sorted_topics= cider_decoder(imgs, caps, caplens,vocab_size,topics, split)
        optimizer.zero_grad()
        
        # # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]
        
        # # Remove timesteps that we didn't decode at, or are pads
        # # pack_padded_sequence is an easy trick to do this
        scores_copy = scores
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)
 
        scores = scores.data
        targets = targets.data
        scores_copy = F.log_softmax(scores_copy, dim=2)
#        loss_ce  = criterion_ce(scores,targets)
        # # Store references (true captions), and hypothesis (prediction) for each image
        # # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
        # # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

        # # References
        allcaps = repeat(allcaps,[5,1,1])

        allcaps = allcaps[sort_ind]  # because images were sorted in the decoder
        for j in range(allcaps.shape[0]):
            img_caps = allcaps[j].tolist()
            img_captions = list(
                map(lambda c: [rev_word_map[w] for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                    img_caps))  # remove <start> and pads
            img_caps = [' '.join(c) for c in img_captions]
            #print(img_caps)
            references.append(img_caps)
#        print(references)
        # # Hypotheses
        # print(scores_copy)
        # print(scores_copy.shape)
        # input('enter')
        # top_k_words = []
        # top_k_scores = []
        # sampleprobs = []
        decoded_sample_scores  =[torch.mean(sample_probs[batch][:length]).unsqueeze(0) for batch, length in enumerate(decode_lengths)]
        decoded_sample_words  =[sample_words[batch][:length].squeeze(1) for batch, length in enumerate(decode_lengths)]
        # for b_scores in decoded_scores:
        #     # b_scores = b_scores.detach().cpu()
        #     distr = distributions.Categorical(logits=b_scores)
        #     b_words = distr.sample()
        #     b_log_scores = distr.log_prob(b_words)
            
        #     # sampleLogprobs = b_scores.gather(1,b_words.view(-1,1)).squeeze(1)
        #     b_log_scores = distr.log_prob(b_words)
        #     top_k_words.append(b_words)
        #     top_k_scores.append(b_log_scores)
        # top_k_scores = torch.cat(top_k_scores)
      
        # distr = distributions.Categorical(logits=scores_copy)
        # top_k_words = distr.sample()
        # top_k_scores = distr.log_prob(top_k_words)
#        _, preds = torch.max(scores_copy, dim=2)
        # preds = top_k_words.tolist()
        # print(decoded_sample_scores)

        temp_preds = []
        for k, p in enumerate(decoded_sample_words):
            preds = decoded_sample_words[k].tolist()

            # seq = preds[:decode_lengths[j]]
            seq = preds
            seq.append(13532.0)
  
            temp_preds.append(preds[:decode_lengths[k]])  # remove pads
            # preds = temp_preds
            hypothesis = ([rev_word_map[w] for w in seq if w not in {word_map['<start>'], word_map['<pad>']}])
            hypothesis = ' '.join(hypothesis)
            hypotheses.append(hypothesis)
        # print(hypotheses)
        # print(repeat_baselines)
        repeat_baselines = np.array(repeat_baselines)
        # print(sort_ind)
        repeat_baselines = repeat_baselines[sort_ind.cpu().numpy()].tolist()
        # rint(baselines_final)
        assert len(references) == len(hypotheses) == len(repeat_baselines)
#        assert len(ref_baselines_final) == len(baselines_final)
        # cider_reward= nlgeval.compute_metrics(references, hypotheses)
        # # print(cider_reward)
        # # print(hypotheses)
        # input('enter')
        # reward = cider_reward['CIDEr']

        # cider_reward_baseline  = nlgeval.compute_metrics(references,repeat_baselines)
        # # print(cider_reward_baseline)
        # # print(baselines_final)
        # reward_baseline = cider_reward_baseline['CIDEr']
        # gts = []
        # gen=[]
        # bas =[]
        # for i, (gts_i, gen_i, bas_i) in enumerate(zip(references, hypotheses, repeat_baselines)):
        #     gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
        #     gts.append([gts_i, ])
        #     gen.append(gen_i)
        #     bas.append(bas_i)
        # print(it)
        # input('enter')
        # print(repeat_baselines)
        # input('enter')
        # print(references)
        # input('enter')
        # print(hypotheses)
        # input('enter')
        caps_gts = ptbtokenizer.PTBTokenizer().tokenize(references)
        caps_gen = ptbtokenizer.PTBTokenizer().tokenize(hypotheses)
        caps_baseline = ptbtokenizer.PTBTokenizer().tokenize(repeat_baselines)

        reward = cider_train.Cider().compute_score(caps_gts, caps_gen)[1].astype(np.float32)
        reward_baseline = cider_train.Cider().compute_score(caps_gts, caps_baseline)[1].astype(np.float32)
      
        reward = torch.from_numpy(np.array(reward)).to(device).float()
        reward_baseline = torch.from_numpy(np.array(reward_baseline)).to(device).float()
        # print(decoded_sample_scores)
        # print(reward,reward_baseline)
        unrolled_seq = torch.cat(decoded_sample_words).unsqueeze(1)
        mask = (unrolled_seq>0).float()
        mask = (torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)).view(-1)
        mask = mask.contiguous()
        loss_cap = -torch.cat(decoded_sample_scores) * (reward-reward_baseline)
        loss_cap =  torch.sum(loss_cap) / torch.sum(mask) 
#        loss_cap = -(torch.mean(torch.cat(decoded_sample_scores), -1)) * (reward - reward_baseline)
#        loss_cap = loss_cap.mean()
        # loss_cap = (loss_ce) * torch.mean((reward_baseline - reward))
        # loss_cap = loss_cap.mean()
#        loss_cap = loss_ce
        loss = loss_cap
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, cider_decoder.parameters()), 0.25)
        optimizer.step()
        iters +=1
        running_loss += loss_cap.item()
        running_reward += torch.mean(reward - reward_baseline).item()
        running_cider+=torch.mean(reward).item()
        if (it % print_freq == 0):
            print('Epoch: [{0}][{1}/{2}]\t' 
                        'Loss {loss:.4f}\t'
                        'Reward {reward:.4f}\t'
                        'Cider {cider:.4f}\t'.format(epoch, it, len(trainloader),
                                                    loss=running_loss/ (iters), reward=running_reward / (iters), cider=running_cider/(iters)))
        # if (it % 2000 == 0):
        #     save_checkpoint(model_dir, data_name, it, epochs_since_improvement, decoder, optimizer ,running_cider/(iters),
        #                 running_cider/(iters))


def cider_optim_val(valloader, epoch,cider_decoder):
    cider_decoder.eval()
    gen = list()
    gts = list()
    beam_size=5
    type = 'test'
    with torch.no_grad():
        for it, (image_features, caps, caplens,img_id, topics) in enumerate(
                tqdm(valloader, desc="EVALUATING AT BEAM SIZE " + str(beam_size))):
            gt_caps, gen_caps, _ = beam_search(cider_decoder, image_features, caps,topics, img_id, beam_size,type)

            gen.append(gen_caps)
            gts.append(gt_caps)

    cider_val_reward  = nlgeval.compute_metrics(gts, gen)
    val_cider = cider_val_reward['CIDEr']
    # gts = ptbtokenizer.PTBTokenizer().tokenize(gts)
    # gen = ptbtokenizer.PTBTokenizer().tokenize(gen)

    # val_cider, _ = cider_train.Cider().compute_score(gts, gen)
    print('CIDEr', val_cider)

    return val_cider



def main():
    ### Cider Optimization based on SCST
    global best_cider, epochs_since_improvement, checkpoint, start_epoch, data_name, word_map
    batch_size  =20
    workers = 16
    # Custom dataloaders
    train_loader = torch.utils.data.DataLoader(
        CaptionNewDictDataset(data_folder, data_name, 'TRAIN'),
        batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    print('train data loaded')
    val_loader = torch.utils.data.DataLoader(
        CaptionNewDictDataset(data_folder, data_name, 'VAL'),
        batch_size=1, shuffle=False, num_workers=workers, pin_memory=True)
    print('val data loaded')
    # Epochs
    criterion_ce = nn.CrossEntropyLoss() 
    cider_decoder = DecoderWithTopicAttention(attention_dim=attention_dim,
                                       embed_dim=emb_dim,
                                       decoder_dim=decoder_dim,
                                       vocab_size=len(word_map),
                                       weights_matrix=glove,
                                       dropout=dropout)
    cider_decoder = cider_decoder.to(device)

    cider_decoder.load_state_dict(ce_decoder.state_dict(), strict=False)
    for epoch in range(start_epoch, total_epochs):

        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_improvement == 20:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(optimizer, 0.8)

        # One epoch's training
        cider_optim_train(train_loader,val_loader,optimizer, epoch,criterion_ce, cider_decoder)
        

        # One epoch's validation
        recent_cider = cider_optim_val(val_loader, epoch, cider_decoder)

        # Check if there was an improvement
        is_best = recent_cider > best_cider
        best_cider = max(recent_cider, best_cider)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(model_dir, data_name, epoch, epochs_since_improvement, cider_decoder, optimizer ,recent_cider,
                        is_best)



if __name__ == '__main__':
    main()
