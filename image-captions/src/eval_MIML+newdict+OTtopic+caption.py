import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
import torch.nn.functional as F
import sys
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
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

use_gpu = torch.cuda.is_available()
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
# Data parameters
data_folder = osp.join(this_dir, 'dataset_2014')  # folder with data files saved by create_input_files.py
data_name = 'coco_5_cap_per_img_5_min_word_freq'  # base name shared by data files

# checkpoint_file = os.path.join(this_dir, 'e2e_newdict+Notopic+caption[CE]+mimlattn[I,H]Softmax_2014_models', 'BEST_33checkpoint_coco_5_cap_per_img_5_min_word_freq.pth')#'BEST_1checkpoint_coco_5_cap_per_img_5_min_word_freq.pth')  # model checkpoint
# checkpoint_file = os.path.join(this_dir, 'cider_optim_e2e_newdict+[H+ET]topic+caption+alignattn_2014_models', 'BEST_775checkpoint_coco_5_cap_per_img_5_min_word_freq.pth')#'BEST_1checkpoint_coco_5_cap_per_img_5_min_word_freq.pth')  # model checkpoint
checkpoint_folder = os.path.join(this_dir, 'e2e_newdict+Notopic+caption[CE+OT-0.9]+mimlattn[I,H]Softmax[128]_2014_models')

### When testing on all the checkpoints in a folder
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
def evaluate(beam_size,decoder, loader):
    """
    Evaluation

    :param beam_size: beam size at which to generate captions for evaluation
    :return: Official MSCOCO evaluator scores - bleu4, cider, rouge, meteor
    """
    # DataLoader
    # loader = torch.utils.data.DataLoader(
    #     CaptionNewDictDataset(data_folder, data_name, 'TEST'),
    #     batch_size=1, shuffle=True, num_workers=16, pin_memory=torch.cuda.is_available())

    # Lists to store references (true captions), and hypothesis (prediction) for each image
    # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
    # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]
    references = list()
    hypotheses = list()
    all_captions=[]
    cap_json_file = osp.join(this_dir, 'generated_captions', 'e2e_newdict+Notopic+caption[CE]+alignattn[I,3W]_2014_models')
    # For each image
    for i, (image_features, caps, caplens,img_id, topics) in enumerate(
            tqdm(loader, desc="EVALUATING AT BEAM SIZE " + str(beam_size))):
        k = beam_size

        topics =  topics.squeeze().to(device)
        #print(topics.shape)
        pred_topics = next(item for item in pred_topic_file if item['image_id']== int(img_id))
        pred_topics = pred_topics['topic']
        pred_topics = torch.Tensor(pred_topics).to(device)
        data = {'image_id': int(0), 'caption': str(0)}
        # Move to GPU device, if available
        image_features = image_features.to(device)  # (1, 3, 256, 256)
        image_features_mean = image_features.mean(1)
        image_features_mean = image_features_mean.expand(k,2048)
        
        image_features = image_features.expand(k, 36, 2048)
        pred_topics = pred_topics.expand(k, 512)
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
        #h2, c2 = decoder.init_hidden_state(k)
        index= np.arange(512)
        concat = 'early'
        matchattn = 'True'
        attn = 'hidden' # word, 3words, hidden
        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        stacked_embeddings = []
        while True:

            embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)
            # print(image_features.shape, h1.shape)
            

            # embeddings = decoder.load_embeddings(glove, k_prev_words.cpu()).squeeze(1)
            # embeddings= embeddings.float()
            if attn == 'hidden':
                matchattn_inputs, _ =decoder.miml(image_features,h1) 
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
                matchattn_inputs=decoder.miml(image_features,concat_embeddings)
            else: ##word
                matchattn_inputs=decoder.miml(image_features,embeddings)#
            # 
            # print(topic_output)
            # topic_output = F.softmax(topic_output, dim=1)
            if (concat=='early'):
                if (matchattn=='False'):
                    h1,c1 = decoder.early_language_model(
                        torch.cat([attention_weighted_encoding, embeddings], dim=1),(h1,c1))
                    scores = decoder.fc(h1)
                else:
                    h1, c1 = decoder.early_language_model(
                        torch.cat([matchattn_inputs,embeddings], dim=1), (h1, c1))
                    # topic_embedding = decoder.topic_embedding(topics)
                    # scores = decoder.fc(torch.cat([h1,topics],dim=1))
                    scores = decoder.fc(h1)
            else:
                h1, c1 = decoder.late_language_model(embeddings, (h1, c1))
                h2 = torch.cat([h1, attention_weighted_encoding, topic_output], dim=1)
                scores = decoder.fc2(h2)  # (s, vocab_size)

            scores = F.log_softmax(scores, dim=1)
            # print(topic_output)
            # print(topics)
            #
            # plt.subplot(1, 2, 1)
            # plt.bar(index,topic_output[0].cpu().detach().numpy().squeeze())
            # plt.title('Predicted Topics')
            # plt.subplot(1, 2, 2)
            # plt.bar(index,topics[0].cpu().detach().numpy().squeeze())
            # plt.title('Ground-Truth Topics')
            # plt.show()
            # raw_input('enter')
            # Add
            scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

            # For the first step, all k points will have the same scores (since same k previous words, h, c)
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
            else:
                # Unroll and find top scores, and their unrolled indices
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

            # Convert unrolled indices to actual indices of scores
            prev_word_inds = top_k_words / vocab_size  # (s)
            next_word_inds = top_k_words % vocab_size  # (s)

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
            #h2 = h2[prev_word_inds[incomplete_inds]]
            #c2 = c2[prev_word_inds[incomplete_inds]]
            image_features = image_features[prev_word_inds[incomplete_inds]]
            image_features_mean = image_features_mean[prev_word_inds[incomplete_inds]]
            pred_topics = pred_topics[prev_word_inds[incomplete_inds]]
            topics = topics[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            # Break if things have been going on too long
            if step > 50:
                break
            step += 1
        # print(seqs)
        # s = [rev_word_map[w] for w in seqs[0]]
        # print(s)
        # input('enter')
        # print(complete_seqs_scores)
        try:
            i = complete_seqs_scores.index(max(complete_seqs_scores))
            seq = complete_seqs[i]

            # References
            img_caps = caps[0].tolist()
            img_captions = list(
                map(lambda c: [rev_word_map[w] for w in c if w not in {word_map['<start>'], word_map['<eoc>'], word_map['<pad>']}],
                    img_caps))  # remove <start> and pads
            img_caps = [' '.join(c) for c in img_captions]
            #print(img_caps)
            references.append(img_caps)

            # Hypotheses
            hypothesis = ([rev_word_map[w] for w in seq if w not in {word_map['<start>'], word_map['<eoc>'], word_map['<pad>']}])
            hypothesis = ' '.join(hypothesis)
            data = {"image_id": int(img_id), "caption": str(hypothesis)}
            # data['image_id'] = int(img_id)
            # data['caption'] = str(hypothesis)
            all_captions.append(data)
            # print(hypothesis, img_id)
            # print('enter')
            hypotheses.append(hypothesis)
            assert len(references) == len(hypotheses)
            # metrics_dict = nlgeval.compute_metrics(references, hypotheses)
            # print(metrics_dict)
            # input('enter')
        except:
            print('Empty sequence')
            continue
    # with open(cap_json_file, 'w') as outfile:
    #     json.dump(all_captions, outfile, indent=3)


    # Calculate scores
    metrics_dict = nlgeval.compute_metrics(references, hypotheses)
    print (metrics_dict)
    return metrics_dict


if __name__ == '__main__':
    beam_size = 5
    # DataLoader
    results_dir = os.path.join(checkpoint_folder, 'results_test.npy')
    results = {}
    loader = torch.utils.data.DataLoader(
        CaptionNewDictDataset(data_folder, data_name, 'TEST'),
        batch_size=1, shuffle=True, num_workers=16, pin_memory=torch.cuda.is_available())
    for ch in range(len(chkpt_files)):
        checkpoint_file = chkpt_files[ch]
        checkpoint = torch.load(checkpoint_file,map_location = device)
        decoder = checkpoint['decoder']
        decoder = decoder.to(device)
        decoder.eval()
        epoch = checkpoint['epoch']
        metrics_dict = evaluate(beam_size,decoder, loader)
        print(epoch)
        results[epoch] = metrics_dict
    np.save(results_dir, results)
