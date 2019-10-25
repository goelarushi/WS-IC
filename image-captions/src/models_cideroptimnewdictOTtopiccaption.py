import torch
from torch import nn
import torchvision
from torch.nn.utils.weight_norm import weight_norm

from torch import distributions

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Attention(nn.Module):
    def __init__(self,input_dim, attention_dim):
        super(Attention,self).__init__()
        self.input_dim = input_dim
        self.attention_dim = attention_dim
        self.attn = nn.Sequential(
                    nn.Linear(self.input_dim, self.attention_dim),
                    nn.Linear(self.attention_dim,1),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=0.5),

        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_emd):

        batch_size = input_emd.shape[0]
        input_emd = input_emd.contiguous()

        attn = self.attn(input_emd)  # attention over the sequence length
        #print(attn.shape)
        alpha = self.softmax(attn)  # gives the probability values for the time steps in the sequence (weights to each time step)

        #print(lstm_emd.shape,lstm_emd)
        attn_feature_map = input_emd * alpha  # gives attention weighted embedding
        attn_feature_map = torch.sum(attn_feature_map, dim=1)  # computes the weighted sum

        return attn_feature_map


class AlignmentAttention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, features_dim, decoder_dim, attention_dim, dropout=0.5):

        super(AlignmentAttention, self).__init__()
        self.features_att = (nn.Linear(features_dim, attention_dim))  # linear layer to transform encoded image
        self.decoder_att = (nn.Linear(decoder_dim, attention_dim))  # linear layer to transform decoder's output
        self.full_att = (nn.Linear(attention_dim, 1))  # linear layer to calculate values to be softmax-ed
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, image_features, embed_input):
        """
        Forward propagation.
        :param image_features: encoded images, a tensor of dimension (batch_size, 36, features_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self.features_att(image_features)  # (batch_size, 36, attention_dim)
        att2 = self.decoder_att(embed_input)  # (batch_size, attention_dim)
        att = self.full_att(self.tanh(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, 36)
        alpha = self.softmax(att)  # (batch_size, 36)
        attention_weighted_encoding = (image_features * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, features_dim)

        return attention_weighted_encoding


class MatchingAttention(nn.Module):
    def __init__(self, visual_dim, embed_dim, topic_dim, matchattn_dim):
        super(MatchingAttention,self).__init__()
        self.visual_dim = visual_dim
        self.embed_dim = embed_dim
        self.topic_dim = topic_dim
        self.matchattn_dim = matchattn_dim
        self.vis_linear = nn.Sequential(
                          nn.Linear(self.visual_dim+self.embed_dim, self.matchattn_dim),
                          nn.Linear(self.matchattn_dim, 1),
                          nn.ReLU(True))
        self.vis_weights = nn.Softmax(dim=1)

    def forward(self, vis_input, embed_input):
        batch_size = vis_input.shape[0]
        embed_dim = embed_input.shape[1]
        embed_input = torch.unsqueeze(embed_input, dim=1)

        embed_input = embed_input.expand(batch_size, 36, embed_dim)
        matchattn_input = torch.cat([vis_input, embed_input], dim=2)
        vis_attn = self.vis_linear(matchattn_input)
        vis_weights = self.vis_weights(vis_attn)
        matchattn_vis_inputs = vis_input*vis_weights
        matchattn_vis_inputs = torch.sum(matchattn_vis_inputs, dim=1)

        return matchattn_vis_inputs


class DecoderWithTopicAttention(nn.Module):
    """
    Decoder.
    """

    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size,weights_matrix,features_dim=2048, dropout=0.5):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param features_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(DecoderWithTopicAttention, self).__init__()

        self.features_dim = features_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.topic_dim = 512
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.matchattn_dim = 1024
        self.glove_weights = weights_matrix
        # self.match_attention = MatchingAttention(features_dim, embed_dim, self.topic_dim, self.matchattn_dim)
        self.match_attention = AlignmentAttention(features_dim, embed_dim,self.matchattn_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        self.softmax = nn.Softmax(dim=1)

        self.early_language_model = nn.LSTMCell(features_dim +self.embed_dim, decoder_dim,
                                          bias=True)  # language model LSTMCell
        self.late_language_model = nn.LSTMCell(self.embed_dim, decoder_dim,
                                                bias=True)  # language model LSTMCell
        self.fc = weight_norm(nn.Linear(decoder_dim, vocab_size))  # linear layer to find scores over vocabulary
        self.fc2 = weight_norm(nn.Linear(decoder_dim+features_dim+self.topic_dim, vocab_size))
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, batch_size):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        h = torch.zeros(batch_size, self.decoder_dim).cuda()#.to(device)  # (batch_size, decoder_dim)
        c = torch.zeros(batch_size, self.decoder_dim).cuda()#.to(device)
        return h, c

    def repeat(self,tensor, dims):
        if len(dims) != len(tensor.shape):
            raise ValueError("The length of the second argument must equal the number of dimensions of the first.")
        for index, dim in enumerate(dims):
            repetition_vector = [1] * (len(dims) + 1)
            repetition_vector[index + 1] = dim
            new_tensor_shape = list(tensor.shape)
            new_tensor_shape[index] *= dim
            tensor = tensor.unsqueeze(index + 1).repeat(repetition_vector).reshape(new_tensor_shape)
        return tensor


    def load_embeddings(self, weights_matrix, encoded_captions):

        embeddings = torch.cat([torch.index_select(torch.tensor(weights_matrix), 0, encoded_captions[i]).unsqueeze(0) for i in range(len(encoded_captions))])
        return embeddings.cuda()



    def forward(self,image_features, encoded_captions, caption_lengths, vocab_size, topics,split):
        """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """
        cpi=5
        caption_lengths = caption_lengths.unsqueeze(1)
        image_features = self.repeat(image_features, [cpi, 1, 1])

        batch_size = image_features.size(0)

        concat = 'early' ##'late'
        attn = 'hidden'#word'#'hidden' ## 'word' 3words
        matchattn = 'True'
        # Flatten image
        image_features_mean = image_features.mean(1) # .to(device)  # (batch_size, num_pixels, encoder_dim)

        # Sort input data by decreasing lengths; why? apparent below
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        image_features = image_features[sort_ind]

        image_features_mean = image_features_mean[sort_ind]
        encoded_captions = encoded_captions[sort_ind]
        topics = topics[sort_ind]
        # Train Embeddings
        embeddings = self.embedding(encoded_captions).cuda() # (batch_size, max_caption_length, embed_dim)
        ### Load Glove embeddings
        # embeddings = self.load_embeddings(self.glove_weights, encoded_captions.cpu())
        # embeddings = embeddings.float()
        ########
        # print(embeddings.shape)
        # Initialize LSTM state
        h1, c1 = self.init_hidden_state(batch_size)  # (batch_size, decoder_dim)
        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1

        decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word predicion scores
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).cuda() # .to(device)

        # topic_output = self.topic_attention(image_features)

        # At each time-step, pass the language model's previous hidden state, the mean pooled bottom up features and
        # word embeddings to the top down attention model. Then pass the hidden state of the top down model and the bottom up
        # features to the attention block. The attention weighed bottom up features and hidden state of the top down attention model
        # are then passed to the language model

        timestep_outputs = torch.zeros(batch_size, max(decode_lengths), 1).cuda()
        timestep_probs = torch.zeros(batch_size, max(decode_lengths), 1).cuda()
        if (concat == 'early'):
            if(matchattn == 'False'):
                for t in range(max(decode_lengths)):
                    batch_size_t = sum([l > t for l in decode_lengths])
                    h1, c1 = self.early_language_model(
                        torch.cat([weighted_input[:batch_size_t],embeddings[:batch_size_t, t, :]],
                                    dim=1),
                        (h1[:batch_size_t], c1[:batch_size_t]))

                    preds = self.fc(F.dropout(h1))  # (batch_size_t, vocab_size)
                    predictions[:batch_size_t, t, :] = preds
            else:
                for t in range(max(decode_lengths)):
                    outputs = []
                    log_probs = []
                    batch_size_t = sum([l > t for l in decode_lengths])
                    if (t == 0):
                        prev_outputs = embeddings[:batch_size_t,t,:]
                    if (attn == 'hidden'):
                        matchattn_vis_input = self.match_attention(image_features[:batch_size_t],h1[:batch_size_t])
                    elif(attn == '3words'):
                        
                        if ((t-1 < 0)): 
                            x = [t]
                        elif (t-2 < 0) and (t-1 >= 0):
                            x = [t, t-1]
                        else:
                            x = [t, t-1, t-2]
                        concat_embed  = [embeddings[:batch_size_t,j,:] for j in x]
                        concat_embed = torch.stack(concat_embed)
                        concat_embed = torch.mean(concat_embed, dim=0)
                        
                        matchattn_vis_input = self.match_attention(image_features[:batch_size_t],concat_embed)
                    else:
                        matchattn_vis_input = self.match_attention(image_features[:batch_size_t],embeddings[:batch_size_t,t,:])
                    # if(split=='TRAIN'):
                    #     append_topics = topics[:batch_size_t]
                    #     # append_topics = self.topic_embedding(topics[:batch_size_t])
                    # else:
                    #     append_topics = topics[:batch_size_t]
                        # append_topics = self.topic_embedding(topics[:batch_size_t])
                    h1, c1 = self.early_language_model(
                        torch.cat(
                            [matchattn_vis_input[:batch_size_t],prev_outputs[:batch_size_t]],
                            dim=1),
                        (h1[:batch_size_t], c1[:batch_size_t]))

                    # preds = self.fc(self.dropout(torch.cat([h1,append_topics],dim=1)))  # (batch_size_t, vocab_size)

                    preds = self.fc(self.dropout(h1))

                    log_preds  = self.softmax(preds)
                    for b_scores in log_preds:
                        distr = distributions.Categorical(logits=b_scores)
                        b_words = distr.sample()
                        b_log_scores = distr.log_prob(b_words)
                        outputs.append(b_words)
                        log_probs.append(b_log_scores)
                    outputs = torch.stack(outputs).unsqueeze(1)
                    log_probs = torch.stack(log_probs).unsqueeze(1)
                    prev_outputs = self.embedding(outputs).squeeze(1)
                    
                    timestep_outputs[:batch_size_t,t,:] = outputs
                    timestep_probs[:batch_size_t, t, :] = log_probs
                    predictions[:batch_size_t, t, :] = preds
                # print(timestep_outputs)
        else:
            for t in range(max(decode_lengths)):
                batch_size_t = sum([l > t for l in decode_lengths])
                h1, c1 = self.late_language_model(embeddings[:batch_size_t, t, :],
                    (h1[:batch_size_t], c1[:batch_size_t]))
                h2 = torch.cat([h1[:batch_size_t], weighted_input[:batch_size_t]],
                                dim=1)  ## updated concatenated hidden state for the output and the next hidden of the LSTM
                preds = self.fc2(F.dropout(h2))  # (batch_size_t, vocab_size)
                predictions[:batch_size_t, t, :] = preds

        return predictions, timestep_outputs, timestep_probs, encoded_captions, decode_lengths, sort_ind, topics#,topic_output
