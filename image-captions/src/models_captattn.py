import torch
from torch import nn
import numpy as np
import torchvision
from torch.nn.utils.weight_norm import weight_norm
device = 'cuda'


class CaptionAttention(nn.Module):
    # caption input dim: batchsize,#captions,caplen
    # image input: batchsize, feature_dim

    def __init__(self, embed_dim, feature_dim):
        super(CaptionAttention,self).__init__()
        self.caplen = embed_dim
        self.feature_dim = feature_dim
        self.attention_dim = 512
        self.capt_attn = nn.Sequential(
                         nn.Linear(self.caplen+self.feature_dim, self.attention_dim),
                         nn.Linear(self.attention_dim, 1),
                         nn.ReLU(True))
        self.capt_weights = nn.Softmax(dim=1)

    def forward(self, vis_input, captions):
        image_features_mean = vis_input.mean(1)
        batch_size = vis_input.shape[0]
        image_features_mean = torch.unsqueeze(image_features_mean, dim=1)
        image_features_mean = image_features_mean.expand(batch_size, captions.shape[1],self.feature_dim)
        #print(image_features_mean.shape)
        attn_input = torch.cat([captions, image_features_mean], dim=2)
        capt_attn = self.capt_attn(attn_input.to(device))
        capt_weights = self.capt_weights(capt_attn)
        # attn_capt_inputs = captions * capt_weights
        # attn_capt_inputs = torch.sum(attn_capt_inputs, dim=1)

        return capt_weights

class ImageAttention(nn.Module):
    def __init__(self,input_dim,attention_dim):
        super(ImageAttention,self).__init__()
        self.input_dim = input_dim
        self.attention_dim = 1024
        self.attn = nn.Sequential(
                    nn.Linear(self.input_dim, self.attention_dim),
                    nn.Linear(self.attention_dim,1),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=0.5),

        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, vis_input):

        batch_size = vis_input.shape[0]
        vis_input = vis_input.contiguous()

        attn = self.attn(vis_input)  # attention over the sequence length

        alpha = self.softmax(attn)  # gives the probability values for the time steps in the sequence (weights to each time step)
        attn_feature_map = vis_input * alpha  # gives attention weighted embedding
        attn_feature_map = torch.sum(attn_feature_map, dim=1)  # computes the weighted sum
        return attn_feature_map

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


class AttnTopicModel(nn.Module):
    def __init__(self, input_dim, attention_dim, decoder_dim, num_topics):
        super(AttnTopicModel, self).__init__()
        self.input_dim = input_dim
        self.topics = num_topics
        self.attention_dim =attention_dim
        self.attention = ImageAttention(self.input_dim, self.attention_dim)
        self.topic_model = nn.Sequential(
                           nn.Linear(self.input_dim, 1024),
                           nn.ReLU(True),
                           nn.Linear(1024, self.topics))

    def forward(self, input):
        input_attn = self.attention(input)
        output = self.topic_model(input_attn)
        return input_attn, output

class CaptionLSTM(nn.Module):

    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size,weights_matrix,features_dim=2048, dropout=0.5):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param features_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(CaptionLSTM, self).__init__()

        self.features_dim = features_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.topic_dim = 512
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.matchattn_dim = 1024
        self.glove_weights = weights_matrix
        self.attention = AttnTopicModel(features_dim, attention_dim, self.decoder_dim, self.topic_dim)
        self.match_attention = MatchingAttention(features_dim, embed_dim, self.topic_dim, self.matchattn_dim)
        self.capt_attn = CaptionAttention(embed_dim, self.features_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)

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
        h = torch.zeros(batch_size, self.decoder_dim).cuda()  # .to(device)  # (batch_size, decoder_dim)
        c = torch.zeros(batch_size, self.decoder_dim).cuda()  # .to(device)
        return h, c

    def load_embeddings(self, weights_matrix, encoded_captions):

        embeddings = torch.cat([torch.index_select(torch.tensor(weights_matrix), 0, encoded_captions[i]).unsqueeze(0) for i in range(len(encoded_captions))])
        return embeddings.cuda()

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

    def forward(self, image_features, encoded_captions, caption_lengths, topics, split,cpi):
        """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """
        batch_size = image_features.size(0)
        vocab_size = self.vocab_size
        self.split = split
        concat = 'early'  ##'late'
        self.matchattn = 'True'
        self.cpi = cpi
        topics = topics.cuda()
        # Flatten image
        image_features_mean = image_features.mean(1).cuda()  # .to(device)  # (batch_size, num_pixels, encoder_dim)
        if(split == 'TRAIN'):
            train_image_features = self.repeat(image_features, [self.cpi, 1, 1])
            train_caption_lengths = caption_lengths.view(-1)
            sorted_decode_lengths = (train_caption_lengths-1).tolist()
            train_encoded_captions = encoded_captions.view(-1, encoded_captions.shape[2])

            train_caption_lengths, sort_ind = train_caption_lengths.sort(dim=0, descending=True)
            _, org_idx = sort_ind.sort(dim=0, descending=False)
            # print(org_idx)
            train_encoded_captions = train_encoded_captions[sort_ind]
            train_image_features = train_image_features[sort_ind]

            train_decode_lengths = (train_caption_lengths - 1).tolist()
            ## Train embeddings
            embeddings = self.embedding(train_encoded_captions)  ## B*5,words,300
            ## Convert embeddings to original caption batches of size B,5,words,300

            # attn_embeddings = embeddings[org_idx]
            # attn_embeddings = attn_embeddings.view(batch_size, 5, embeddings.shape[1], self.embed_dim)
            # attn_embeddings = torch.mean(attn_embeddings, dim=2)
            # print(attn_embeddings.shape)
            # capt_weights = self.capt_attn(image_features, attn_embeddings)

            batch_size = embeddings.shape[0]


        else:

            # Sort input data by decreasing lengths; why? apparent below
            train_caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
            train_image_features = image_features[sort_ind]
            train_encoded_captions = encoded_captions[sort_ind]
            org_idx = sort_ind
            # Train Embeddings
            embeddings = self.embedding(train_encoded_captions) # (batch_size, max_caption_length, embed_dim)
            ### Load Glove embeddings
            # embeddings = self.load_embeddings(self.glove_weights, train_encoded_captions.cpu())
            # embeddings = embeddings.float()
            train_decode_lengths = (train_caption_lengths - 1).tolist()
            batch_size = embeddings.shape[0]
            # Create tensors to hold word predicion scores
        predictions = torch.zeros(batch_size, max(train_decode_lengths), vocab_size).cuda()  # .to(device)
        final_h = torch.zeros(batch_size, self.decoder_dim).to(device)

        # Initialize LSTM state
        h1, c1 = self.init_hidden_state(batch_size)  # (batch_size, decoder_dim)
        weighted_input, topic_output = self.attention(image_features)
        # At each time-step, pass the language model's previous hidden state, the mean pooled bottom up features and
        # word embeddings to the top down attention model. Then pass the hidden state of the top down model and the bottom up
        # features to the attention block. The attention weighed bottom up features and hidden state of the top down attention model
        # are then passed to the language model
        if (concat == 'early'):
            if(self.matchattn == 'False'):
                for t in range(max(train_decode_lengths)):
                    batch_size_t = sum([l > t for l in train_decode_lengths])
                    h1, c1 = self.early_language_model(
                        torch.cat([weighted_input[:batch_size_t], topics[:batch_size_t],embeddings[:batch_size_t, t, :]],
                                  dim=1),
                        (h1[:batch_size_t], c1[:batch_size_t]))

                    preds = self.fc(self.dropout(h1))  # (batch_size_t, vocab_size)
                    predictions[:batch_size_t, t, :] = preds
            else:
                for t in range(max(train_decode_lengths)):
                    batch_size_t = sum([l > t for l in train_decode_lengths])
                    matchattn_vis_input = self.match_attention(train_image_features[:batch_size_t],embeddings[:batch_size_t,t,:])
                    h1, c1 = self.early_language_model(
                        torch.cat(
                            [matchattn_vis_input[:batch_size_t], embeddings[:batch_size_t, t, :]],
                            dim=1),
                        (h1[:batch_size_t], c1[:batch_size_t]))
                    # print(h1.shape)
                    final_h[:batch_size_t] = h1
                    preds = self.fc(self.dropout(h1))  # (batch_size_t, vocab_size)
                    predictions[:batch_size_t, t, :] = preds
                final_h = final_h[org_idx]
                final_h = final_h.view(image_features.shape[0], -1, self.decoder_dim)
                        # according to original sequence not sorted
                capt_weights = self.capt_attn(image_features, final_h)

        else:
            for t in range(max(train_decode_lengths)):
                batch_size_t = sum([l > t for l in train_decode_lengths])
                h1, c1 = self.late_language_model(embeddings[:batch_size_t, t, :],
                    (h1[:batch_size_t], c1[:batch_size_t]))
                h2 = torch.cat([h1[:batch_size_t], weighted_input[:batch_size_t], topics[:batch_size_t]],
                               dim=1)  ## updated concatenated hidden state for the output and the next hidden of the LSTM
                preds = self.fc2(self.dropout(h2))  # (batch_size_t, vocab_size)
                predictions[:batch_size_t, t, :] = preds

        if (split == 'TRAIN'):
            return predictions[org_idx], train_encoded_captions[org_idx], capt_weights,  sorted_decode_lengths, sort_ind, org_idx, topic_output
        else:
            return predictions, train_encoded_captions,capt_weights, train_decode_lengths, sort_ind, topic_output




    #
    # def __init__(self,embed_dim, feature_dim, decoder_dim,vocab_size,dropout=0.5):
    #     super(CaptionLSTM, self).__init__()
    #     self.embed_dim = embed_dim
    #     self.feature_dim = feature_dim
    #     self.decoder_dim = decoder_dim
    #     self.dropout = dropout
    #     self.matchattn_dim = 1024
    #     self.topic_dim=512
    #     self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
    #     self.dropout = nn.Dropout(p=self.dropout)
    #     self.vocab_size = vocab_size
    #     self.capt_attn = CaptionAttention(embed_dim, self.feature_dim)
    #     self.match_attention = MatchingAttention(self.feature_dim, embed_dim, self.topic_dim,self.matchattn_dim)
    #     self.image_attn = ImageAttention(self.feature_dim)
    #     self.early_language_model = nn.LSTMCell(self.feature_dim + self.embed_dim, decoder_dim,
    #                                             bias=True)  # language model LaSTMCell
    #     self.fc = weight_norm(nn.Linear(decoder_dim, vocab_size))
	# self.init_weights()  # initialize some layers with the uniform distribution
    #
    # def init_weights(self):
    #     """
    #     Initializes some parameters with values from the uniform distribution, for easier convergence.
    #     """
    #     self.embedding.weight.data.uniform_(-0.1, 0.1)
    #     self.fc.bias.data.fill_(0)
    #     self.fc.weight.data.uniform_(-0.1, 0.1)
    #
    # def init_hidden_state(self, batch_size):
    #     """
    #     Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.
    #     :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
    #     :return: hidden state, cell state
    #     """
    #     h = torch.zeros(batch_size, self.decoder_dim).to(device)  # (batch_size, decoder_dim)
    #     c = torch.zeros(batch_size, self.decoder_dim).to(device)
    #     return h, c
    #
    # def repeat(self,tensor, dims):
    #     if len(dims) != len(tensor.shape):
    #         raise ValueError("The length of the second argument must equal the number of dimensions of the first.")
    #     for index, dim in enumerate(dims):
    #         repetition_vector = [1] * (len(dims) + 1)
    #         repetition_vector[index + 1] = dim
    #         new_tensor_shape = list(tensor.shape)
    #         new_tensor_shape[index] *= dim
    #         tensor = tensor.unsqueeze(index + 1).repeat(repetition_vector).reshape(new_tensor_shape)
    #     return tensor
    #
    # def forward(self, image_features, captions, caption_lengths, split):
    #     ### Flatten captions and caption lengths and then unflatten later
    #     ### Flatten captions and caption lengths and then unflatten later
    #     batch_size = image_features.shape[0]
    #     ### Get attented image features and repear each image feature over 5 captions
    #     # weighted_image_features = self.image_attn(image_features)
    #     # weighted_image_features = self.repeat(weighted_image_features, [5, 1])
    #     if (split == 'TRAIN'):
    #         rep_image_features = self.repeat(image_features, [5, 1, 1])
    #         batch_caption_lengths = caption_lengths.view(-1)
    #         encoded_captions = captions.view(-1,captions.shape[2])
    #
    #         sort_caption_lengths, sort_ind = batch_caption_lengths.sort(dim=0, descending=True)
    #         _,org_idx = sort_ind.sort(dim=0, descending=False)
    #         # print(org_idx)
    #         encoded_captions = encoded_captions[sort_ind]
    #         # encoded_captions = [capt[id] for capt, id in zip(encoded_captions, sort_ind)]
    #         # encoded_captions = torch.stack(encoded_captions)
    #
    #         # weighted_image_features = weighted_image_features[sort_ind]
    #
    #         sort_image_features = rep_image_features[sort_ind]
    #         # sort_image_features = [feat[id] for feat, id in zip(image_features, sort_ind)]
    #         # sort_image_features = torch.stack(sort_image_features)
    #
    #         decode_lengths = (sort_caption_lengths-1).tolist()
    #         ## Train embeddings
    #         embeddings = self.embedding(encoded_captions)   ## B*5,words,300
    #         ## Convert embeddings to original caption batches of size B,5,words,300
    #
    #         attn_embeddings = embeddings[org_idx]
    #         attn_embeddings = attn_embeddings.view(batch_size, 5,embeddings.shape[1], self.embed_dim)
    #         attn_embeddings = torch.mean(attn_embeddings, dim=2)
    #         # print(attn_embeddings.shape)
    #         # capt_weights = self.capt_attn(image_features, attn_embeddings)
    #
    #         flat_batch_size = embeddings.shape[0]
    #
    #
    #         predictions = torch.zeros(flat_batch_size, max(decode_lengths), self.vocab_size).to(device)
    #         h, c = self.init_hidden_state(flat_batch_size)
    #         final_h = torch.zeros(flat_batch_size, self.decoder_dim).to(device)
    #
    #         for t in range(max(decode_lengths)):
    #             batch_size_t = sum([l > t for l in decode_lengths])
    #             matchattn_vis_input = self.match_attention(sort_image_features[:batch_size_t],
    #                                                        embeddings[:batch_size_t, t, :])
    #             h, c = self.early_language_model(
    #                 torch.cat(
    #                     [matchattn_vis_input[:batch_size_t], embeddings[:batch_size_t, t, :]],
    #                     dim=1),
    #                 (h[:batch_size_t], c[:batch_size_t]))
    #
    #             preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
    #             final_h[:batch_size_t] = h
    #             predictions[:batch_size_t, t, :] = preds
    #         final_h = final_h[org_idx]
    #         final_h = final_h.view(batch_size, -1, self.decoder_dim)
    #         # according to original sequence not sorted
    #         capt_weights = self.capt_attn(image_features, final_h)
    #
    #         # print(capt_weights.shape)
    #         sorted_decode_lengths = (batch_caption_lengths-1).tolist()
    #         return predictions[org_idx], encoded_captions[org_idx],capt_weights,sorted_decode_lengths, sort_ind, org_idx
    #     else:
    #         # caption_lengths,arg_lengths = torch.max(caption_lengths, dim=1)
    #         # encoded_captions = [captions[i, arg_lengths[i], :] for i in range(len(arg_lengths))]
    #         # encoded_captions = torch.stack(encoded_captions)
    #         # caption_lengths, sort_ind = caption_lengths.sort(dim=0, descending=True)
    #         # sort_image_features = [feat[id] for feat, id in zip(image_features, sort_ind)]
    #         # sort_image_features = torch.stack(sort_image_features)
    #         # weighted_image_features = self.image_attn(image_features)
    #         # weighted_image_features = weighted_image_features[sort_ind]
    #         caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
    #         sort_image_features = image_features[sort_ind]
    #         encoded_captions = captions[sort_ind]
    #
    #         embeddings = self.embedding(encoded_captions)
    #
    #         decode_lengths = (caption_lengths-1).tolist()
    #         predictions = torch.zeros(batch_size, max(decode_lengths), self.vocab_size).to(device)
    #         h, c = self.init_hidden_state(batch_size)
    #
    #         for t in range(max(decode_lengths)):
    #             batch_size_t = sum([l > t for l in decode_lengths])
    #             matchattn_vis_input = self.match_attention(sort_image_features[:batch_size_t],
    #                                                        embeddings[:batch_size_t, t, :])
    #             h, c = self.early_language_model(
    #                 torch.cat(
    #                     [matchattn_vis_input[:batch_size_t], embeddings[:batch_size_t, t, :]],
    #                     dim=1),
    #                 (h[:batch_size_t], c[:batch_size_t]))
    #
    #             preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
    #             predictions[:batch_size_t, t, :] = preds
    #
    #         return predictions, encoded_captions, decode_lengths, sort_ind
    #
    #
    #







