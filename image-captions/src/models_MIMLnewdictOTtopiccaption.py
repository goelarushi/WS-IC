import torch
from torch import nn
import torchvision
from torch.nn.utils.weight_norm import weight_norm


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MIML(nn.Module):
    def __init__(self,input_dim,decoder_dim):
        super(MIML,self).__init__()
        self.L = input_dim+decoder_dim
        self.D = 1024
        self.K = 128 #64 #512 #128 ### latent classes
        self.M = torch.Tensor([1.])
        self.iclassifier = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.D, self.K),            
        )
        
        # instance detector
        self.det = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.D, self.K),            
        )       

     # The default is log_sum-exp        
    def pool_func(self,y):
        #self.M = torch.ones(1) * 1.0
        # self.M = self.M.to(y.device)
        # y = (1/self.M) * torch.log(torch.sum(torch.exp(self.M*y),dim=1))  
        # Y_prob = torch.sigmoid(y)      
        # y = torch.sum(y, dim=1)  ### dim=1 when sum over instances     
        y = torch.sum(y, dim=2)  ### dim =2 when sum over classes
        # Y_prob = torch.sigmoid(y) 
        return y
    
    # this is the attension function ( In our case it is the Gaussian.)
    def get_att_func(self, x, m, s):
        z = (x - m)/s
        x =  torch.exp( -(z**2))        
        return x
    
    def forward(self, vis_input, embed_input):
        ## input_emd: [Batchsize, n_instances, input_dim]: [64, 36, 2048]
        ##GP0T0
        batch_size = vis_input.shape[0]
        embed_dim = embed_input.shape[1]
        embed_input = torch.unsqueeze(embed_input, dim=1)

        embed_input = embed_input.expand(batch_size, 36, embed_dim)
        miml_input = torch.cat([vis_input, embed_input], dim=2)
        self.x = miml_input
        self.y = self.iclassifier(self.x)

        # m = torch.mean(self.y, dim=1).unsqueeze(1)
        # s = torch.std(self.y,dim=1).unsqueeze(1)
        
        # z = self.get_att_func(self.y,m,s)
        # self.y = torch.mul(z,self.y)    
        # Y_prob = self.pool_func(self.y)
        # Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        # Y_hat = torch.ge(Y_prob, 0.5).float()      
 
        # return Y_prob, Y_hat, self.x
        z = self.det(self.x)        
        z = torch.softmax(z,dim=1)        
        y = torch.softmax(self.y, dim=2)
        
        y = torch.mul(z,y)    # [B, 36, K]
        att_both = y       
        Y_prob = self.pool_func(y)
        attention_weighted_encoding = (vis_input* Y_prob.unsqueeze(2)).sum(dim=1)
       
        
        Y_hat = torch.ge(Y_prob, 0.5).float()       

        return attention_weighted_encoding, att_both  

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
        self.topic_embedding_dim=256
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.matchattn_dim = 1024
        self.glove_weights = weights_matrix
        # self.match_attention = MatchingAttention(features_dim, embed_dim, self.topic_dim, self.matchattn_dim)
        # self.match_attention = AlignmentAttention(features_dim, embed_dim,self.matchattn_dim)   ###if concat word embeddings
        self.match_attention = AlignmentAttention(features_dim, decoder_dim,self.matchattn_dim) ### if concat hidden dim
        self.miml = MIML(features_dim, decoder_dim) ### if concat hidden dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        self.topic_embedding = nn.Sequential(nn.Linear(self.topic_dim, self.topic_embedding_dim),
                                             nn.ReLU(True))
        # self.topic_attention = AttnTopicModel(features_dim, attention_dim, self.topic_dim)
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

    def forward(self, image_features, encoded_captions, caption_lengths, topics,split):
        """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """
        self.cpi=5
        caption_lengths = caption_lengths.unsqueeze(1)
        image_features = self.repeat(image_features, [self.cpi, 1, 1])

        batch_size = image_features.size(0)
        vocab_size = self.vocab_size
        self.split = split
        concat = 'early' ##'late'
        attn = 'hidden'#'hidden' ## 'word'
        self.matchattn = 'True'
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
        if (concat == 'early'):
            if(self.matchattn == 'False'):
                for t in range(max(decode_lengths)):
                    batch_size_t = sum([l > t for l in decode_lengths])
                    h1, c1 = self.early_language_model(
                        torch.cat([weighted_input[:batch_size_t],embeddings[:batch_size_t, t, :]],
                                  dim=1),
                        (h1[:batch_size_t], c1[:batch_size_t]))

                    preds = self.fc(self.dropout(h1))  # (batch_size_t, vocab_size)
                    predictions[:batch_size_t, t, :] = preds
            else:
                for t in range(max(decode_lengths)):
                    batch_size_t = sum([l > t for l in decode_lengths])
                    if (attn == 'hidden'):
                        matchattn_vis_input,_ = self.miml(image_features[:batch_size_t],h1[:batch_size_t])
                    else:
                        matchattn_vis_input,_ = self.miml(image_features[:batch_size_t],embeddings[:batch_size_t,t,:])
                    if(split=='TRAIN'):
                        append_topics = topics[:batch_size_t]
                        # append_topics = self.topic_embedding(topics[:batch_size_t])
                    else:
                        append_topics = topics[:batch_size_t]
                        # append_topics = self.topic_embedding(topics[:batch_size_t])
                    
                    h1, c1 = self.early_language_model(
                        torch.cat(
                            [matchattn_vis_input[:batch_size_t],embeddings[:batch_size_t, t, :]],
                            dim=1),
                        (h1[:batch_size_t], c1[:batch_size_t]))

                    # preds = self.fc(self.dropout(torch.cat([h1,append_topics],dim=1)))  # (batch_size_t, vocab_size)

                    preds = self.fc(self.dropout(h1))

                    predictions[:batch_size_t, t, :] = preds
        else:
            for t in range(max(decode_lengths)):
                batch_size_t = sum([l > t for l in decode_lengths])
                h1, c1 = self.late_language_model(embeddings[:batch_size_t, t, :],
                    (h1[:batch_size_t], c1[:batch_size_t]))
                h2 = torch.cat([h1[:batch_size_t], weighted_input[:batch_size_t]],
                               dim=1)  ## updated concatenated hidden state for the output and the next hidden of the LSTM
                preds = self.fc2(self.dropout(h2))  # (batch_size_t, vocab_size)
                predictions[:batch_size_t, t, :] = preds

        return predictions, encoded_captions, decode_lengths, sort_ind, topics#,topic_output
