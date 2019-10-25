import torch
from torch import nn
import torchvision
from torch.nn.utils.weight_norm import weight_norm
import torch.nn.functional as F
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, embed_dim):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRUCell(embed_dim, hidden_size)

    def forward(self, embedded, hidden):
        output = embedded
        hidden = self.gru(output, hidden)
        output = hidden
        return output, hidden

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size,embed_dim):
        super(DecoderRNN, self).__init__()

        self.gru = nn.GRUCell(embed_dim, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        # self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = input
        hidden = self.gru(output, hidden)
        output = hidden
        output = self.out(output)
        # output = self.softmax(self.out(output[0]))
        return output, hidden

class AttnDecoderRNN(nn.Module):
     def __init__(self, hidden_size, output_size,embed_dim,dropout_p=0.1, max_length=52):
         super(AttnDecoderRNN, self).__init__()
         self.hidden_size = hidden_size
         self.output_size = output_size
         self.dropout_p = dropout_p
         self.max_length = max_length
         self.embed_dim = 1024
         self.attn = nn.Linear(self.hidden_size + self.embed_dim, self.max_length)
         self.attn_combine = nn.Linear(self.hidden_size + self.embed_dim, self.hidden_size)
         self.dropout = nn.Dropout(self.dropout_p)
         self.gru = nn.GRUCell(self.hidden_size, self.hidden_size)
         self.out = nn.Linear(self.hidden_size, self.output_size)

     def forward(self, input, hidden, encoder_outputs):
         embedded = input
         embedded = self.dropout(embedded)

         attn_weights = F.softmax(
             self.attn(torch.cat((embedded, hidden), 1)), dim=1)
#         print(attn_weights.shape)
#         print(encoder_outputs.shape)
         e_len = encoder_outputs.shape[1]
         a_zeros = torch.zeros([encoder_outputs.shape[0],self.max_length-e_len,self.hidden_size]).to(device)
         e_outputs = torch.cat((encoder_outputs,a_zeros),1)
         attn_applied = torch.bmm(attn_weights.unsqueeze(1),e_outputs)

         output = torch.cat((embedded, attn_applied.squeeze(1)), 1)
         output = self.attn_combine(output)

         output = F.relu(output)
         hidden = self.gru(output, hidden)
         output = hidden
         output = self.out(output)
         return output, hidden, attn_weights


class TranslationModel(nn.Module):
    """
    Decoder.
    """

    def __init__(self, attention_dim, embed_dim, hidden_dim, vocab_size,word_map,weights_matrix,features_dim=2048, dropout=0.5):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param features_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(TranslationModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.glove_weights = weights_matrix
        self.word_map = word_map
        self.embedding = nn.Embedding(self.vocab_size, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        self.encoder_rnn =EncoderRNN(self.vocab_size, self.hidden_dim,self.embed_dim)
                                      
        self.decoder_rnn =AttnDecoderRNN(self.hidden_dim, self.vocab_size,self.embed_dim)  

        # self.fc = weight_norm(nn.Linear(decoder_dim, vocab_size))  # linear layer to find scores over vocabulary
        # self.fc2 = weight_norm(nn.Linear(decoder_dim+features_dim+self.topic_dim, vocab_size))
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        # self.fc.bias.data.fill_(0)
        # self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, batch_size):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        h = torch.zeros(batch_size,self.hidden_dim).to(device)  # (batch_size, decoder_dim)
        return h

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
    def get_prev_word_from_decoder(self,decoder_output):
    
        topv, topi = decoder_output.topk(1)
    
        decoder_input = topi.detach()  # detach from history as input
        
        d_i_embedding = self.embedding(decoder_input).to(device)
        # if decoder_input.item() == word_map['<eos>']:
        #     break
        return d_i_embedding.squeeze(1)


    def load_embeddings(self, weights_matrix, encoded_captions):

        embeddings = torch.cat([torch.index_select(torch.tensor(weights_matrix), 0, encoded_captions[i]).unsqueeze(0) for i in range(len(encoded_captions))])
        return embeddings.cuda()

    
    def forward(self, encoded_captions, caption_lengths,split):
        """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """
        self.cpi=5
        caption_lengths = caption_lengths.unsqueeze(1)
        batch_size = encoded_captions.size(0)
        vocab_size = self.vocab_size
        self.split = split
        teacher_forcing_ratio = 0.5
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        # Sort input data by decreasing lengths; why? apparent below
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoded_captions = encoded_captions[sort_ind]
     
        # Train Embeddings
        embeddings = self.embedding(encoded_captions).cuda() # (batch_size, max_caption_length, embed_dim)
        ### Load Glove embeddings
        # embeddings = self.load_embeddings(self.glove_weights, encoded_captions.cpu())
        # embeddings = embeddings.float()

        ########
        # print(embeddings.shape)
        # Initialize Encoder State
        h_e = self.init_hidden_state(batch_size)  # (batch_size, decoder_dim)
        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
    
        decode_lengths = (caption_lengths - 1).tolist()
        # Encode the whole batch
        encoder_hiddens = torch.zeros(batch_size, max(decode_lengths), self.hidden_dim).to(device)
        encoder_outputs = torch.zeros(batch_size, max(decode_lengths), self.hidden_dim).to(device)
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            e_o, h_e = self.encoder_rnn(embeddings[:batch_size_t, t, :], h_e[:batch_size_t])
            encoder_hiddens[:batch_size_t,t,:] = h_e
            encoder_outputs[:batch_size_t,t,:] = e_o

        # Decode the whole batch
        decoder_outputs = torch.zeros(batch_size, max(decode_lengths), self.vocab_size).to(device)
        for d in range(max(decode_lengths)):
        
            batch_size_d = sum([l > d for l in decode_lengths])#
            h_de = torch.stack([encoder_hiddens[b, m-1,:] for b,m in enumerate(decode_lengths)]) if d==0 else h_d[:batch_size_d]
            # h_de = encoder_hiddens[:batch_size_d,-1,:] if d==0 else h_d[:batch_size_d]
            if(self.split=='TRAIN'):
                #### For Scheduled Sampling during Training
                if use_teacher_forcing:
                    d_i = self.embedding(torch.tensor([self.word_map['<start>']]).expand(batch_size_d).unsqueeze(1).to(device)).squeeze(1) if d==0 else embeddings[:batch_size_d, d, :]
                else:
                    d_i = self.embedding(torch.tensor([self.word_map['<start>']]).expand(batch_size_d).unsqueeze(1).to(device)).squeeze(1) if d==0 else self.get_prev_word_from_decoder(d_o[:batch_size_d])
            else:
                d_i = self.embedding(torch.tensor([self.word_map['<start>']]).expand(batch_size_d).unsqueeze(1).to(device)).squeeze(1) if d==0 else self.get_prev_word_from_decoder(d_o[:batch_size_d])
                # print(d_i.shape)

            e_o = encoder_outputs[:batch_size_d,:,:]
            
            d_o, h_d, attn_weights = self.decoder_rnn(d_i,h_de,e_o)
            decoder_outputs[:batch_size_d,d,:] = d_o
        

        return decoder_outputs, encoded_captions, decode_lengths, sort_ind


