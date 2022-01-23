# This is the encoder decoder model with attention
# The encoder creates the encoded representation of the textual sequence 
# The decoder generates the face landmark predictions for the input textual sequence 
import torch
import torch.nn as nn
import torch.nn.functional as F
from text import symbols

class Encoder(nn.Module):
    def __init__(self, embedding_dim, in_channels, out_channels, kernel_size, decoder_dim):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(len(symbols), embedding_dim)
        self.dropout = nn.Dropout(0.5)

        convs = 3
        convolutions = list()
        for j in range(convs):
            conv_layer = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=int((kernel_size-1)/2)),
                nn.BatchNorm1d(out_channels)
                )
            # conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=int((kernel_size-1)/2))

            convolutions.append(conv_layer)

        self.conv = nn.ModuleList(convolutions)
        self.rnn = nn.GRU(out_channels, out_channels//2, 1, batch_first=True, bidirectional=True)
        self.linear = nn.Linear((out_channels//2) * 2, decoder_dim)
    
    def forward(self, x):
        # x.shape -> batch_size x seq_len
        x = self.dropout(self.embedding(x).permute(0, 2, 1)) # x.shape -> batch_size x embedding_dim x seq_len

        for conv in self.conv:
            x = self.dropout(F.relu(conv(x)))

        # x.shape -> batch_size x out_channels x seq_len
        x = x.permute(0, 2, 1) # x.shape -> batch_size x seq_len x out_channels

        outputs, hidden = self.rnn(x)
        # outputs.shape -> batch_size x seq_len x encoder_hidden_dim * num_directions 
        # hidden.shape -> num_layers * num_directions x batch_size x encoder_hidden_dim

        # concatenate the forward and backward hidden represenation from the last layer and pass through linear layer
        h_concat = torch.cat([hidden[-2,:,:], hidden[-1,:,:]], dim=1)
        hidden = torch.tanh(self.linear(h_concat)) # hidden.shape -> batch_size x decoder_dim

        return outputs, hidden

# The attention module takes the encoder hidden states and the previous decoder state
# And computes a vector of scores called the attention weights
class Attention(nn.Module):
    def __init__(self, encoder_hidden_dim, decoder_dim):
        super(Attention, self).__init__()
        # print(f'Encoder dim : {encoder_hidden_dim}, decoder dim : {decoder_dim}')
        self.attn = nn.Linear(2*encoder_hidden_dim + decoder_dim, decoder_dim)
        self.v = nn.Linear(decoder_dim, 1, bias=False)

    def forward(self, encoder_states, decoder_hidden):
        # encoder_states.shape -> batch_size x seq_len x encoder_dim*2
        # decoder_hidden.shape -> batch_size x decoder_dim
        # repeat the decoder_hidden seq_len number of times 
        src_len = encoder_states.shape[1]
        # print(f'decoder hidden : {decoder_hidden.shape}')
        decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)
        # print(f'Reached here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1')
        # decoder_hidden.shape -> batch_size x seq_len x decoder_dim
        # print(f'Encoder states : {encoder_states.shape}, decoder_hidden : {decoder_hidden.shape}')
        combined = torch.cat([encoder_states, decoder_hidden], dim=2) # batch_size x seq_len x (2*encoder_dim + decoder_dim)
        # print(f'Combined : {combined.shape}')
        energies = torch.tanh(self.attn(combined))

        # print(f'Energies : {energies.shape}')
        # energies.shape -> batch_size x seq_len x decoder_dim
        attention = self.v(energies).squeeze(2)
        # print(f'Attention : {attention.shape}')
        # attention.shape -> batch_size x seq_len

        return F.softmax(attention, dim=1)

class Decoder(nn.Module):
    def __init__(self, attention, encoder_hidden_dim, decoder_dim, image_dim, dropout=0.5):
        super(Decoder, self).__init__()
        self.rnn = nn.GRU(2*encoder_hidden_dim + image_dim*image_dim, decoder_dim, 1, batch_first=True)
        self.attention = attention
        self.fsout = nn.Linear(decoder_dim, image_dim*image_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input, encoder_outputs, decoder_input):
        # decoder_input and encoder_outputs are used to compute the attention 
        # decoder_input.shape -> batch_size x decoder_dim, encoder_outputs.shape -> batch_size x seq_len x 2*encoder_dim
        # print(f'Input : {input.shape}, encoder_outputs : {encoder_outputs.shape}, decoder_input : {decoder_input.shape}')
        attention_weights = self.attention(encoder_outputs, decoder_input).unsqueeze(1)

        # attention_weights.shape -> batch_size x 1 x seq_len

        # use the attention weights and encoder_outputs to compute the context vector 
        weights = torch.bmm(attention_weights, encoder_outputs)
        # print(f'Weights : {weights.shape}')

        # weights.shape -> batch_size x 1 x encoder_hidden * 2

        # concatenate the context vector and input 
        # input.shape -> batch_size x image_dim x image_dim
        input = input.view(input.shape[0], -1).unsqueeze(1)
        # print(f'Input shape : {input.shape}')
        # print(f'Weights shape : {weights.shape}')
        # input.shape -> batch_size x 1 x image_dim*image_dim

        output, hidden = self.rnn(torch.cat([input, weights], dim=2), decoder_input.unsqueeze(0))
        # print(f'Output -> {output.shape}, hidden -> {hidden.shape}')
        # output.shape -> batch_size x 1 x decoder_dim, hidden.shape -> num_layer x batch_size x decoder_dim
        output = self.fsout(output.squeeze(1)) # batch_size x image_dim*image_dim
        # print(f'Final output -> {output.shape}')
        return output, hidden.squeeze(0)

class Seq2Seq(nn.Module):
    def __init__(self, encoder, attention, decoder, image_dim):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.attention = attention
        self.decoder = decoder
        self.image_dim = image_dim

    def forward(self, src, src_len, trg, trg_len):
        encoder_outputs, hidden = self.encoder(src)
        # encoder_outputs.shape -> batch_size x seq_len x encoder_hidden_dim*2
        # encoder_hidden.shape -> batch_size x decoder_dim
        # print(f'Encoder outputs -> {encoder_outputs.shape}, encoder_hidden : {hidden.shape}')

        target_length = trg.shape[1]
        batch_size = trg.shape[0]
        output_predictions = torch.zeros(batch_size, target_length, self.image_dim*self.image_dim)

        # trg.shape -> batch_size x trg_len x image_dim x image_dim
        input = trg[:, 0] # batch_size x image_dim x image_dim
        for j in range(1, target_length):
            # generate the predictions 
            output, hidden = self.decoder(input, encoder_outputs, hidden)
            output_predictions[:, j] = output
            input = trg[:, j]

        return output_predictions