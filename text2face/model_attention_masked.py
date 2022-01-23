# Trains the text2face model
# Generates a sequence of facial landmarks for the input text sequence 
import torch
import torch.nn as nn
import torch.nn.functional as F
from text import symbols

class Encoder(nn.Module):
    def __init__(self, embedding_dim, out_channels, kernel_size, encoder_hidden_dim, decoder_hidden_dim, dropout=0.5):
        super(Encoder, self).__init__()
        
        # generate the embedding 
        self.embedding = nn.Embedding(len(symbols), embedding_dim)

        self.dropout = nn.Dropout(dropout)
        # apply the sequence of 1D convolutions for n-gram language model 
        convs = 3
        convolutions = list()
        for j in range(convs):
            convolution = nn.Sequential(
                nn.Conv1d(embedding_dim, out_channels, kernel_size, padding=(kernel_size-1)//2),
                nn.BatchNorm1d(out_channels)
                )

            convolutions.append(convolution)
        
        self.conv = nn.ModuleList(convolutions)
        self.rnn = nn.GRU(out_channels, encoder_hidden_dim, bidirectional=True, batch_first=True)

        self.fc = nn.Linear(2*encoder_hidden_dim, decoder_hidden_dim) # combine the forward and backward RNNs
    
    def forward(self, x):
        # x.shape -> batch_size x seq_len 
        x = self.dropout(self.embedding(x).permute(0, 2, 1)) # embedding.shape -> batch_size x embedding_dim x seq_len

        for conv in self.conv:
            x = self.dropout(F.relu(conv(x)))

        # x.shape -> batch_size x out_channels x seq_len
        x = x.permute(0, 2, 1) # x.shape -> batch_size x seq_len x out_channels 

        # self.rnn.flatten_parameters()
        outputs, hidden = self.rnn(x)

        hidden = torch.tanh(self.fc(torch.cat([hidden[-2,:], hidden[-1,:]], dim=1)))

        # outputs.shape -> batch_size x seq_len x encoder_hidden_dim * 2
        # hidden.shape -> batch_size x decoder_hidden_dim
        return outputs, hidden

class Attention(nn.Module):
    def __init__(self, encoder_hidden_dim, decoder_dim):
        super(Attention, self).__init__()
        self.attention = nn.Linear(2*encoder_hidden_dim + decoder_dim, decoder_dim)
        self.v = nn.Linear(decoder_dim, 1, bias=False)

    def forward(self, encoder_outputs, decoder_hidden):
        # encoder_outputs.shape -> batch_size x seq_len x 2*encoder_hidden_dim
        # decoder_hidden.shape -> batch_size x decoder_dim
        src_len = encoder_outputs.shape[1]
        decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)
        # decoder_hidden.shape -> batch_size x seq_len x decoder_dim

        energies = torch.tanh(self.attention(torch.cat([encoder_outputs, decoder_hidden], dim=2)))
        # energies.shape -> batch_size x seq_len x decoder_dim

        energies = self.v(energies).squeeze(2) # shape -> batch_size x seq_len
        attention_coefficients = torch.softmax(energies, dim=1)

        # attention_coefficients.shape -> batch_size x seq_len
        return attention_coefficients
    
class Decoder(nn.Module):
    def __init__(self, attention, encoder_hidden_dim, image_dim, decoder_hidden_dim, dropout=0.5):
        super(Decoder, self).__init__()
        self.attention = attention
        self.rnn = nn.GRU(2*encoder_hidden_dim + image_dim*image_dim, decoder_hidden_dim, num_layers=1, batch_first=True)
        self.fsout = nn.Linear(2*encoder_hidden_dim + image_dim*image_dim + decoder_hidden_dim, image_dim*image_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_outputs, hidden, input):
        # encoder_outputs.shape -> batch_size x seq_len x 2*encoder_hidden_dim 
        # hidden.shape -> batch_size x decoder_hidden_dim
        # input.shape -> batch_size x image_dim x image_dim

        attention_weights = self.attention(encoder_outputs, hidden)
        # attention_weights.shape -> batch_size x seq_len 

        attention_context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs) # batch_size x 1 x 2*encoder_hidden_dim
        input = input.view(input.shape[0], -1).unsqueeze(1) # batch_size x 1 x image_dim*image_dim
        hidden = hidden.unsqueeze(0) # 1 x batch_size x decoder_hidden 

        # compute the output and hidden 
        output, hidden = self.rnn(torch.cat([attention_context, input], dim=1), hidden)
        # output.shape -> batch_size x 1 x decoder_dim 
        # hidden.shape -> 1 x batch_size x decoder_dim

        input = input.squeeze(1) # batch_size x image_dim*image_dim 
        attention_context = attention_context.squeeze(1) # batch_size x 2*encoder_hidden_dim
        hidden = hidden.squeeze(0) # batch_size x decoder_hidden_dim

        # predictions.shape -> batch_size x image_dim*image_dim
        predictions = torch.sigmoid(self.fsout(torch.cat([input, attention_context, hidden], dim=1)))

        return predictions, hidden


class Decoder1(nn.Module):
    def __init__(self, attention, encoder_hidden_dim, image_dim, decoder_hidden_dim, dropout=0.5):
        super(Decoder, self).__init__()
        self.attention = attention
        self.rnn = nn.GRU(2*encoder_hidden_dim + image_dim*image_dim, decoder_hidden_dim, 1, batch_first=True)
        # self.fcout = nn.Linear(image_dim*image_dim + decoder_hidden_dim + 2*encoder_hidden_dim, image_dim*image_dim)
        self.fcout = nn.Linear(image_dim*image_dim + 2*encoder_hidden_dim + decoder_hidden_dim, image_dim*image_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_outputs, decoder_hidden, input):
        # encoder_outputs.shape -> batch_size x seq_len x 2*encoder_dim
        # decoder_hidden.shape -> batch_size x decoder_hidden
        attention_weights = self.attention(encoder_outputs, decoder_hidden).unsqueeze(1) # batch_size x 1 x seq_len
        # attention_weights.shape -> batch_size x 1 x seq_len
        # attention_context = torch.bmm(attention_weights, encoder_outputs).unsqueeze(1)
        attention_context = torch.bmm(attention_weights, encoder_outputs) # shape -> batch_size x 1 x 2*encoder_dim

        # attention_context.shape -> batch_size x 2*encoder_dim -> batch_size x 1 x 2*encoder_dim
        # input.shape -> batch_size x 256*256
        input = input.view(input.shape[0], -1).unsqueeze(1) # shape -> batch_size x 1 x image_dim*image_dim

        # decoder_hidden.unsqueeze(1).shape -> batch_size x 1 x decoder_hidden
        output, hidden = self.rnn(torch.cat([input, attention_context], dim=1), decoder_hidden.unsqueeze(0))

        # output.shape -> batch_size x 1 x decoder_dim
        # hidden.shape -> num_layers x batch_size x decoder_dim

        # the pixel values are between 0 and 1

        attention_context = attention_context.squeeze(1) # attention_context.shape -> batch_size x 2*encoder_hidden_dim
        input = input.squeeze(1) # input.shape -> batch_size x image_dim*image_dim
        hidden = hidden.squeeze(0) # hidden.shape -> batch_size x decoder_hidden_dim

        predictions = torch.softmax(self.fcout(torch.cat([input, attention_context, hidden], dim=1)))
        # predictions.shape -> batch_size x image_dim*image_dim

        return predictions, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, attention, decoder, im_dim):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.attention = attention
        self.decoder = decoder
        self.image_dim = im_dim

    def forward(self, src, trg):
        # generate hidden predictions using the encoder
        # src.shape -> batch_size 
        # trg.shape -> batch_size x trg_len x im_dim x im_dim
        encoder_outputs, hidden = self.encoder(src)

        # encoder_outputs.shape -> batch_size x seq_len x 2*encoder_hidden_dim
        # hidden.shape -> batch_size x decoder_dim

        batch_size = src.shape[0]
        trg_len = trg.shape[1]

        output_predictions = torch.zeros(batch_size, trg_len, self.image_dim*self.image_dim)
        input = trg[:,0]

        for j in range(1, trg_len):
            predictions, hidden = self.decoder(encoder_outputs, hidden, input)
            # predictions.shape -> batch_size x image_dim*image_dim
            # hidden.shape -> batch_size x decoder_dim
            output_predictions[:,j] = predictions
            # assuming that the teacher forcing ratio is set to 1
            input = trg[:,j]

        return output_predictions