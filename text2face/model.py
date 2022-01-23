# Model is composed of 3 1D convolutional layers followed by bidirectional LSTM layer -> 
import torch
import torch.nn as nn
import torch.nn.functional as F
from text import symbols

class Encoder(nn.Module):
    def __init__(self, embedding_dim, in_channels, out_channels, kernel_size):
        super(Encoder, self).__init__()
        # encoder is composed of 1D convolutional layer and bidirectional LSTM
        self.embedding = nn.Embedding(len(symbols), embedding_dim)
        convs = 3
        convolutions = list()
        self.dropout = nn.Dropout(0.5)
        for j in range(convs):
            conv_layer = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=int((kernel_size-1)/2)),
                nn.BatchNorm1d(out_channels)
                )
            # conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=int((kernel_size-1)/2))

            convolutions.append(conv_layer)

        # self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=int((kernel_size-1)/2))
        self.conv = nn.ModuleList(convolutions)
        self.lstm = nn.LSTM(out_channels, out_channels // 2, 1, batch_first=True, bidirectional=False)
    
    def forward(self, x, src_len):
        # x.shape -> batch_size x seq_length x character_embedding 
        x = self.embedding(x).transpose(1, 2) # shape -> batch_size x character_embedding x seq_length
        for conv in self.conv:
            x = self.dropout(conv(F.relu(x)))

        # x.shape -> batch_size x out_channels x seq_length
        x = x.transpose(1, 2) # x.shape -> batch_size x seq_length x out_channels (batch_size x seq_length x 512)

        # since the sequences are variable length sequences, created packed sequences from padded sequences 
        embedded_packed = nn.utils.rnn.pack_padded_sequence(x, src_len.to('cpu'), batch_first=True) # src_len are being explicitly pushed to the CPU
        
        # outputs, (hidden, cell) = self.lstm(x) 
        # hidden is from the last non-padded hidden state in the sequence
        packed_outputs, (hidden, cell) = self.lstm(embedded_packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs) # this step is not required but still doing

        # outputs.shape -> batch_size x seq_length x hidden_dim * n_channels -> batch_size x seq_length x 512
        # return outputs
        return hidden, cell # since this is bidirectional -> batch_size*num_directions x seq_length x hidden_dim
        
# decoder generates the sequence of faces for the given encoded textual sequence
class Decoder(nn.Module):
    def __init__(self, image_dim, hidden_dim, num_layers, dropout=0.5):
        super(Decoder, self).__init__()
        # image would be of dimension 256x256 which would be flattened 
        self.rnn = nn.LSTM(image_dim*image_dim, hidden_dim, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fsout = nn.Linear(hidden_dim, image_dim*image_dim)
    
    def forward(self, y, hidden, cell):
        # y would be of dimension -> batch_size x image_dim x image_dim 
        batch_size = y.shape[0]
        y = y.view(batch_size, -1).unsqueeze(1) # dimension -> batch_size x 1 x image_dim*image_dim
        # print(f'Shape of y : {y.shape}')
        # hidden, cell = hidden.long(), cell.long()
        # y = y.long()
        y = y.float()
        output, (hidden, cell) = self.rnn(y, (hidden, cell)) # output.shape -> batch_size x 1 x hidden_dim
        output = self.fsout(output.squeeze(1)) # output.shape -> batch_size x image_dim*image_dim
        # print(f'Output shape : {output.shape}')
        return output, hidden, cell

# Class used to generate a sequence of face landmarks for the given encoded text sequence
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, image_dim):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.image_dim = image_dim
    
    # src is the encoded textual sequence
    def forward(self, src, src_len, trg, trg_len, tf_ratio=1):
        # src.shape -> batch_size x seq_len x char embedding 
        # batch_size x seq_len x 512 
        hidden, cell = self.encoder(src, src_len) # hidden.shape -> num_layers x batch_size x hidden_dim
        # print(f'encoder hidden : {hidden.shape}') # 1 x 32 x 256
        # trg is a batch of videos (sequence of image frames) 
        # trg.shape -> batch_size x seq_len x height x width
        input = trg[:, 0].float() # batch_size x height x width -> 32 x 256 x 256
        # print(f'Input shape : {input.shape}')
        batch_size = src.shape[0]
        # MAX_LEN = 50
        MAX_LEN = trg.shape[1]

        output_predictions = torch.zeros(batch_size, MAX_LEN, self.image_dim*self.image_dim)

        # generate upto a maximum of MAX_LEN face landmarks 
        for j in range(1, MAX_LEN):
            # generate the facial landmarks for the current position 
            landmark_prediction, hidden, cell = self.decoder(input, hidden, cell)
            # landmark_prediction.shape -> batch_size x image_dim*image_dim
            output_predictions[:,j] = landmark_prediction

            input = trg[:, j] if tf_ratio == 1 else landmark_prediction.view(landmark_prediction.shape[0], 256, 256)

        return output_predictions # shape -> batch_size x MAX_LEN x image_dim*image_dim

def test_seq2seq():
    batch_size = 4
    in_channels = 512
    out_channels = 512
    kernel_size = 5
    image_dim = 256
    decoder_hidden_dim = 256
    num_layers = 1 
    encoder = Encoder(in_channels, out_channels, kernel_size)
    decoder = Decoder(image_dim, decoder_hidden_dim, num_layers)
    model = Seq2Seq(encoder, decoder, image_dim)

    # create the input
    seq_len = 50
    src = torch.randn(batch_size, in_channels, seq_len)
    trg = torch.randn(batch_size, seq_len, image_dim, image_dim)
    output_predictions = model(src, trg)   
    print(f'Output predictions : {output_predictions.shape}')

# test the Encoder 
def test_encoder():
    batch_size = 32
    in_channels = 512 
    out_channels = 512
    kernel_size = 5 
    seq_length = 12

    model = Encoder(in_channels, out_channels, kernel_size)
    input = torch.randn(batch_size, in_channels, seq_length) # in_channels indicates the dimension of the character embedding 
    output = model(input)

    print(f'Input : {input.shape}, output : {output.shape}')

if __name__ == '__main__':
    # test_encoder()
    test_seq2seq()