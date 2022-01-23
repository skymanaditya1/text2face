from dataloader import TextLandmarksDataset, TextImageCollate
import torch
import torch.nn as nn
from model_attention_masked import Encoder, Decoder, Seq2Seq, Attention
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import datetime
import os
import random

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

train_filename = 'train_vlog_text_landmarks.txt' # train file having text|video landmark mapping
val_filename = 'val_vlog_text_landmarks.txt' # val file having text|video landmark mapping

trainset = TextLandmarksDataset(train_filename)
valset = TextLandmarksDataset(val_filename)
collate_fn = TextImageCollate()

batch_size = 2
train_loader = DataLoader(trainset, shuffle=True, num_workers=0, batch_size=batch_size, collate_fn=collate_fn)
val_loader = DataLoader(valset, num_workers=0, batch_size=batch_size, collate_fn=collate_fn)

# criterion = nn.MSELoss(reduction='sum') # used to compute the squared pixel-distance loss
criterion = nn.MSELoss() # use the mean as reduction

# define the model parameters and model 
embedding_dim = 512 # this is equivalent to in_channels
encoder_hidden_dim = 256
out_channels = 512
kernel_size = 5 # defines the 5 gram language model
decoder_hidden_dim = 512
num_layers = 1
image_dim = 256

val_after_steps = 10

encoder = Encoder(embedding_dim, out_channels, kernel_size, encoder_hidden_dim, decoder_hidden_dim)
attention = Attention(encoder_hidden_dim, decoder_hidden_dim)
decoder = Decoder(attention, encoder_hidden_dim, image_dim, decoder_hidden_dim)
model = Seq2Seq(encoder, attention, decoder, image_dim).to(device)

# define the optimizer 
optimizer = optim.Adam(model.parameters(), lr=1e-4)

epochs = 100
steps = 0
video_sequence_threshold = 250

# this is the training procedure 
for epoch in range(epochs):
    for i, batch in enumerate(train_loader):
        model.train()
        steps += 1

        text, text_lengths, video, video_lengths = batch
        video = video/255.0
        text, video = text.to(device), video.to(device)
        print(f'text : {text.shape}, video : {video.shape}', flush=True)
        optimizer.zero_grad()

        # Ignore current batch if video sequence length greater than threshold
        if video.shape[1] > video_sequence_threshold:
            print(f'Skipping current batch because of sequence length : {video.shape[1]}', flush=True)
            continue

        else:
            output_predictions = model(text, text_lengths, video, video_lengths) # shape -> batch_size x seq_len x image_dim*image_dim
            print(f'predictions : {output_predictions.shape}', flush=True)

            # gt_video = video.view(video.shape[0], 50, image_dim*image_dim)
            video = video.view(video.shape[0], -1, image_dim*image_dim)
            
            loss = criterion(output_predictions.float().to(device), video.float())
            print(f'Epoch : {epoch}, step : {steps}/{len(train_loader)}, Loss : {loss.item()}', flush=True)
            loss.backward() # backpropagate the loss and compute the gradients 

            optimizer.step()

        # validation loop
        if steps%val_after_steps == val_after_steps:
            model.eval()
            with torch.no_grad():
                print(f'Saving image prediction to disk', flush=True)
                # generate the predictions on the evaluation dataset
                dataiter = iter(val_loader)
                batch = dataiter.next()

                # Generate the predictions for the current batch
                text, text_lengths, video, video_lengths = batch
                text, video = text.to(device), video.to(device)
                video = video/255.0
                predictions = model(text, text_lengths, video, video_lengths)
                # predictions = predictions.view(predictions.shape[0], 50, image_dim*image_dim)
                print(f'Val : {predictions.shape}', flush=True)

                # write the predictions to the disk 
                DIR_PATH = '/home2/aditya1/text2face/temp_audio_files/sample_outputs'
                os.makedirs(DIR_PATH, exist_ok=True)
                to_plot = predictions[0]
                # plot all the images in that frame 
                print(f'To plot : {to_plot.shape}', flush=True)

                ch = 'abcdefghijklmnopqrstuvwxyz0123456789'
                filename = list()
                for i in range(5):
                    filename.append(ch[random.randint(0, len(ch)-1)])

                filename = ''.join(filename)

                filename = str(steps).zfill(5) + '_' + filename

                dirname = os.path.join(DIR_PATH, filename) + '_attn'
                os.makedirs(dirname, exist_ok=True)
                print(f'DIRPATH : {dirname}', flush=True)
                for i in range(len(to_plot)):
                    current_image = to_plot[i]
                    current_image = current_image.view(image_dim, image_dim)
                    filename = os.path.join(dirname, str(i+1) + '.jpg')
                    save_image(current_image, filename)

                dirname = dirname + '_gt'
                os.makedirs(dirname, exist_ok=True)
                print(f'DIRPATH : {dirname}', flush=True)

                video = video[0].to('cpu')
                print(f'GT video shape : {video.shape}', flush=True)

                for i in range(len(video)):
                    current_image = video[i].float()
                    # current_image = current_image.view(image_dim, image_dim)
                    filename = os.path.join(dirname, str(i+1) + '.jpg')
                    save_image(current_image, filename)