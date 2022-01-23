# defines the training code for training the text2face generation network 
from dataloader import TextLandmarksDataset, TextImageCollate
import torch
import torch.nn as nn
from model import Encoder, Decoder, Seq2Seq
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import datetime
import os
import random

# choosing the device for running the compute on 
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# filename = 'vlog_train_text_landmarks.txt' # filename containing the text to face landmark images mapping 
train_filename = 'train_vlog_text_landmarks.txt' # train file having text|video landmark mapping
val_filename = 'val_vlog_text_landmarks.txt' # val file having text|video landmark mapping

# get the datasets ready
trainset = TextLandmarksDataset(train_filename)
valset = TextLandmarksDataset(val_filename)
collate_fn = TextImageCollate()

# get the dataloaders ready
batch_size = 8
# use the collate function
train_loader = DataLoader(trainset, shuffle=True, num_workers=0, batch_size=batch_size, collate_fn=collate_fn)
# train_loader = DataLoader(trainset, shuffle=True, num_workers=1, batch_size=batch_size)
val_loader = DataLoader(valset, num_workers=0, batch_size=batch_size, collate_fn=collate_fn)

# define the loss function 
# criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss(reduction='sum') # used to compute the squared pixel-distance loss

# define the model parameters and model 
embedding_dim = 512
input_dim = 512
image_dim = 256
out_channels = 512
kernel_size = 5
decoder_hidden_dim = 256
num_layers = 1

val_after_steps = 1000

encoder = Encoder(embedding_dim, input_dim, out_channels, kernel_size)
decoder = Decoder(image_dim, decoder_hidden_dim, num_layers)
model = Seq2Seq(encoder, decoder, image_dim).to(device)

# define the optimizer 
optimizer = optim.Adam(model.parameters(), lr=1e-3)

epochs = 10
steps = 0
video_sequence_threshold = 250

# this is the training procedure 
for epoch in range(epochs):
    # define the training procedure
    # for i, vals in enumerate(train_loader):
        # print(vals.shape)
    for i, batch in enumerate(train_loader):
        model.train()
        steps += 1
        # text, video = text.to(device), video.to(device)
        text, text_lengths, video, video_lengths = batch
        video = video/255.0
        text, video = text.to(device), video.to(device)
        print(f'text : {text.shape}, video : {video.shape}', flush=True)
        optimizer.zero_grad()
        # print(f'Before predictions')

        # Ignore current batch if video sequence length greater than threshold
        if video.shape[1] > video_sequence_threshold:
            print(f'Skipping current batch because of sequence length : {video.shape[1]}', flush=True)
            continue

        else:
            output_predictions = model(text, text_lengths, video, video_lengths) # shape -> batch_size x seq_len x image_dim*image_dim
            # video.shape -> batch_size x seq_len1 x image_dim x image_dim
            print(f'predictions : {output_predictions.shape}', flush=True)

            # gt_video = video.view(video.shape[0], 50, image_dim*image_dim)
            video = video.view(video.shape[0], -1, image_dim*image_dim)
            # print(f'GT video : {video.shape}')

            # loss = criterion(output_predictions, video[:,:50].view(video.shape[0], 50, image_dim*image_dim)) # computes the squared pixel-distance loss between the images frames for all videos 
            loss = criterion(output_predictions.float().to(device), video.float())
            print(f'Epoch : {epoch}, step : {steps}/{len(train_loader)}, Loss : {loss.item()}', flush=True)
            loss.backward() # backpropagate the loss and compute the gradients 

            # print(f'Loss : {loss.item()}')

            # update the parameters using the computed gradients
            optimizer.step()

            # print('Completed the step')

        # if steps%10 == 1:
        #     model.eval()
        #     print(f'Saving image prediction to disk')
        #     # generate the predictions on the evaluation dataset
        #     for i, batch in enumerate(val_loader):
        #         text, text_lengths, video, video_lengths = batch
        #         text, video = text.to(device), video.to(device)
        #         predictions = model(text, text_lengths, video, video_lengths)
        #         # predictions = predictions.view(predictions.shape[0], 50, image_dim*image_dim)
        #         print(f'Val : {predictions.shape}')

        #         # write the predictions to the disk 
        #         DIR_PATH = '/home2/aditya1/text2face/temp_audio_files/sample_outputs'
        #         os.makedirs(DIR_PATH, exist_ok=True)
        #         to_plot = predictions[0]
        #         # plot all the images in that frame 
        #         print(f'To plot : {to_plot.shape}')

        #         ch = 'abcdefghijklmnopqrstuvwxyz0123456789'
        #         filename = list()
        #         for i in range(5):
        #             filename.append(ch[random.randint(0, len(ch)-1)])

        #         dirname = os.path.join(DIR_PATH, ''.join(filename))
        #         os.makedirs(dirname, exist_ok=True)
        #         print(f'DIRPATH : {dirname}')
        #         for i in range(len(to_plot)):
        #             current_image = to_plot[i]
        #             current_image = current_image.view(image_dim, image_dim)
        #             filename = os.path.join(dirname, str(i+1) + '.jpg')
        #             save_image(current_image, filename)

        # Generate predictions only for the first batch
        if steps%val_after_steps == 0:
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

                dirname = os.path.join(DIR_PATH, filename)
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


# used for evaluating the network
# def eval():