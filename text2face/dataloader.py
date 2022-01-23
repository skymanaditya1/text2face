# The text file contains a mapping between text transcript and the path of image landmarks 
from torch.utils.data import Dataset
from utils import load_text_landmarkpath
from text import text_to_sequence
import numpy as np
import torch

class TextLandmarksDataset(Dataset):
    def __init__(self, filename):
        # read the text file inside the init function 
        self.text_landmarkpaths = load_text_landmarkpath(filename)

    def get_text_landmark_pair(self, text_landmarkpath):
        # return the text and image landmark pair from the text file 
        text, landmark_path = text_landmarkpath
        
        # normalized text 
        norm_text = self.get_text(text)

        # load the image from the image landmark path 
        landmark_image = self.get_landmark_image(landmark_path)

        return (norm_text, landmark_image)

    # get representation of text as sequence of numbers 
    def get_text(self, text, text_cleaners=['english_cleaners']):
        torch_text = torch.IntTensor(text_to_sequence(text, text_cleaners))
        return torch_text 

        # MAX_LEN = 50
        # # print(f'text : {torch_text.shape}')
        # # print(text)
        # # text_zeros = torch.zeros(torch_text.shape)
        # # batch_size = torch_text.shape[0]
        # text_zeros = torch.zeros(MAX_LEN).long()
        # # text_zeros = torch.zeros(batch_size, MAX_LEN, torch_text.shape[2])
        # # find the min of the two 
        # min_val = min(MAX_LEN, torch_text.shape[0])
        # text_zeros[:min_val] = torch_text[:min_val]
        # return text_zeros
        # return torch_text

    def get_landmark_image(self, landmark_path):
        # image of dimension -> num_frames x height x width, zero padding to the empty frames?
        video = np.load(landmark_path, allow_pickle=True)['data'] # frames x height x width
        # TODO: apply some preprocessing 
        video = torch.from_numpy(video)
     
        # # print(f'Video : {video.shape}')
        # MAX_LEN = 50
        # batch_size = video.shape[0]
        # image_dim = video.shape[2]
        # video_zeros = torch.zeros(MAX_LEN, image_dim, image_dim)
        # # video_zeros = torch.zeros(video.shape)
        # min_val = min(MAX_LEN, video.shape[0])
        # video_zeros[:min_val] = video[:min_val]
        # # video_zeros[:MAX_LEN] = video[:MAX_LEN]
        # return video_zeros
        return video

    def __len__(self):
        return len(self.text_landmarkpaths)

    def __getitem__(self, index):
        return self.get_text_landmark_pair(self.text_landmarkpaths[index])

class TextImageCollate():
    ''' Zero pads the input text sequence and the landmarks video'''

    def __init__(self):
        pass

    # batch is the input which contains the sequence of text tokens and video landmarks
    def __call__(self, batch):
        # Right zero-pad all one-hot text sequences to max input length
        # Find the length of the non-padded text sequences
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        # Add padding using the max_input_len to all non-padded text sequences 
        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        # Each video has a varying number of frames, right pad with zeros 
        # video (x[1]) - frames x height x width (number of frames could be varying)
        # Find the video with the max number of frames to append with 0s
        max_frames = max([current[1].shape[0] for current in batch])
        # print(f'Max frames : {max_frames}')
        # print(f'Batch shape : {type(batch)}')
        # print(f'Text shape : {batch[0].shape}, video shape : {batch[1].shape}')
        video_frames = torch.LongTensor(len(batch), max_frames, 256, 256)
        video_frames.zero_()

        out_lengths = torch.LongTensor(len(batch))
        # follow the same convention as above 
        for i in range(len(ids_sorted_decreasing)):
            video = batch[ids_sorted_decreasing[i]][1]
            # print(f'Video shape : {video.shape}')
            # video.shape -> frames x height x width 
            video_frames[i, :video.shape[0]] = video
            out_lengths[i] = video.shape[0] # gives the number of frames before right zero pad

        # return the length of the text and video sequences 
        # print(f'text length : {text_padded.shape}, video sequence length : {video_frames.shape}')

        return text_padded, input_lengths, video_frames, out_lengths
        