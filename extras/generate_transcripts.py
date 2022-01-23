# Generate transcripts using Silero STT 
import torch
import zipfile
import torchaudio
from glob import glob
import threading
from tqdm import tqdm
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor

# Generate transcripts from audio files
def generate_transcript_batch(thread_id, folder):
    # folder will be of type 'MEAD/M007'
    transcripts_file = 'transcripts.txt'
    audio_transcripts = list()
    batch_size = 20

    print(f'Inside folder {folder} for thread {thread_id}')
    wav_files = glob(folder + '/audio/*/*/*.wav')

    model, decoder, utils = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                       model='silero_stt',
                                       language='en',
                                       device=device)

    (read_batch, split_into_batches, read_audio, prepare_model_input) = utils
    batches = split_into_batches(wav_files, batch_size=batch_size)
    
    for i in tqdm(range(len(batches))):
        current_audio_batch = wav_files[i*batch_size:(i+1)*batch_size]
        input = prepare_model_input(read_batch(batches[i]), device=device)
        output = model(input)
        for j in range(len(output)):
            transcription = decoder(output[j].cpu())
            # save the audio_file_path and transcription in file
            audio_transcripts.append(current_audio_batch[j] + '|' + transcription)

    # write audio_file_path and transcripts as audio_file_path|transcripts
    write_to_file(os.path.join(folder, transcripts_file), audio_transcripts)

# Write content to file
def write_to_file(filepath, content):
    print(f'Writing to file : {filepath}')
    with open(filepath, 'w') as f:
        for line in content:
            f.write(line + '\n')

if __name__ == '__main__':
    # folder of format -> MEAD/M007/audio/angry/level_1/*.wav
    folder_list = glob('MEAD/*')
    device = 'cuda'
    thread_limit = 10

    # Threadpool ensures that the maximum number of threads is limited to thread_limit
    with ThreadPoolExecutor(thread_limit) as e:
		results = e.map(sleep_job, ((thread_id, folder) for thread_id, folder in enumerate(folder_list)))