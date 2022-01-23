from glob import glob
from pydub import AudioSegment 
import os
import threading
from tqdm import tqdm
import random
import time
from concurrent.futures import ThreadPoolExecutor
from moviepy.editor import *
import numpy as np
import unicodedata
import re

# Accepts folder path as argument and converts all files inside the folder
# Full folder path -> MEAD/M007/audio/angry/level_1/*.wav
# Function for converting m4a file format to wav file format in MEAD

# To make the sound files compatible for Tacotron2, following changes are required 
# 1. Convert the m4a file format to wav 
# 2. Change the sampling rate from 48000 to 22050 
# 3. Convert dual channel input audio to mono input audio - taking the left mono_input for now
# 4. ThreadPoolExecutor helps manage the total number of threads and limits the total number of threads to thread_limit

def convert_files(thread_id_folder):
	thread_id = thread_id_folder[0]
	folder = thread_id_folder[1]
	sampling_rate = 22050
	print(f'Inside the folder : {folder}, for thread : {thread_id}')
	audio_files = glob(folder + '/audio/*/*/*.m4a')

	for audio in tqdm(audio_files):
		try:
			track = AudioSegment.from_file(audio, 'm4a')
			track = track.set_frame_rate(sampling_rate)
			# Split the stereo track into mono audio
			if track.channels == 2:
				mono_audios = track.split_to_mono()
				mono_left = mono_audios[0]
				mono_right = mono_audios[1]

				wav_filename = audio.replace('m4a', 'wav')
				# Export the single channel mono_left for now 
				mono_left.export(wav_filename, format='wav')
		except Exception as ex:
			print(f'Error converting : {audio}, exception : {type(ex).__nam__}, {ex.args}')
			pass

def generate_audio_from_video(id_video_file, format='wav', sampling_rate=22050):
	video_path, j = id_video_file
	print(f'{id}, processing : {video_path}')
	# read the video file 
	dirname = '/'.join(video_path.split('/')[:-1])
	audio_file_name = os.path.basename(video_path).replace('mp4', format)
	audio_path = os.path.join(dirname, audio_file_name)

	try:
		audio = AudioSegment.from_file(video_path)
		audio = audio.set_frame_rate(sampling_rate).set_channels(1)
		
		# write the audio file with sampling rate 22050 Hz
		# AudioSegment.from_file(audio_combined).export(audio_path, format=format)
		audio.export(audio_path, format=format)
		print(f'{audio_path} created successfully')
	except Exception as e:
		print(f'Exception : {e}')

def extract_transcript(id_text_file):
	text_file, j = id_text_file
	audio_file = text_file.replace('txt', 'wav')

	transcript_dir = 'transcript_files'
	os.makedirs(transcript_dir, exist_ok=True)

	failed_files = os.path.join(transcript_dir, 'failed_transcripts.txt')
	tts_mapping = os.path.join(transcript_dir, 'tts_vlog.txt')

	# check for the existence of the wav file
	if not os.path.isfile(audio_file):
		# print(f'Failed adding transcript for file : {text_file}')
		with open(failed_files, 'a') as f:
			f.write(text_file + '\n')
		return

	# iterate through the lines to get the transcript
	with open(text_file, 'r') as f:
		line = f.read().splitlines()

	# print(line)
	
	text = line[0].split("{'text': ")[1].split(", 'start'")[0].strip('\"\'')
	# normalize the string and replace the \n characters with ' ' character
	normalized_text = unicodedata.normalize("NFKD", text).replace('\n', ' ')

	if '[' in text or ']' in normalized_text:
		# print(f'Failed adding transcript for file : {text_file}')
		with open(failed_files, 'a') as f:
			f.write(text_file + '\t' + normalized_text + '\n')
		return

	# Finally create mapping between audio file and transcript
	with open(tts_mapping, 'a') as f:
		f.write(audio_file + '|' + normalized_text + '\n')


# remove audio files greater than audio_threshold duration, audio_threshold is specified in seconds
def filter_audio_duration(filename, audio_threshold=15):
	with open(filename, 'r') as f:
		files = f.read().splitlines()

	files_filtered = dict()
	added_files = list()
	for file in files:
		audio_file = file.split('|')[0]
		audio = AudioSegment.from_file(audio_file)
		if audio.duration_seconds > audio_threshold:
			# files_filtered += 1
			files_filtered[audio_file] = audio.duration_seconds
			continue 
		else:
			added_files.append(file)
	
	# write the file to location 
	print(f'Files ignored : {len(files_filtered)}, files to add : {len(added_files)}')
	new_filename = os.path.join('/'.join(filename.split('/')[:-1]), os.path.basename(filename).split('.')[0] + '_updated.txt')
	with open(new_filename, 'w') as f:
		for file in added_files:
			f.write(file + '\n')

	print(f'All files ignored')
	print(files_filtered)

# code used to normalize the text transcripts 
# removes all unnecessary characters, replaces multiple spaces with a single space, trims leading and training space characters
def normalize_text(filename):
	with open(filename, 'r') as f:
		lines = f.read().splitlines()

	new_lines = list()
	for line in lines:
		line = line.replace('\\xa0', ' ').replace('\\n', ' ').replace('\\\'', '\'')
		# more than one spaces are replaced with a single space, leading and trailing spaces are stripped
		line = re.sub(' +', ' ', line).strip()
		new_lines.append(line)

	# write the new_lines to the modified tts file 
	new_tts_path = os.path.join('/'.join(filename.split('/')[:-1]), os.path.basename(filename).split('.')[0] + '_normalized.txt')
	with open(new_tts_path, 'w') as f:
		for line in new_lines:
			f.write(line + '\n')
		
	print(f'Lines written successfully: {len(new_lines)}')

if __name__ == '__main__':
	thread_limit = 5
	video_files = glob('SpeakerData/videos/*/*/*.mp4')
	transcript_files = glob('SpeakerData/videos/*/*/????.txt')

	# jobs = [(vfile, i%thread_limit) for i, vfile in enumerate(video_files)]
	# p = ThreadPoolExecutor(thread_limit)
	# futures = [p.submit(generate_audio_from_video, j) for j in jobs]
	# _ = [r.result() for r in tqdm(as_completed(futures), total=len(futures))]

	# for transcript_file in tqdm(transcript_files):
	# 	extract_transcript((transcript_file, 0))

	# filter audio files based on audio duration 
	# filename = 'filelists/vlog_train.txt'
	# filter_audio_duration(filename)

	# normalize text by processing text transcript 
	filename = 'filelists/vlog_val.txt'
	normalize_text(filename)

	# filename = 'SpeakerData/videos/AnfisaNava/1Kd3JiQBxXQ/0192.txt'
	# extract_transcript((filename, 0))

	# filename = 'SpeakerData/videos/AnfisaNava/1Kd3JiQBxXQ/0191.mp4'
	# generate_audio_from_video((filename, 0))

	# with ThreadPoolExecutor(thread_limit) as e:
	# 	results = e.map(sleep_job, ((thread_id, folder) for thread_id, folder in enumerate(folder_list)))