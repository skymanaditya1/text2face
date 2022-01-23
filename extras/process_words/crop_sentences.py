import os
import sys
import threading

import cv2
import random 
import os
import sys
import threading
from imutils import face_utils
import numpy as np
from moviepy.editor import VideoFileClip
import datetime
from datetime import datetime as dt
from glob import glob 
from concurrent.futures import as_completed, ThreadPoolExecutor
from tqdm import tqdm

# speaker=sys.argv[1]
# speaker = 'SpeakerData/videos_copy/AnfisaNava'
HD = False

sentences=True

'''
The flow should be as follows:
Iterate through all the vtt files 
For each vtt file, generate the word timestamps (word, corresponding timestamp)
Sentence level transcripts are given in video_transcript.txt -> generate the start times of sentences using that file 
Identify words that belong to a sentence based on their timestamp
'''


# For each vtt file, generate the word timestamps 

# Use the sentence level transcripts to generate the start and end times of sentences 

# Identify the words that belong to sentences based on their timestamps 

# Write the words in <sentence_id>_words.txt file


# files = [x for x in sorted(os.listdir(speaker)) if 'vtt' in x]

def run(file_ift):
	file, ift = file_ift
	# print(f'splitting {file} #{ift+1}/{len(files)}')
	# fname = file.split('.')[0]
	transcript_file = os.path.join('/'.join(file.split('/')[:-1]), os.path.basename(file).split('.')[0] + '_transcript.txt')
	print(f'Processing file : {file} using thread : {ift}', flush=True)
	youtube_file_folder = '/'.join(file.split('/')[:-1])
	fname = os.path.basename(file).split('.')[0]
	# vtt = open(os.path.join(speaker, file)).read().splitlines()
	vtt = open(file).read().splitlines()
	# transcript_file = os.path.join(speaker, transcript_file)
	vtt = [x for x in vtt if (all(ch in x for ch in ('<c>','</c>')) or ('align:start position:0%' in x))]

	indices = []
	for i, x in enumerate(vtt):
		if all(ch in x for ch in ('<c>','</c>')):
			indices.append(i)

	vtt_mod = []
	for i in indices:
		vtt_mod.append(vtt[i-1])
		vtt_mod.append(vtt[i])

	timestamps, i = [], 0
	while i+1 < len(vtt_mod):
		ts = vtt_mod[i].split(' --> ')
		start = ts[0]
		end = ts[1].split(' ')[0]
		words = vtt_mod[i+1]
		i = i+2
		words = words.replace('</c>', '<c>').split('<c>')
		words = [x.strip() for x in words]
		words = list(filter(None, words))
		fw = words[0].split('<')
		pts = fw[1].replace('>', '')
		timestamps.append([start, fw[0], pts])
		words = words[1:]
		ii = 0
		while ii+1 < len(words):
			word = words[ii]
			ets = words[ii+1].replace('>', '').replace('<', '')
			timestamps.append([pts, word, ets])
			pts = ets
			ii = ii+2
		timestamps.append([pts, words[ii], end])

	# len of the timestamps indicates the number of words in the transcript 
	# print(len(timestamps))
	# print(timestamps[5])

	# open the sentence level transcript file and read all sentences 
	sentences = list()
	start_times = list()
	with open(transcript_file, 'r') as f:
		sentences = f.read().split('\n')

	sentences = [sentence.strip('\n') for sentence in sentences if sentence is not '']

	for sentence in sentences:
		start = float(sentence.split('\'start\': ')[1].split(',')[0])
		duration = float(sentence.split('\'duration\': ')[1].split('}')[0])

		start_times.append(start)

	start_times.append(start + duration)

	sentence_durations = list()
	for i in range(0, len(start_times)-1):
		start_time = datetime.timedelta(seconds=start_times[i] + 0.3)
		end_time = datetime.timedelta(seconds=start_times[i+1] + 0.3)
		sentence_durations.append([start_time, end_time])

	# print(f'Sentence duration lengths : {len(sentence_durations)}')
	# print(timestamps)
	# sentence_id indicates the sentence id 
	sentence_id = 0
	word_id = 0
	current_words = list()
	save_folder = os.path.join(youtube_file_folder, fname)
	# print(f'Save folder : {save_folder}')
	os.makedirs(save_folder, exist_ok=True)
	while word_id < len(timestamps):
		# print(f'Word time : {timestamps[word_id][-1]}')
		# print(f'Sentence time : {str(sentence_durations[sentence_id][1])}')
		try:
			word_time = dt.strptime(timestamps[word_id][-1], '%H:%M:%S.%f')
		except ValueError:
			word_time = dt.strptime(timestamps[word_id][-1], '%H:%M:%S')
		try:
			sentence_time = dt.strptime(str(sentence_durations[sentence_id][1]), '%H:%M:%S.%f')
		except ValueError:
			sentence_time = dt.strptime(str(sentence_durations[sentence_id][1]), '%H:%M:%S')
		if word_time <= sentence_time:
			# the word goes in the current sentence - (sentence_id + 1)
			current_words.append(timestamps[word_id])
			# print(f'Timestamps are : {timestamps[word_id][-1]}, {str(sentence_durations[sentence_id][1])}, {timestamps[word_id][1]}')
			word_id += 1
			# print(f'Appending to current words : ')
		else:
			# write the current_words to the file - (sentence_id + 1)
			word_file = os.path.join(save_folder, str(sentence_id + 1).zfill(4) + '_words.txt')
			with open(word_file, 'w') as f:
				for current_word in current_words:
					f.write('{};{};{}'.format(current_word[0], current_word[1], current_word[2]) + '\n')
			current_words = list()
			# we move to the next sentence
			sentence_id += 1
		# word_id += 1

	# check if there are more words to be written 
	if len(current_words) > 0:
		word_file = os.path.join(save_folder, str(sentence_id + 1).zfill(4) + '_words.txt')
		with open(word_file, 'w') as f:
			for current_word in current_words:
				f.write('{};{};{}'.format(current_word[0], current_word[1], current_word[2]) + '\n')
		
	print(len(sentences))
	# print(sentences[-1])

	if len(timestamps) == 0:
		return

	def get_sec(time_str):
	    h, m, s = time_str.split(':')
	    return int(h) * 3600 + int(m) * 60 + float(s)

	def get_word(x):
		return x.replace("'", "").lower().replace(' ', '')

	# prefix = '{}/{}'.format(speaker, fname)

	# vpath = '{}/{}.mp4'.format(speaker, fname)

	# os.makedirs(prefix, exist_ok=True)

	# buff = 0.3
	# updateat = 20
	# saveas = 'unset'
	# nidx = 0
	# iid  = 0
	# it = 0

	# timestamps = [x for x in timestamps if len(x) > 0]

	# print(vpath)
	# video = VideoFileClip(vpath)
	# print(video.duration)

	# def get_sentence(timestamps, idx):
	# 	sentence_len = random.randint(3, 9)

	# 	nidx = idx+sentence_len
	# 	words = timestamps[idx:nidx]

	# 	sentence = ' '.join([get_word(x[1]) for x in words])
	# 	s = max(0, get_sec(words[0][0])-buff)
	# 	e = get_sec(words[-1][2])+buff

	# 	return nidx, sentence, words, s, e

	# while nidx < len(timestamps):
	# 	it = it+1
	# 	nidx, sentence, words, s, e = get_sentence(timestamps, nidx)
	# 	print(s,e)

	# 	savefilename = '{}-{}.mp4'.format(sentence[:15].replace(' ', '_'), iid)
		
	# 	saveas = '{}/{}'.format(prefix, savefilename)

	# 	iid = iid+1
	# 	if it % updateat == 0:
	# 		print(f'done {it+1}, lastsaveas: {saveas}')

	# 	with open(saveas.replace('.mp4', '.txt'), 'w') as wr:
	# 		for w in words:
	# 			wr.write('{};{};{}\n'.format(w[0], w[2], w[1]))

	# 	# try:
	# 	# vformat = '' if HD else '-crf 5 -preset ultrafast'
	# 	# command = 'ffmpeg -i "{}" -ss "{}" -to "{}" -c:v libx264 {} "{}"'.format(vpath, s, e, vformat, saveas)
	# 	clip = video.subclip(s,e)
	# 	clip.write_videofile(saveas, codec="libx264", temp_audiofile='t.m4a', remove_temp=True, audio_codec='aac')

	# 	# os.system(command)
	# 	# except:
	# 	# 	pass

	# return iid

# total_sentences = 0

# for ift, file in enumerate(files):
# 	# thread = threading.Thread(target=run, args=(ift, file), kwargs={})
# 	# print(f'starting thread ... {ift+1}, {file}')
# 	# thread.start()
# 	total_sentences = total_sentences + run (ift, file)
# 	break

# print(f'Total {total_sentences} done!')

if __name__ == '__main__':
	# Iterate through the vtt files 
	max_threads = 5
	vtt_files = glob('SpeakerData/videos/*/*.en.vtt')
	# vtt_files = ['SpeakerData/videos/AnfisaNava/1Kd3JiQBxXQ.en.vtt']
	# vtt_file = '1Kd3JiQBxXQ.en.vtt'
	# transcript_file = '1Kd3JiQBxXQ_transcript.txt'
	p = ThreadPoolExecutor(max_threads)
	jobs = [(vtt_file, job_id) for job_id, vtt_file in enumerate(vtt_files)]
	futures = [p.submit(run, job) for job in jobs]
	_ = [r.result() for r in tqdm(as_completed(futures), total=len(futures))]
	# for id, vtt_file in enumerate(vtt_files):
	# 	transcript_file = os.path.join('/'.join(vtt_file.split('/')[:-1]), os.path.basename(vtt_file).split('.')[0] + '_transcript.txt')
	# 	run(id, vtt_file, transcript_file)
