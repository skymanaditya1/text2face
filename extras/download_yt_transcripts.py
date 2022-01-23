# Download YouTube transcripts for VLOG videos 

from youtube_transcript_api import YouTubeTranscriptApi
import json 
import os
import datetime
from glob import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import argparse
  
# assigning srt variable with the list
# of dictonaries obtained by the get_transcript() function
# srt = YouTubeTranscriptApi.get_transcript("2p3NIR8LYoo")
  
# prints the result
# print(srt)

# Buffer is specified in seconds
def crop_single_video(video_txt, buffer=0.3):
    # gets the video.txt file as input
    folder_name = '/'.join(video_txt.split('/')[:-1])
    print(folder_name)
    with open(video_txt, 'r') as f:
        for video_file in f:
            video_file = video_file.strip('\n') # gets the video_id
            transcript_file = os.path.join(folder_name, video_file + '_transcript.txt')
            srt = YouTubeTranscriptApi.get_transcript(video_file)

            transcript_lines = list()

            # write the transcripts to a file 
            with open(transcript_file, 'w') as f:
                for line in srt:
                    f.write('{}\n'.format(line))
                    transcript_lines.append('{}\n'.format(line))
            
            # get the start times from the srt 
            start_times = list()
            for sentence in transcript_lines:
                sentence = sentence.strip('\n')
                start = float(sentence.split('\'start\': ')[1].split(',')[0])
                duration = float(sentence.split('\'duration\': ')[1].split('}')[0])

                start_times.append(start)

            # Add the duration corresponding to the last sentence in the video 
            start_times.append(start + duration)

            # create a folder inside the directory 
            video_dir = os.path.join(folder_name, video_file)
            os.makedirs(video_dir, exist_ok=True)

            # use the duration from the list to split the video into sentences 
            for index in range(0, len(start_times)-1):
                start_time = datetime.timedelta(seconds=start_times[index] - buffer)
                end_time = datetime.timedelta(seconds=start_times[index+1] + buffer)

                # video_path gives the path of the video to be split
                video_path = os.path.join(folder_name, video_file + '.mp4')
                save_path = os.path.join(video_dir, str(index+1).zfill(4) + '.mp4')
                transcript_save_path = os.path.join(video_dir, str(index+1).zfill(4) + '.txt')

                command = 'ffmpeg -i {} -ss {} -to {} -strict -2 -c:v libx264 {}'.format(video_path, start_time, end_time, save_path)
                # command = 'ffmpeg -loglevel quiet -i {} -ss {} -to {} -strict -2 -c:v libx264 {}'.format(video_path, start_time, end_time, save_path)
                os.system(command)

                # write the transcript corresponding to the (index).mp4 file
                with open(transcript_save_path, 'w') as f:
                    f.write(transcript_lines[index])

# Used as part of the multithreading code
def crop_video(thread_video_txt):
    thread_id, video_txt = thread_video_txt

    print(f'Processing with thread : {thread_id}')
    folder_name = '/'.join(video_txt.split('/')[:-1])
    print(folder_name)
    # folder_name = video_txt.split('/')[1].split('/')[0]
    with open(video_txt, 'r') as f:
        for video_file in f:
            video_file = video_file.strip('\n')
            transcript_file = os.path.join(folder_name, video_file + '_transcript.txt')
            srt = YouTubeTranscriptApi.get_transcript(video_file)

            transcript_lines = list()

            # Write the transcript to a file 
            with open(transcript_file, 'w') as f:
                for line in srt:
                    f.write('{}\n'.format(line))
                    transcript_lines.append('{}\n'.format(line))

            start_times = list()
            with open(transcript_file, 'r') as f:
                for sentence in f:
                    sentence = sentence.strip('\n')
                    start = float(sentence.split('\'start\': ')[1].split(',')[0])
                    duration = float(sentence.split('\'duration\': ')[1].split('}')[0])

                    start_times.append(start)
            
            start_times.append(start_times[-1] + duration) # duration here indicates the duration of the last file

            # create a folder inside the directory
            video_dir = os.path.join(folder_name, video_file)
            os.makedirs(video_dir, exist_ok=True)

            # use the duration from the file to split the video into sequences 
            for i in range(0, len(start_times)-1):
                if i != 0:
                    start_time = datetime.timedelta(seconds=start_times[i] - 0.3)
                else:
                    start_time = datetime.timedelta(seconds=start_times[i])

                end_time = datetime.timedelta(seconds=start_times[i+1] + 0.3)
            
                video_path = os.path.join(folder_name, video_file + '.mp4')
                # print(video_path)
                save_path = os.path.join(video_dir, str(i+1).zfill(4) + '.mp4')
                transcript_save_path = os.path.join(video_dir, str(i+1).zfill(4) + '.txt')
                # print(save_path)
                command = 'ffmpeg -loglevel quiet -i {} -ss {} -to {} -strict -2 -c:v libx264 {}'.format(video_path, start_time, end_time, save_path)
                os.system(command)

                # write the transcript into a file
                with open(transcript_save_path, 'w') as f:
                    for line in transcript_lines[i]:
                        f.write(line)

# This is the proper working code for generating video crops
def crop_video_different(video_job):
    buffer = 0.3
    video_file, thread_id = video_job

    folder_name = '/'.join(video_file.split('/')[:-1])
    print(f'Processing folder {folder_name} with thread : {thread_id}', flush=True)

    video_raw_name = os.path.basename(video_file).split('.')[0]
    
    transcript_file = os.path.join(folder_name, video_raw_name + '_transcript.txt')
    try:
        srt = YouTubeTranscriptApi.get_transcript(video_raw_name, languages=['en'])
    except Exception as e:
        print(e, flush=True)
        print(f'Skipping video file : {video_file}', flush=True)
        return

    transcript_lines = list()

    with open(transcript_file, 'w') as f:
        for line in srt:
            f.write('{}\n'.format(line))
            transcript_lines.append('{}\n'.format(line))

    start_times = list()
    with open(transcript_file, 'r') as f:
        for sentence in f:
            sentence = sentence.strip('\n')
            start = float(sentence.split('\'start\': ')[1].split(',')[0])
            duration = float(sentence.split('\'duration\': ')[1].split('}')[0])

            start_times.append(start)

    start_times.append(start + duration) # this indicates appending the last item

    # create folder inside directory for storing video splits
    video_dir = os.path.join(folder_name, video_raw_name)
    os.makedirs(video_dir, exist_ok=True)

    # use the duration from the file to split the video into sequences
    for i in range(0, len(start_times)-1):
        # if i == 0:
        #     start_time = datetime.timedelta(seconds=start_times[i])
        # else:
        #     start_time = datetime.timedelta(seconds=start_times[i])

        # no need to add buffer in the beginning of the video
        start_time = datetime.timedelta(seconds=start_times[i])
        end_time = datetime.timedelta(seconds=start_times[i+1] + buffer)

        save_path = os.path.join(video_dir, str(i+1).zfill(4) + '.mp4')
        transcript_save_path = os.path.join(video_dir, str(i+1).zfill(4) + '.txt')

        # command to split the file 
        command = 'ffmpeg -loglevel quiet -i {} -ss {} -to {} -strict -2 -c:v libx264 {}'.format(video_file, start_time, end_time, save_path)
        os.system(command)

        # write the transcript into a file
        with open(transcript_save_path, 'w') as f:
            f.write(transcript_lines[i])


if __name__ == '__main__':
    # Assumption: Directory structure is as follows: videos/Speaker/videos.txt -> videos/Speaker/*.mp4
    # Generates cuts with folder name (MP4 Folder)/i.mp4

    max_threads = 10
    # video_txt_files = glob('videos/*/videos.txt')
    # speaker_video_path = '/ssd_scratch/cvit/aditya1/SpeakerData/videos/MKBHD/videos.txt'
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--speakers", help="Speaker list")
    args = parser.parse_args()

    # use the speaker list

    # generate files for certain speakers 
    files = list()
    # speakers = ['KritikaGoel', 'Superwoman', 'MKBHD']
    # speakers = args.speakers.split(',')
    speakers = args.speakers
    print(f'Speakers found : {speakers}', flush=True)
    speakers = speakers.split(',')
    FILEPATH = '/ssd_scratch/cvit/aditya1/videos/{}/*.mp4'
    for speaker in speakers:
        speaker_files = glob(FILEPATH.format(speaker))
        files.extend(speaker_files)

    print(f'Number of speakers to use : {len(files)}', flush=True)
    print(files, flush=True)

    # process for all the speakers
    p = ThreadPoolExecutor(max_threads)
    jobs = [(video_file, job_id) for job_id, video_file in enumerate(files)]
    futures = [p.submit(crop_video_different, job) for job in jobs]
    _ = [r.result() for r in tqdm(as_completed(futures), total=len(futures))]
    
    # with ThreadPoolExecutor(max_threads) as e:
    #     results = e.map(crop_video, ((thread_id, video_path) for thread_id, video_path in enumerate(video_txt_files)))

    # Run the processing for a single speaker directory
    

    # p = ThreadPoolExecutor(max_threads)
    # speaker_video_path = '/ssd_scratch/cvit/aditya1/JohnnyHarris/videos.txt'
    # crop_single_video(speaker_video_path)