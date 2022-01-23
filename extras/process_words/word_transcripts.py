# Download YouTube transcripts in the VTT format
# For the transcripts downloaded, get word level timestamps 
# Using sentences, categorize word level boundaries within sentences 
from glob import glob 
import os
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

command = 'youtube-dl --sub-lang en --write-auto-sub --sub-format vtt --skip-download https://www.youtube.com/watch?v={} -o {}'

def generate_transcripts(job_video_file):
    # read video_file and get individual video IDs
    video_file, job_id = job_video_file
    folder_name = '/'.join(video_file.split('/')[:-1])
    with open(video_file, 'r') as f:
        for video in f:
            video = video.strip('\n')
            subtitle_filepath = os.path.join(folder_name, video)
            # new_command = command.format(video, subtitle_filepath)
            # time.sleep(5)
            os.system(command.format(video, subtitle_filepath))
            # print(new_command)

# Iterate through all videos.txt files and generate transcripts for all the YouTube files 
if __name__ == '__main__':
    max_threads = 5
    p = ThreadPoolExecutor(max_threads)
    video_files = glob('SpeakerData/videos/*/videos.txt')
    jobs = [(video, job_id) for job_id, video in enumerate(video_files)]
    futures = [p.submit(generate_transcripts, job) for job in jobs]
    _ = [r.result() for r in tqdm(as_completed(futures), total=len(futures))]
    # for video_file in video_files:
    #     generate_transcripts(video_file)