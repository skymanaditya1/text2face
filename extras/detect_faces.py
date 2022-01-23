# Use sfd facial detector to filter videos where faces are not detected across frames 
from concurrent.futures import ThreadPoolExecutor, as_completed
import face_alignment
from tqdm import tqdm
import cv2
from glob import glob
import os
import numpy as np
import torch

ngpus = 4

# Creates ngpu instances of face_alignment
fa = [face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cuda:{}'.format(id)) for id in range(ngpus)]
# fa = [face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cpu') for _ in range(ngpus)]
video_files_processed = 0
total_video_files = 0
processed_files = 'face_detection_result'
os.makedirs(processed_files, exist_ok=True)

# Takes a video as input and generates facial predictions for each frame
def detect_faces(video_file_gpu_id):
    video_file, gpu_id = video_file_gpu_id
    
    print(f'Processing {video_file} with gpu : {gpu_id}')
    # read video using opencv 
    video_stream = cv2.VideoCapture(video_file)
    batch_size = 2

    frames = list()
    # success, frame = video_stream.read()thickness = 2
    # while success:
    #     frames.append(frame)
    #     success, frame = video_stream.read() # After the last frame is read, success will be false

    success = 1
    while success:
        success, frame = video_stream.read()
        frames.append(frame)

    video_stream.release()

    # batch the frames into sizes of batch_size
    print(f'Total number of frames read : {len(frames)}')
    batches = [frames[i:i+batch_size] for i in range(0, len(frames), batch_size)]
    
    # generate the predictions for the entire batch
    no_faces = 0
    total_frames = 0
    for current_batch in batches:
        print('Generating predictions')
        current_batch = torch.from_numpy(np.asarray(current_batch)).permute(0, 3, 1, 2).to('cuda:{}'.format(gpu_id)) # converted to batch of dimension -> batch_size x channels x height x width
        # current_batch = torch.from_numpy(np.asarray(current_batch)).permute(0, 3, 1, 2)
        print(f'Size of batch : {current_batch.shape}')
        predictions = fa[gpu_id].face_detector.detect_from_batch(current_batch)
        print('Generated predictions')
        for _, prediction in enumerate(predictions):
            total_frames += 1
            if prediction is None:
                no_faces += 1

    # print the total number of faces and frames
    print(f'Total frames : {total_frames}, no faces detected : {no_faces}')

def detect_faces_single_video(video_file_gpu_id):
    video_file, gpu_id = video_file_gpu_id

    threshold_limit = 1000
    failed_file = os.path.join(processed_files, 'failed_detect_faces.txt')
    valid_file = os.path.join(processed_files, 'valid_files.txt') # faces were detected
    invalid_file = os.path.join(processed_files, 'invalid_files.txt') # faces were not detected

    # define the face alignment and detection object 
    # fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cuda:0')

    # Face detection applied to a single video
    video_stream = cv2.VideoCapture(video_file)
    batch_size = 32

    total_frames = int(video_stream.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f'Total frames present in the {video_file} is {total_frames}', flush=True)

    frames = list()

    if total_frames > threshold_limit:
        print(f'Execution failed for video : {video_file}, continuing')
        with open(failed_file, 'a') as f:
            f.write(video_file + '\n')
        return

    # Handle exception and continue execution if the processing of current video fails
    try:
        # count = 0
        success, frame = video_stream.read()
        while success:
            # count += 1
            frames.append(frame)
            success, frame = video_stream.read()
            # if count%100 == 0:
            #     print(f'Frames covered : {count}')
    except Exception as e:
        print(e, flush=True)
        print(f'Execution failed for video : {video_file}, continuing!')
        with open(failed_file, 'a') as f:
            f.write(video_file + '\n')
        return

    # Release the video stream after processing all frames
    # video_stream.release()

    if total_frames == 0:
        with open(failed_file, 'a') as f:
            f.write(video_file + '\n')
        return

    print(f'Total number of frames read for video {video_file} : {len(frames)}', flush=True)
    batches = [frames[i:i+batch_size] for i in range(0, len(frames), batch_size)]
    # reducing the size of the batch until it can fit in the GPU 
    processed = False
    no_faces = total_frames
    while not processed:
        try:
            # check if the batch size becomes equal to 0
            if batch_size == 0:
                print(f'Execution failed for video : {video_file}, batch size has become 0, continuing', flush=True)
                with open(failed_file, 'a') as f:
                    f.write(video_file + '\n')
                return
            _, no_faces = process_batch(batches, fa[gpu_id], gpu_id)
            processed = True # indicates that the batch was processed successfully
        except Exception as e:
            print(e, flush=True)
            print(f'Encountered CUDA error, reducing batch size to : {int(batch_size/2)}', flush=True)
            batch_size = int(batch_size/2)
            batches = [frames[i:i+batch_size] for i in range(0, len(frames), batch_size)]
            continue
    
    # check if the total number of faces detected is less than 5% of the total number of frames in the video
    if no_faces/total_frames*100 >= 5:
        with open(invalid_file, 'a') as f:
            f.write(video_file + '\n')
    else:
        with open(valid_file, 'a') as f:
            f.write(video_file + '\n')

    # video_files_processed += 1
    print(f'Video : {video_file}, Total frames : {total_frames}, no faces detected : {no_faces}', flush=True)


# Code used for detecting faces in a batch
def process_batch(batches, fa, gpu_id):
    no_faces = 0
    total_frames = 0
    print(f'Started processing on data size {len(batches)}, on gpu : {gpu_id}', flush=True)
    for current_batch in batches:
        # print('Generating predictions')
        current_batch = torch.from_numpy(np.asarray(current_batch)).permute(0, 3, 1, 2).to('cuda:{}'.format(gpu_id)) # converted to batch of dimension -> batch_size x channels x height x width
        # current_batch = torch.from_numpy(np.asarray(current_batch)).permute(0, 3, 1, 2)
        # print(f'Size of batch : {current_batch.shape}')
        # predictions = fa[gpu_id].face_detector.detect_from_batch(current_batch)
        predictions = fa.face_detector.detect_from_batch(current_batch)
        for _, prediction in enumerate(predictions):
            total_frames += 1
            if len(prediction) == 0:
                no_faces += 1
    
    print(f'Finished processing on data : {len(batches)}, on gpu : {gpu_id}', flush=True)

    return total_frames, no_faces


if __name__ == '__main__':
    # folder_name = 'sample_dir'
    
    # filelist = glob(os.path.join(folder_name, '*.mp4'))
    

    # jobs = [(vfile, i%ngpus) for i, vfile in enumerate(filelist)]
    # p = ThreadPoolExecutor(ngpus)
    # futures = [p.submit(detect_faces, j) for j in jobs]
    # _ = [r.result() for r in tqdm(as_completed(futures), total=len(futures))]

    # video_files = glob('sample_dir/000*.mp4')
    # video_files = ['sample_dir/7fcqhU-23TA.mp4', 'sample_dir/0099.mp4', 'sample_dir/IMY4UOA7E_w.mp4', 'sample_dir/0011.mp4', 'sample_dir/output.mp4']

    video_files = glob('VLOG_CROPPED/*/*/*.mp4') # SpeakerVideos/AnfisaNava/Video_ID/split.mp4

    total_video_files = len(video_files)

    jobs = [(vfile, i%ngpus) for i, vfile in enumerate(video_files)]
    p = ThreadPoolExecutor(ngpus)
    futures = [p.submit(detect_faces_single_video, j) for j in jobs]
    _ = [r.result() for r in tqdm(as_completed(futures), total=len(futures))]

    # for video_file in video_files:
    #     detect_faces_single_video((video_file, 0))