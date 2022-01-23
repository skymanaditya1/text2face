# Generate keypoints/facial landmarks for the frames in the video 
# 1. Generate the frame from videos 
# 2. Face detection model (not required as MEAD has only talking face videos)
# 3. Generate the facial keypoints using MEAD 
# 4. Make changes to the dataloader to process talking face facial keypoint sequence 
# 5. Add face decoder model and generate facial keypoints 

# A directory of videos is given as input
# Code for untaring the video.tar folders 
from glob import glob  
import tarfile
import os
import threading
from concurrent.futures import ThreadPoolExecutor
import cv2
import dlib 
from imutils import face_utils
import numpy as np
from tqdm import tqdm

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor.dat')

# code used to first generate the facial image with landmarks 

def untar_folder(thread_id_tar_path):
    thread_id = thread_id_tar_path[0]
    tar_path = thread_id_tar_path[1]
    # Check for the existence of extracted folder before untaring
    if os.path.isdir(tar_path.replace('.tar', '')):
        print(f'Extracted file already exists for folder : {tar_path} for thread : {thread_id}')
    else:
        print(f'Untaring {tar_path} using thread {thread_id}')
        tar = tarfile.open(tar_path)
        tar.extractall()
        tar.close()

# generate images from npz 
def process_npz(filename, image_dir):
	os.makedirs(image_dir, exist_ok=True)
	data = np.load(filename, allow_pickle=True)['data']
	for i in range(len(data)):
		cv2.imwrite(os.path.join(image_dir, str(i).zfill(3) + '.jpg'), data[i])

# write file to disk
def write_image(directory, filename, image, file_format='jpg'):
    filepath = os.path.join(directory, filename) + '.' + file_format
    os.makedirs(directory, exist_ok=True)
    cv2.imwrite(filepath, image)

def drawPolyline(image, landmarks, start, end, isClosed=False):
    points = []
    for i in range(start, end+1):
        point = [landmarks[i][0], landmarks[i][1]]
        points.append(point)

    points = np.array(points, dtype=np.int32)
    cv2.polylines(image, [points], isClosed, (0, 255, 255), 2, 16)

# Draw lines around landmarks corresponding to different facial regions
def drawPolylines(image, landmarks):
    drawPolyline(image, landmarks, 0, 16)           # Jaw line
    drawPolyline(image, landmarks, 17, 21)          # Left eyebrow
    drawPolyline(image, landmarks, 22, 26)          # Right eyebrow
    drawPolyline(image, landmarks, 27, 30)          # Nose bridge
    drawPolyline(image, landmarks, 30, 35, True)    # Lower nose
    drawPolyline(image, landmarks, 36, 41, True)    # Left eye
    drawPolyline(image, landmarks, 42, 47, True)    # Right Eye
    drawPolyline(image, landmarks, 48, 59, True)    # Outer lip
    drawPolyline(image, landmarks, 60, 67, True)    # Inner lip

def detect_landmarks(image):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	faces = detector(gray)
	landmarks = None
	for (i, face) in enumerate(faces): 
		shape = predictor(gray, face)
		landmarks = face_utils.shape_to_np(shape)

	return landmarks

def process_frames(video, image_dir, thread_id, output_folder='landmarks', write_frames=False, landmarks_only=True, landmark_detection_threshold=0.8):
	print(f'Processing video file : {video} using thread : {thread_id}')
	vidcap = cv2.VideoCapture(video)
	total_frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
	fps = vidcap.get(cv2.CAP_PROP_FPS)
	landmarks_detected = 0
	current_frame = 0
	video_sequence = list() # used for appending all images in the video
	success, image = vidcap.read()
	# check if the number of landmarks generated is significantly less than the number of frames in the video 
	while success:
		landmarks = detect_landmarks(image)
		if landmarks is not None:
			landmarks_detected += 1
			# overlay the landmarks on top of the face image or generate landmarks only
			if landmarks_only:
				image = np.ones((image.shape[0], image.shape[1]), np.uint8)*255
			drawPolylines(image, landmarks)
			
			video_sequence.append(image)
			# save landmark image if write_frames is set to True
			if write_frames:
				os.makedirs(image_dir, exist_ok=True)
				write_image(image_dir, str(current_frame).zfill(3), image)

		success, image = vidcap.read()
		current_frame += 1

	# save the sequence of landmarks only if landmark detection rate is greater than threshold
	if landmarks_detected / current_frame >= landmark_detection_threshold:
		dirpath = '/'.join(video.split('/')[:-1])
		npz_folder = os.path.join(dirpath, output_folder)
		os.makedirs(npz_folder, exist_ok=True)
		npz_filename = os.path.basename(video).split('.')[0] + '.npz'
		output_path = os.path.join(npz_folder, npz_filename)
		np.savez_compressed(output_path, data=video_sequence)
		print(f'Saving npz file to : {output_path} using thread : {thread_id}', flush=True)

def process_video(thread_id_video_path):
	# folder_path -> MEAD/M003
	thread_id = thread_id_video_path[0]
	# folder_path = thread_id_folder_path[1]
	video_path = thread_id_video_path[1]
	# print(f'Processing folder : {folder_path} with thread : {thread_id}')
	print(f'Processing video : {video_path} with thread : {thread_id}', flush=True)
	# video_files = glob(folder_path + '/video/front/*/*/*.mp4')
	# for video_file in tqdm(video_files):
	# 	landmark_images_folder = os.path.join('/'.join(video_file.split('/')[:-1]), video_file.split('/')[-1].split('.')[0] + '_landmark_image') # This stores landmarks for all frames in a video
	# 	process_frames(video_file, landmark_images_folder, thread_id, output_folder='landmarks_npz') # This folder stores the landmark npz for all videos in the parent folder
	# landmark_images_folder (one folder per video) -> 002_landmark_image
	# print('Starting processing')
	landmark_images_folder = os.path.join('/'.join(video_path.split('/')[:-1]), os.path.basename(video_path).split('.')[0] + '_landmark_image')
	# print(f'landmark_images_folder {landmark_images_folder}')
	process_frames(video_path, landmark_images_folder, thread_id, output_folder='landmarks_npz') # landmarks_npz stores npz landmarks for all videos in that folder

if __name__ == '__main__':
	thread_limit = 5
    # tar_folders = glob('tacotron2/MEAD/*/video.tar')
    # with ThreadPoolExecutor(thread_limit) as e:
    #     results = e.map(untar_folder, ((thread_id, folder) for thread_id, folder in enumerate(tar_folders)))

	# There were errors when running multi-threading code on multiple folders
	# Code performs multi-threading processing on videos 
	videos = glob('MEAD/*/video/front/*/*/*.mp4')
	# video_folders = glob('MEAD_test/*')
	with ThreadPoolExecutor(thread_limit) as e:
		results = e.map(process_video, ((thread_id, video) for thread_id, video in enumerate(videos)))

	# Single run, there is problem with the multi-threading code, I need to run threads at the video level
	# video_file = 'test/030.mp4'
	# save_dir = 'test/030_landmarks'
	# process_frames(video_file, save_dir, 1)

	# Specify folder path 
	# folders = glob('MEAD_test/*') # Gives list of folders M003, W004 etc.
	# process_video(folders[0])

	# sample code to process npz file 
	# process_npz('test/026.npz', 'test/landmark_npz_026')