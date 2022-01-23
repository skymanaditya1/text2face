# Multi GPU multi batch code for generating keypoints using the defined keypoint generator
import face_alignment
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np
import cv2
import os
import torch

ngpus = 1


landmarks_folder = 'landmarks_result'

fa = [face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cuda:{}'.format(id)) for id in range(ngpus)]
os.makedirs(landmarks_folder, exist_ok=True)

def generate_landmarks(vfile_gpu):
    video_file, gpu_id = vfile_gpu

    batch_size = 32
    max_frames_threshold = 1000
    landmark_threshold = 90
    landmarks_only = True

    successful_landmarks = os.path.join(landmarks_folder, 'successful_landmarks.txt')
    unsuccessful_landmarks = os.path.join(landmarks_folder, 'unsuccessful_landmarks.txt')
    failed_landmarks = os.path.join(landmarks_folder, 'failed_landmarks.txt') # Due to issues outside the control of the program
    
    # generate frames from the video file
    video_stream = cv2.VideoCapture(video_file)
    # check the total number of frames greater than threshold
    total_frames = int(video_stream.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames > max_frames_threshold:
        print(f'Landmark generation failed for video : {video_file}, continuing')
        with open(failed_landmarks, 'a') as f:
            f.write(video_file + '\n')
        return

    frames = list()
    success, image = video_stream.read()
    while success:
        frames.append(image)
        success, image = video_stream.read()

    # Create batches of data using the given batch size 
    batches = [frames[i:i+batch_size] for i in range(0, len(frames), batch_size)]
    processed = False
    while not processed:
        try:
            if batch_size == 0:
                return
            landmarks, landmarks_detected = batch_landmarks(batches, fa[gpu_id], gpu_id)
            # landmarks.extend(batch_landmarks(batches, fa[gpu_id], gpu_id))
            processed = True
        except Exception as e: # Exception arising out of CUDA memory unavailable
            print(e)
            batch_size = int(batch_size/2)
            print(f'Cuda memory unavailable, reducing batch size to : {batch_size}')
            continue

    # generate the npz file only if the landmarks generated are greater than the threshold 
    # save the npz file irrespective, but store paths of only valid npz files
    height = int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    # Add code for face cropping and transformation

    if landmarks_only:
        frames = list()
        for landmark in landmarks:
            image = np.ones((height, width), np.uint8)*255
            drawPolylines(image, landmark) # generate just the images with landmarks
            frames.append(image)
    else:
        for i, image_frame in enumerate(frames):
            # draw the landmarks on top of the image frames 
            drawPolylines(image_frame, landmarks[i])

    npz_filepath = os.path.join('/'.join(video_file.split('/')[:-1]), os.path.basename(video_file).split('.')[0] + '.npz')
    np.savez_compressed(npz_filepath, data=frames)

    # If there are no faces, landmarks won't be detected
    if landmarks_detected / total_frames * 100 >= landmark_threshold:
        with open(successful_landmarks, 'a') as f:
            f.write(npz_filepath + '\n')
    else:
        with open(unsuccessful_landmarks, 'a') as f:
            f.write(npz_filepath + '\n')
    
    # we will regenerate the sequence of landmarks again from the npz file generated
    
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

# Detect landmarks for the given batch
def batch_landmarks(batches, fa, gpu_id):
    landmarks_detected = 0
    batch_landmarks = list()
    for current_batch in batches:
        current_batch = torch.from_numpy(np.asarray(current_batch)).permute(0, 3, 1, 2).to('cuda:{}'.format(gpu_id))
        landmarks = fa.get_landmarks_from_batch(current_batch)
        landmarks_detected += len(landmarks)
        batch_landmarks.extend(landmarks)

    return batch_landmarks, landmarks_detected

def generate_landmarks_video(video_path, debug=True):
    print(f'Processing video file : {video_path}')
    batch_size = 32
    resize_dim = 256
    video_stream = cv2.VideoCapture(video_path)

    bad_filepath = 'bad_files.txt'

    image_width = int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    image_height = int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video_stream.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'Frames read : {total_frames}, image height and width: {image_height}, {image_width}')
    gpu_id = 0

    out_folder_path = os.path.join('sample_dir', os.path.basename(video_path).split('.')[0])
    # print(f'Folder name : {folder_name}')
    # out_folder_path = os.path.join('/home2/aditya1/text2face/temp_audio_files/', folder_name)
    print(f'Out folder path : {out_folder_path}')
    os.makedirs(out_folder_path, exist_ok=True)

    frames = list()
    success, image = video_stream.read()
    while success:
        frames.append(image)
        success, image = video_stream.read()

    batches = [frames[i:i+batch_size] for i in range(0, len(frames), batch_size)]
    processed = False
    while not processed:
        try:
            if batch_size == 0:
                return
            landmarks, landmarks_detected = batch_landmarks(batches, fa[gpu_id], gpu_id)
            processed = True
        except Exception as e: # Exception arising out of CUDA memory unavailable
            print(e)
            batch_size = batch_size // 2
            print(f'Cuda memory unavailable, reducing batch size to : {batch_size}')
            continue

    print(f'Image frames generated : {len(landmarks)}')

    # Draw image outline based on the image landmarks generated 
    # drop frames where the number of landmarks detected is not equal to 68
    landmark_threshold = 68
    frames_ignored = 0
    frame_ignore_threshold = 10 # reject video if more than 10% of frames are bad 
    
    # save the resized ground truth image 
    resized_gt = list()
    # save the image with landmarks drawn on the cropped image
    resized_image_landmarks = list()
    # save the image with just the landmarks 
    resized_landmarks = list()

    for i, landmark in enumerate(landmarks):
        image = frames[i]
        print(f'{i}, landmark length : {np.asarray(landmark).shape}')
        # This is done to mostly prevent frames with more than one face from getting added 
        # Unfortunately, sometimes even with one face more than one set of landmarks are detected 
        # They can potentially be removed using common area over bounding boxes 
        if (len(landmark) != landmark_threshold):
            frames_ignored += 1
            continue

        min_x, min_y, max_x, max_y = min(landmark[:, 0]), min(landmark[:, 1]), max(landmark[:, 0]), max(landmark[:, 1])

        lower_face_buffer = 0.3
        upper_face_buffer = 0.8
        # There is a possibility that the coordinates can exceed the bounds of the frame, modification of the coordinates
        x_left = max(0, int(min_x - (max_x - min_x) * lower_face_buffer))
        x_right = min(image_width, int(max_x + (max_x - min_x) * lower_face_buffer))
        y_top = max(0, int(min_y - (max_y - min_y) * upper_face_buffer))
        y_down = min(image_height, int(max_y + (max_y - min_y) * lower_face_buffer))

        print(f'{i} Coordinates after modification : {x_left, x_right, y_top, y_down}')
        size = max(x_right - x_left, y_down - y_top)

        # save the original image with the landmarks drawn 
        # outfile = os.path.join(out_folder_path, 'original_' + str(i+1).zfill(3) + '.jpg')
        # cv2.imwrite(outfile, image)

        # add centering sceheme, the centering is done only on the width side 
        # the centering is done using the formula -> (x_left + x_right) / 2 - size // 2 : (x_left + x_right) / 2 + size // 2
        sw = int((x_left + x_right) / 2 - size // 2)

        # handling edge cases 
        if (sw < 0):
            sw = 0
        if (sw + size > image_width):
            frames_ignored += 1
            continue

        # generate the original image with just the crop according to the landmarks 
        original_cropped = image[y_top:y_down, sw:sw+size]
        resized_original = cv2.resize(original_cropped, (resize_dim, resize_dim), cv2.INTER_LINEAR)
        resized_gt.append(resized_original)

        # draw the lines around the landmarks in the original image
        drawPolylines(image, landmark)

        # cropped image with the landmarks drawn
        cropped_image_landmarks = image[y_top:y_down, sw:sw+size]
        resized_im_landmarks = cv2.resize(cropped_image_landmarks, (resize_dim, resize_dim), cv2.INTER_LINEAR)
        resized_image_landmarks.append(resized_im_landmarks)

        # create an image with just the landmarks
        blank_image = np.ones((image_height, image_width), np.uint8)*255
        # draw the landmarks 
        drawPolylines(blank_image, landmark)

        # crop the landmark image and then resize 
        cropped_landmarks = blank_image[y_top:y_down, sw:sw+size]
        resized_cropped_landmarks = cv2.resize(cropped_landmarks, (resize_dim, resize_dim), cv2.INTER_LINEAR)
        resized_landmarks.append(resized_cropped_landmarks)
        
        # cropped_image = image[y_top:y_down, sw:sw+size]
        # cropped_file = os.path.join(out_folder_path, 'cropped_' + str(i+1).zfill(3) + '.jpg')
        # cv2.imwrite(cropped_file, cropped_image)

        # generate the resized version of the image
        # resized_image = cv2.resize(cropped_image, (resize_dim, resize_dim), cv2.INTER_LINEAR)
        # resized_file = os.path.join(out_folder_path, 'resized_' + str(i+1).zfill(3) + '.jpg')
        # cv2.imwrite(resized_file, resized_image)
        
        # ignore videos where frames_ignored is greater than a certain threshold 
        # print(f'Total frames : {total_frames}, ignored frames : {frames_ignored}')
        # generate the resized version of the ground truth image 

        
        # there are two images that need to be saved
        # ground truth image with just the face cropped and the image with the landmarks cropped
    
    # check if we want to save the npz files for the videos 
    if (frames_ignored / total_frames) * 100 > frame_ignore_threshold:
        print(f'Bad video {video_path}, ignoring!')
        with open(bad_filepath, 'a') as f:
            f.write(video_path + '\n')
        return

    # save the npz files inside the folder, if debug is True, save the intermediate image files generated
    # files to save are the npz files for gt image, landmarks on gt image, raw landmarks (all images are cropped and resized)
    np.savez_compressed(os.path.join(out_folder_path, 'gt.npz'), data=resized_gt)
    np.savez_compressed(os.path.join(out_folder_path, 'image_landmarks.npz'), data=resized_image_landmarks)
    np.savez_compressed(os.path.join(out_folder_path, 'landmarks.npz'), data=resized_landmarks)
    
    print(f'Total frames : {total_frames}, gt file len : {len(resized_gt)}, im landmark len : {len(resized_image_landmarks)}, landmark len : {len(resized_landmarks)}')

    # save image files only if debug is True
    if debug == True:
        for i in range(len(resized_gt)):
            gt_filepath = os.path.join(out_folder_path, 'gt_' + str(i+1).zfill(3) + '.jpg')
            im_landmarks_filepath = os.path.join(out_folder_path, 'imlandmarks_' + str(i+1).zfill(3) + '.jpg')
            landmarks_filepath = os.path.join(out_folder_path, 'landmarks_' + str(i+1).zfill(3) + '.jpg')
            
            # write the images to disk 
            cv2.imwrite(gt_filepath, resized_gt[i])
            cv2.imwrite(im_landmarks_filepath, resized_image_landmarks[i])
            cv2.imwrite(landmarks_filepath, resized_landmarks[i])


def process_video(video_path):
    print(f'Processing video file : {video_path}')
    batch_size = 32
    resize_dim = 256
    video_stream = cv2.VideoCapture(video_path)
    print(f'Frames read : {video_stream.get(cv2.CAP_PROP_FRAME_COUNT)}')
    frame_width = int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    gpu_id = 0

    folder_name = os.path.basename(video_path).split('.')[0]
    print(f'Folder name : {folder_name}')
    out_folder_path = os.path.join('/home2/aditya1/text2face/temp_audio_files/', folder_name)
    print(f'Out folder path : {out_folder_path}')
    os.makedirs(out_folder_path, exist_ok=True)

    frames = list()
    success, image = video_stream.read()
    while success:
        frames.append(image)
        success, image = video_stream.read()

    # create bataches of data using the batch size 
    batches = [frames[i:i+batch_size] for i in range(0, len(frames), batch_size)]
    processed = False
    while not processed:
        try:
            if batch_size == 0:
                return
            landmarks, landmarks_detected = batch_landmarks(batches, fa[gpu_id], gpu_id)
            processed = True
        except Exception as e: # Exception arising out of CUDA memory unavailable
            print(e)
            batch_size = int(batch_size/2)
            print(f'Cuda memory unavailable, reducing batch size to : {batch_size}')
            continue

    print(f'Image frames generated : {len(landmarks)}')

    # we have the landmarks for the entire batch as landmarks
    # Iterate through all batch landmarks
    # landmarks are returned as a batch even for a single image
    # landmark denotes the landmarks for a single image
    for index, landmark in enumerate(landmarks):
        print(np.asarray(landmark).shape) # i am printing the shape of the landmarks - what if I get more than 1 landmark 
        min_x, min_y, max_x, max_y = min(landmark[:, 0]), min(landmark[:, 1]), max(landmark[:, 0]), max(landmark[:, 1])  

        lower_face_buffer = 0.3
        upper_face_buffer = 0.8

        # prevent overflow and underflow of face coordinates from the boundaries of the image frame
        x_left = max(0, int(min_x - (max_x - min_x) * lower_face_buffer))
        x_right = min(frame_width, int(max_x + (max_x - min_x) * lower_face_buffer))
        y_top = max(0, int(min_y - (max_y - min_y) * upper_face_buffer))
        y_down = min(frame_height, int(max_y + (max_y - min_y) * lower_face_buffer))

        size = max(x_right - x_left, y_down - y_top)
        center_x, center_y = np.mean(landmark, axis=0) # not required 

        # not required 
        # sw = int(center_x - (size//2))
        # sh = int(center_y - (size//2))

        # before cropping, handle cases when the face landmarks exceed frame boundaries
        # we can choose to reject those frames, in this case we are modifying the face coordinates and generating crops
        # if (sw < 0):
        #     sw = 0
        # if y_top < 0:
        #     y_top = 0
        # if (sw + size > frame_width): # case where the right frame coordinate exceeds the image frame
        #     return

        # cropped_centered = image[sh:sh+size, sw:sw+size] # this centering strategy centers in both x and y direction which is not required due to assymettry in y-direction due to forehead
        # generate the cropped centered image for all image frames 
        image = frames[index]
        drawPolylines(image, landmark)
        # write this file 
        original_path = os.path.join(out_folder_path, 'original_' + str(index).zfill(3) + '.jpg')
        cv2.imwrite(original_path, image)

        # compute the width as (x_left + x_right) / 2 - size // 2 : (x_left + x_right) / 2 + size // 2
        sw = int((x_left + x_right) / 2 - size // 2)
        # sw = int((x_right + x_left)/2 - size//2)

        if (sw < 0):
            sw = 0
        if y_top < 0:
            y_top = 0
        if (sw + size > frame_width): # case where the right frame coordinate exceeds the image frame
            return

        cropped_centered = image[y_top:y_down, sw:sw+size]
        # outfile_2 = os.path.join(out_folder_path, os.path.basename(image_path).split('.')[0] + '_landmarks_original_cropped_centered.jpg')
        # cv2.imwrite(outfile_2, cropped_centered)

        outfile_2 = os.path.join(out_folder_path, 'cropped_' + str(index).zfill(3) + '.jpg')
        cv2.imwrite(outfile_2, cropped_centered)

        # handle cases when the calculated dimensions exceed the frame of the image before cropping
        # resize is done using bilinear interpolation
        resized = cv2.resize(cropped_centered, (resize_dim, resize_dim), cv2.INTER_LINEAR)
        outfile_2 = os.path.join(out_folder_path, 'resized_' + str(index).zfill(3) + '.jpg')
        cv2.imwrite(outfile_2, resized)

        # two sets of images need to be saved - one representing the ground truth faces and the other representing the landmarks 

# Test the setting for different speakers
def process_single_image(image_path):
    resize_dim = 256
    image = cv2.imread(image_path)
    image_height, image_width = image.shape[0], image.shape[1]
    # generate the bounding box 
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cpu')
    bb = fa.face_detector.detect_from_image(image_path)
    # bb represents the coordinates of the bounding box 
    x1,y1,x2,y2,confidence = bb[0]
    cropped_image = image[int(y1):int(y2),int(x1):int(x2)]

    # write the cropped image to folder 
    out_folder_path = '/home2/aditya1/text2face/temp_audio_files/bb_test'
    out_folder2 = 'output3_images'
    output_file = os.path.join(out_folder_path, os.path.basename(image_path).split('.')[0] + '_bb.jpg')
    cv2.imwrite(output_file, cropped_image)


    landmarks = fa.get_landmarks_from_image(image_path)
    # parse the landmarks and find the min and max values 
    min_x, min_y, max_x, max_y = min(landmarks[0][:, 0]), min(landmarks[0][:, 1]), max(landmarks[0][:, 0]), max(landmarks[0][:, 1])

    # draw the landmarks on top of the image and draw a 10% buffer using the detected landmarks 
    lower_face_buffer = 0.3
    upper_face_buffer = 0.8
    # There is a possibility that the coordinates can exceed the bounds of the frame, modification of the coordinates
    x_left = max(0, int(min_x - (max_x - min_x) * lower_face_buffer))
    x_right = min(image_width, int(max_x + (max_x - min_x) * lower_face_buffer))
    y_top = max(0, int(min_y - (max_y - min_y) * upper_face_buffer))
    y_down = min(image_height, int(max_y + (max_y - min_y) * lower_face_buffer))


    print(f'coordinates after modification')
    print(x_left, x_right, y_top, y_down)

    size = max(x_right - x_left, y_down - y_top)
    print(size)
    center_x, center_y = np.mean(landmarks[0], axis=0) # I have found centering to be useless 
    # but centering can also be used for deciding what should be the center of the cropped image with the face 

    # different centerin approaches can be used here
    # but to preserve the forehead region, centering is not applied in the y direction


    # face needs to be centered and resized
    # different centering approaches can be used here
    # to include the forehead (which are missed due to uppermost landmarks being present at the eyebrows)
    # the buffer for y_top is consequently higher 
    # for centering across the width dimension, either x_left and x_right could be calculated by shifting using the difference in distance 
    # or the mean could be calculated and x_left and x_right could be shifted by mean-size//2 and mean+size//2 respectively
    # for this code landmark centering is used for centering the image along the width dimension 

    # two things need to be generated - the ground truth frames and ground truth landmarks
    # ground truth landmarks to compute the landmark loss and ground truth frames to compute image generation loss 

    # draw the landmarks on top of the image and crop using the coordinates defined above
    drawPolylines(image, landmarks[0])
    outfile_2 = os.path.join(out_folder_path, os.path.basename(image_path).split('.')[0] + '_landmarks_original.jpg')
    cv2.imwrite(outfile_2, image)

    # In case the face is too close to the border and the coordinates cross frame boundaries
    # Perform the following to ensure that the coordinats stay within the frame
    # sw = int(center_x - (size//2))
    # sh = int(center_y - (size//2))

    # define image centers in a different fashion, this is just pure wrong, it isn't going to work
    # sw = int((x_right - x_left) // 2 - size // 2)
    # sh = int((y_down - y_top) // 2 - size // 2)

    

    # modify the dimension using centering (in both height and width dimension)

    # handle case where the landmark width exceeds image frame width
    
    # then let's modify the coordinates in both x and y directions, although it could pose problems in cases where the coordinates exceed the boundaries of the frame

    # if (sw < 0):
    #     sw = 0
    # if y_top < 0:
    #     y_top = 0
    # if (sw + size > image.shape[1]): # case where the right frame coordinate exceeds the image frame
    #     return

    cropped_image = image[y_top:y_down, x_left:x_right]
    outfile_2 = os.path.join(out_folder_path, os.path.basename(image_path).split('.')[0] + '_landmarks_original_cropped_different_buffers.jpg')
    cv2.imwrite(outfile_2, cropped_image)

    # generate another cropped image which is centered 
    

    # before cropping, handle cases when the face landmarks exceed frame boundaries
    # we can choose to reject those frames, in this case we are modifying the face coordinates and generating crops
    # lets start by modifying first only the coordinates on the width 
    sw = int((x_right + x_left)/2 - size//2) # crop frame from width -> sw : sw + size
    # no modification required in the height dimension 

   

    # cropped_centered = image[sh:sh+size, sw:sw+size] # this centering strategy centers in both x and y direction which is not required due to assymettry in y-direction due to forehead
    cropped_centered = image[y_top:y_down, sw:sw+size]
    outfile_2 = os.path.join(out_folder_path, os.path.basename(image_path).split('.')[0] + '_landmarks_original_cropped_centered.jpg')
    cv2.imwrite(outfile_2, cropped_centered)

     # modify the dimension using centering (only in the width dimension)
    # use the landmark centering coordinates to identify the middle of the cropped region
    sw = int(center_x - size // 2) # size is getting reduced due to the cropping in the upper dimension

    cropped_centered = image[y_top:y_down, sw:sw+size]
    outfile_2 = os.path.join(out_folder_path, os.path.basename(image_path).split('.')[0] + '_landmarks_original_cropped_centered_centering.jpg')
    cv2.imwrite(outfile_2, cropped_centered)

    resized = cv2.resize(cropped_centered, (resize_dim, resize_dim), cv2.INTER_LINEAR)
    outfile_2 = os.path.join(out_folder_path, os.path.basename(image_path).split('.')[0] + '_landmarks_original_cropped_resized_centering.jpg')
    cv2.imwrite(outfile_2, resized)

    # follow the centering strategy in both height and width dimension 
    sh = max(0, int(center_y - size // 2)) # this could potentially exceed the frame coordinates 

    cropped_centered = image[sh:sh+size, sw:sw+size]
    outfile_2 = os.path.join(out_folder_path, os.path.basename(image_path).split('.')[0] + '_landmarks_original_cropped_centered_centering_yalso.jpg')
    cv2.imwrite(outfile_2, cropped_centered)

    # handle cases when the calculated dimensions exceed the frame of the image before cropping
    # resize is done using bilinear interpolation
    resized = cv2.resize(cropped_centered, (resize_dim, resize_dim), cv2.INTER_LINEAR)
    outfile_2 = os.path.join(out_folder_path, os.path.basename(image_path).split('.')[0] + '_landmarks_original_cropped_resized.jpg')
    cv2.imwrite(outfile_2, resized)

    # resize the cropped_centered image to the dimension specified

    # two images need to be saved
    # write failure cases and save those to file 
    

    # print(landmarks)
    print(min_x, min_y, max_x, max_y)
    
    # crop the landmark image and resize to fixed dimension of 256x256

    # Landmark coordinates have to be used before the image is cropped
    # run landmark detection on top of the cropped image 
    
    # draw the landmarks on top of the cropped image 
    # drawPolylines(cropped_image, landmarks[0])
    # outfile_2 = os.path.join(out_folder_path, os.path.basename(image_path).split('.')[0] + '_bb_landmarks.jpg')
    # cv2.imwrite(outfile_2, cropped_image)

    # # image with just the landmarks
    # landmark_image = np.ones((cropped_image.shape[0], cropped_image.shape[1]), np.uint8)*255
    # drawPolylines(landmark_image, landmarks[0])
    # outputfile_2 = os.path.join(out_folder_path, os.path.basename(image_path).split('.')[0] + '_bb_landmarks_only.jpg')
    # cv2.imwrite(outputfile_2, landmark_image)

    print(bb)

# Code to generate the face images from the video 
def generate_images(video_file, out_dir):
    video_stream = cv2.VideoCapture(video_file)
    success, image = video_stream.read()
    frames = list()
    while success:
        frames.append(image)
        success, image = video_stream.read()
    
    print(f'Write folder path : {out_dir}')
    os.makedirs(out_dir, exist_ok=True)

    # write all frames to the disk 
    for index, frame in enumerate(frames):
        writefile = os.path.join(out_dir, str(index).zfill(3) + '.jpg')
        cv2.imwrite(writefile, frame)
    
    print(f'Frames written successfully')

# Couple of approaches can be tried out for generating the keypoints with bounding box 
# generate the bounding box frame with different buffer sizes and then finally scale to a certain size (aspect ratio cannot be maintained)
# generate the facial landmarks and take the min and max of x,y values and add different buffer sizes
if __name__ == '__main__':
    # Only those files 
    filelist = list()

    # jobs = [(vfile, i%ngpus) for i, vfile in enumerate(filelist)]
    # p = ThreadPoolExecutor(ngpus)
    # futures = [p.submit(generate_landmarks, j) for j in jobs]
    # _ = [r.result() for r in tqdm(as_compelted(futures), total=len(futures))]

    # Generate landmarks for a single file
    # fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cuda:0')
    # video_file = 'output3.mp4'
    # generate_landmarks((video_file, 0))
    # folder_path = '/home2/aditya1/text2face/temp_audio_files/0157_images'
    # image_file = os.path.join(folder_path, '004.jpg')
    # # image_file = 'output3_images/001.jpg'
    # process_single_image(image_file)

    # video_file = 'SpeakerData/videos/KritikaGoel/_m1FoLcg0jo/0483.mp4'
    # video_file = 'SpeakerData/videos/Superwoman/Q9vx4Y-HN5w/0157.mp4'
    # process_video(video_file)

    # method to generate images from a video file
    # folderpath = os.path.join('/home2/aditya1/text2face/temp_audio_files/', os.path.basename(video_file).split('.')[0] + '_images')
    # generate_images(video_file, folderpath)
    
    video_files = ['SpeakerVideos/BestDressed/U4mQQc2XUOk/0716.mp4', 'SpeakerVideos/Superwoman/H7k3bSAwbN0/0128.mp4', 'SpeakerVideos/SejalKumar/ch9WDV6wG5E/0252.mp4', 'SpeakerVideos/BestDressed/22CZdj4L8Sk/0457.mp4', 'SpeakerVideos/BestDressed/J7pLfxgfic4/0170.mp4', 'SpeakerVideos/BestDressed/J7pLfxgfic4/0296.mp4', 'SpeakerVideos/AnfisaNava/dAuJFhbtlCs/0302.mp4', 'SpeakerVideos/SejalKumar/ch9WDV6wG5E/0009.mp4', 'SpeakerVideos/AnfisaNava/wxmU_4zSTc0/0344.mp4', 'SpeakerVideos/BestDressed/Futb4fpp3Ug/0826.mp4']

    video_file = '0247.mp4'
    generate_landmarks_video(video_file)

    # # video_file = 'SpeakerData/videos/Superwoman/Q9vx4Y-HN5w/0157.mp4'
    # for video_file in video_files:
    #     # video_file = 'SpeakerData/videos/AnfisaNava/wxmU_4zSTc0/0344.mp4'
    #     video_file = video_file.replace('SpeakerVideos', 'SpeakerData/videos')
    #     generate_landmarks_video(video_file)