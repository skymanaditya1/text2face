# file to preprocess text and video landmarks
import os 

# filename = 'vlog_tts/vlog_train.txt'

dir_name = 'SpeakerKeypoints'
files = ['vlog_tts/vlog_train.txt', 'vlog_tts/vlog_test.txt', 'vlog_tts/vlog_val.txt']
for filename in files:
    print(f'Processing file : {filename}')
    with open(filename, 'r') as f:
        lines = f.read().splitlines()

    files_to_write = list()

    for line in lines:
        wav_file = line.split('|')[0]
        npz_file = os.path.basename(wav_file).split('.')[0] + '_landmarks.npz' 
        # check the existence of the file by constructing it 
        npz_filepath = os.path.join(os.path.join(dir_name, '/'.join(wav_file.split('/')[2:-1])), npz_file)
        if os.path.exists(npz_filepath):
            files_to_write.append(line.split('|')[1] + '|' + npz_filepath + '\n')

    print(f'Lines written : {len(files_to_write)}')

    write_filename = os.path.basename(filename).split('.')[0] + '_text_landmarks.txt'
    with open(write_filename, 'w') as f:
        for line in files_to_write:
            f.write(line)

    print(f'Written file : {write_filename}')