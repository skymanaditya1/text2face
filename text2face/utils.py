# load the text and filename from the text file
def load_text_landmarkpath(filename, split='|'):
    with open(filename, 'r') as f:
        text_landmarkpaths = f.read().splitlines()

    text_and_landmarkpaths = [line.strip().split(split) for line in text_landmarkpaths]
    return text_and_landmarkpaths