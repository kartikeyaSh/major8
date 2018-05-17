import numpy as np
import cv2
import pandas as pd 
from keras.preprocessing.text import text_to_word_sequence
from nltk import FreqDist
from keras.preprocessing.sequence import pad_sequences

frame_width, frame_height = (112,112)

def load_labels(filename):
    f = open(filename)
    lines = f.readlines()
    labels = []
    
    for line in lines:
        label = (line.split('\t'))[1]
        labels.append((label.split('\n'))[0])

    f.close()
    
    return labels
    
def convert_video_to_np_array(path, max_frames):

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise Exception('Video not found.')

    end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if end_frame > max_frames:
        end_frame = max_frames

    frames = []
    for i in range(end_frame):
        ret, frame = cap.read()
        frame = cv2.resize (frame,(frame_width, frame_height))
        frames.append(frame)

    video = np.array(frames, dtype=np.float32)
    #to convert it into dimension ordering of theano
    video = video.transpose(3, 0, 1, 2)
    return video

def convert_video_to_np_array_untransposed(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise Exception('Video not found.')

    end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = []
    for i in range(end_frame):
        ret, frame = cap.read()
        frame = cv2.resize (frame,(frame_width, frame_height))
        frames.append(frame)

    video = np.array(frames, dtype=np.float32)
    return video
    
def convert_video_to_np_array_multiple(paths):
    videos = convert_video_to_np_array_untransposed(paths[0])
    for i in range(1,len(paths)):
        videos = np.concatenate((videos,convert_video_to_np_array_untransposed(paths[i])))

    videos= videos.transpose(3,0,1,2)
    return videos

        
def get_frames_count (path):

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise Exception('Video not found')

    num_frames = int( cap.get( cv2.CAP_PROP_FRAME_COUNT))
    
    return num_frames

def get_video_duration (path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise Exception('Video not found')
    
    num_frames = int( cap.get( cv2.CAP_PROP_FRAME_COUNT))

    #frames per second
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    duration = num_frames/fps

    return duration


def prepare_data():
    l=[str(i)+'.csv' for i in range(1,81)]
    all_data=[]
    for f in l:
        df = pd.read_csv(f, sep=',')
        df = df.values
        df = df[:,1:]
        v=np.array([2000 for _ in range(0,4096)])
        num_features = df.shape[0]

        if num_features > 46:
            df=df[0:46]
        if num_features < 46:
            a=v[:]
            for _ in range(num_features + 1, 46):
                a = np.vstack((a,v))

            df = np.vstack((df,a))


        all_data.append(df)
    return np.array(all_data)


def prepare_text_data():
    op_length = 25
    file_name='des80.txt'

    f=open(file_name)
    
    sentences=[]
    for line in f:
        l=line.split()
        des=" ".join(l[2:])
        if len(des)>0:
            sentences.append(text_to_word_sequence(des))

    vocab=set()
    for l in sentences:
        for word in l:
            vocab.add(word)
    
    vocab_length=len(vocab)
    #print vocab_length
    #print vocab
    dist = FreqDist(np.hstack(sentences))
    output_vocab = dist.most_common(vocab_length-1)

    y_ix_to_word = [word[0] for word in output_vocab]
    y_ix_to_word.insert(0, 'ZERO')
    y_ix_to_word.append('UNK')
    y_word_to_ix = {word:ix for ix, word in enumerate(y_ix_to_word)}

    for i, sentence in enumerate(sentences):
        for j, word in enumerate(sentence):
            if word in y_word_to_ix:
                sentences[i][j] = y_word_to_ix[word]
            else:
                sentences[i][j] = y_word_to_ix['UNK']
    sentences = pad_sequences(sentences, maxlen=op_length, dtype='int32')
    return sentences, y_word_to_ix, y_ix_to_word


def prepare_test_data():
    l=[str(i)+'.csv' for i in range(81,101)]
    all_data=[]
    for f in l:
        df = pd.read_csv(f, sep=',')
        df = df.values
        df = df[:,1:]
        v=np.array([2000 for _ in range(0,4096)])

        num_features = df.shape[0]

        if num_features > 46:
            df=df[0:46]
        if num_features < 46:
            a=v[:]
            for _ in range(num_features + 1, 46):
                a = np.vstack((a,v))

            df = np.vstack((df,a))

        all_data.append(df)

    return np.array(all_data)
