import numpy as np

from keras.models import Model, Sequential
from keras.layers import LSTM, BatchNormalization, Convolution3D, Dense, Dropout, Flatten, Input, MaxPooling3D, TimeDistributed, ZeroPadding3D

from utilities import *

def format_video_array(v_array, clips_count, length=16):
    v_array = v_array.transpose(1,0,2,3)
    v_array = v_array[:clips_count * length, :, :, :]
    v_array = v_array.reshape((clips_count, length, 3, 112, 112))
    v_array = v_array.transpose(0,2,1,3,4)
    return v_array

def extract_clip(video_path, threshold, save_file_name):
    input_size = (112,112)
    length = 16
    labels = load_labels('/home/kartikeya/major_project_7th_sem/data/labels.txt')
    print("Please wait video is being loaded...")

    duration = get_video_duration(video_path)
    frames_count = get_frames_count(video_path)
    fps = frames_count/duration

    if duration > 30:
        duration = 30
        max_frame=int(duration*fps)
        frames_count = max_frame

        v_array = convert_video_to_np_array(video_path, max_frame)
    else:
        v_array = convert_video_to_np_array(video_path, frames_count)

    print("Duration of video in seconds: {:.1f}".format(duration))
    print("Frames per second: {:.1f}".format(fps))
    print("Number of frames: {}".format(frames_count))

    clips_count = frames_count // length
    v_array = format_video_array(v_array, clips_count, length)

    print ("Loading 3D Convolution Network...")

    model = C3D_model()
    model.compile(optimizer='sgd', loss='mse')

    mean_total = np.load("data/c3d-sports1M_mean.npy")
    mean = np.mean(mean_total, axis=(0,2,3,4), keepdims=True)

    print "Extracting Features..."

    X = v_array - mean
    

    Y = model.predict(X, batch_size =1, verbose=1)


    print "Saving As {0}...".format(save_file_name)
    import pandas as pd 
    df = pd.DataFrame(Y)
    df.to_csv(save_file_name)
    #np.savetxt("1.csv",Y)

    
def C3D_model():
    model = Sequential()

    #1st group
    model.add(Convolution3D(64,3,3,3,activation='relu',
                            border_mode='same',name='conv1',
                            subsample=(1,1,1), input_shape=(3,16,112,112),
                            trainable=False))
    model.add(MaxPooling3D(pool_size=(1,2,2), strides=(1,2,2),
                           border_mode='valid',name='pool1'))

    #2nd group
    model.add(Convolution3D(128,3,3,3, activation='relu',
                            border_mode='same',name='conv2',
                            subsample=(1,1,1),trainable=False))
    model.add(MaxPooling3D(pool_size=(2,2,2),strides=(2,2,2),
                           border_mode='valid', name='pool2'))

    #3rd group
    model.add(Convolution3D(256,3,3,3, activation='relu',
                            border_mode='same', name='conv3a',
                            subsample=(1,1,1), trainable=False))
    model.add(Convolution3D(256,3,3,3, activation='relu',
                            border_mode='same', name='conv3b',
                            subsample=(1,1,1), trainable=False))
    model.add(MaxPooling3D(pool_size=(2,2,2),strides=(2,2,2),
                           border_mode='valid', name='pool3'))

    #4th group
    model.add(Convolution3D(512,3,3,3, activation='relu',
                            border_mode='same',name='conv4a',
                            subsample=(1,1,1), trainable=False))
    model.add(Convolution3D(512,3,3,3, activation='relu',
                            border_mode='same',name='conv4b',
                            subsample=(1,1,1), trainable=False))
    model.add(MaxPooling3D(pool_size=(2,2,2),strides=(2,2,2),
                           border_mode='valid', name='pool4'))
    
    #5th group
    model.add(Convolution3D(512,3,3,3, activation='relu',
                            border_mode='same', name='conv5a',
                            subsample=(1,1,1),trainable=False))
    model.add(Convolution3D(512,3,3,3, activation='relu',
                            border_mode='same', name='conv5b',
                            subsample=(1,1,1),trainable=False))
    model.add(ZeroPadding3D(padding=(0,1,1), name='zeropadding'))
    model.add(MaxPooling3D(pool_size=(2,2,2),strides=(2,2,2),
                           border_mode='valid', name='pool5'))
    model.add(Flatten(name='flatten'))

    #FC group
    model.add(Dense(4096, activation='relu', name='fc6', trainable=False))
    model.add(Dropout(0.5, name='do1'))
    model.add(Dense(4096, activation='relu', name='fc7'))
    model.add(Dropout(0.5, name='do2'))
    model.add(Dense(487, activation='softmax', name='fc8'))

    model.load_weights('data/c3d-sports1M_weights.h5')

    #pop the last 4 layers of the model
    for _ in range(0,4):
        model.layers.pop()
    model.outputs= [model.layers[-1].output]
    model.layers[-1].outbound_nodes=[]
    
    return model                          

#paths=['7.MKV','8.MKV','9.MP4','10.MP4','11.MKV','12.MP4','13.MKV','14.MKV','15.MKV','16.MP4','17.MKV','18.MKV','19.MKV','20.MKV','21.MKV','22.MP4','23.MP4','24.MP4','25.MP4','26.MKV','27.MKV','28.MKV','29.MP4','30.MP4','31.MP4','32.MP4','33.MKV','34.MP4','35.MP4','36.MP4','37.MKV','38.MKV','39.MP4','40.MKV','41.MP4','42.MP4','43.MP4','44.MP4','45.MKV','46.MP4','47.MP4','48.MP4']

#paths=['49.MP4','50.MKV','51.MP4','52.MP4','53.MP4','54.MP4','55.MP4','56.MP4','57.MP4','58.MP4','59.MP4','60.MP4','61.MP4','62.MP4','63.MP4','64.MP4','65.MP4','66.MKV','67.MP4','68.MP4','69.MKV','70.MP4','71.MP4','72.MP4','73.MP4','74.MP4','75.MP4','76.MP4','77.MP4','78.MP4','79.MP4','80.MP4','81.MP4','82.MP4','83.MKV','84.MP4','85.MP4','86.MP4','87.MP4','88.MP4','89.MP4','90.MP4','91.MP4','92.MP4','93.MP4','94.MP4','95.MP4','96.MKV','97.MP4','98.MP4','99.MKV','100.MP4']
#i=49
#same_path="/home/kartikeya/major_project_7th_sem/videos/"
#for video_path in paths:

#    save_path = str(i)
#    i+=1
#    extract_clip(same_path+video_path,0.2, save_path+'.csv')

video_path = str(raw_input("Enter the path of video: "))
save_path = str(raw_input("Enter save file name: "))
extract_clip(video_path,0.2, save_path)

