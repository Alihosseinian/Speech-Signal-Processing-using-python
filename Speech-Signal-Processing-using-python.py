import matplotlib.pyplot as plt
import numpy as np
from fastdtw import fastdtw
import librosa
import math
from spafe.features.lpc import lpc, lpcc
##########################################################################################################
lastnames = []
tedad = 10

for i in range(tedad):
    y, sr = librosa.load("/content/amozeshi/Hosseinian ({}).wav".format(i+1))
    lastnames.append(y)
# nemoodar
plt.rcParams.update({'figure.max_open_warning': 0})
def nemoodar (signals,x):
    for i in range(len(signals)):
        plt.figure(x*10 + i+1)
        plt.title("seda {}".format(i+1))
        plt.plot(signals[i])

nemoodar (lastnames,0)

def normal(signals):
    for i in range(len(signals)):
        temp = librosa.util.normalize(signals[i])
        for n in range(0,len(temp)):
            if abs(temp[n]) < 0.05 :
                temp[n] = 0
        temp = np.array(temp)
        signals[i] = temp
    return signals

lastnames = normal(lastnames)
# nemoodar
nemoodar (lastnames,1)
def hazf_sokoot(signals):
    for i in range(len(signals)):
        temp = []
        for data in signals[i]:
            if data != 0.0:
                temp.append(data)
        temp = np.array(temp)
        signals[i] = temp
    return signals

lastnames = hazf_sokoot(lastnames)
# nemoodar
nemoodar (lastnames,1)
##########################################################################################################
dtw_1 = np.zeros((len(lastnames),len(lastnames)))
for first in range(0,len(lastnames)):
    for second in range(0, len(lastnames)):
        dtw_1[first][second] , dummy = fastdtw(lastnames[first], lastnames[second])

for row in dtw_1:
    print(' '.join(map(str, row)))

num = 0
sum = 0
row_avg =[]
for i in range(0,tedad):
    row_sum = 0
    row_num = 0
    for j in range(0,tedad):
        if i != j:
            sum+=dtw_1[i][j]
            num+=1
            row_sum+=dtw_1[i][j]
            row_num+=1
    temp = row_sum/row_num
    row_avg.append(temp)

avg = sum/num

print("\nTotal AVG = ",avg)
print("Total MAX = ",max(row_avg))
print("Total MIN = ",min(row_avg),"\n")
for i in range(len(row_avg)):
    print("row AVG "+str(i+1),row_avg[i])

reference_signal = lastnames[0]
min_dist = abs(avg - row_avg[0])
index = 0
for i in range(len(row_avg)):
    dist = abs(avg - row_avg[i])
    if dist < min_dist:
        reference_signal = lastnames[i]
        min_dist = dist
        index = i

print("signal marja = Hosseinian",index+1)
##########################################################################################################
ali_test = []
tedad_ali = 5
for i in range(tedad_ali):
    y, sr = librosa.load("/content/test/test Hosseinian ({}).wav".format(i+1))
    ali_test.append(y)

ali_test = normal(ali_test)
ali_test = hazf_sokoot(ali_test)

dtw_ali_1 = []
for i in range(0,tedad_ali):
    temp , dummy = fastdtw(reference_signal, ali_test[i])
    dtw_ali_1.append(temp)


print("Natije Ali Test = ",dtw_ali_1)
##########################################################################################################
others_words = []
tedad_ow = 6
for i in range(tedad_ow):
    y, sr = librosa.load("/content/test/test others ({}).wav".format(i+1))
    others_words.append(y)

others_words = normal(others_words)
others_words = hazf_sokoot(others_words)

dtw_others_words = []
for i in range(0,len(others_words)):
    temp , dummy = fastdtw(reference_signal, others_words[i])
    dtw_others_words.append(temp)

print("Natije other voices other words Test = ",dtw_others_words)
##########################################################################################################

def stride_trick(a, stride_length, stride_step):
    
    nrows = ((a.size - stride_length) // stride_step) + 1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a,shape=(nrows, stride_length),strides=(stride_step*n, n))
     
def framing(sig, fs=11025, win_len=0.19, win_hop=0.08):
    
    frame_length = win_len * fs
    frame_step = win_hop * fs
    signal_length = len(sig)
    frames_overlap = frame_length - frame_step

    rest_samples = np.abs(signal_length - frames_overlap) % np.abs(frame_length - frames_overlap)
    pad_signal = np.append(sig, np.array([0] * int(frame_step - rest_samples) * int(rest_samples != 0.)))


    frames = stride_trick(pad_signal, int(frame_length), int(frame_step))
    return frames, frame_length

def ezafe_kardan_vizhegi(vizhegiha,temp):
    vizhegiha.append(np.average(temp))
    vizhegiha.append(np.mean(temp))
    vizhegiha.append(np.min(temp))
    vizhegiha.append(np.max(temp))
    return vizhegiha

def estekhraje_vizhegi(list_frames):
    list_vizhegiha = []
    for signalframes in list_frames:
        vizhegiha=[]
        for frame in signalframes:
            vizhegiha_frame = []
        
            # Energy
            temp = np.array([data * data for data in frame])
            vizhegiha_frame = ezafe_kardan_vizhegi(vizhegiha_frame,temp)
            # ZeroCrossing
            temp = librosa.feature.zero_crossing_rate(frame, 11025)
            vizhegiha_frame = ezafe_kardan_vizhegi(vizhegiha_frame,temp)
            # MFCC
            temp = librosa.feature.mfcc(frame,11025)
            vizhegiha_frame = ezafe_kardan_vizhegi(vizhegiha_frame,temp)
            # LPC
            temp = lpc(frame, fs=11025, num_ceps=13)
            vizhegiha_frame = ezafe_kardan_vizhegi(vizhegiha_frame,temp)
            # RMS
            temp = librosa.feature.rms(frame,11025)
            vizhegiha_frame = ezafe_kardan_vizhegi(vizhegiha_frame,temp)
            vizhegiha.append(vizhegiha_frame)
        list_vizhegiha.append(vizhegiha)
    return list_vizhegiha

lastnames_frames = []

for signal in lastnames:
    temp, le = framing(signal,11025)
    lastnames_frames.append(temp)
#lastnames_frames = np.ndarray(lastnames_frames)
lastnames_features = []

lastnames_features = estekhraje_vizhegi(lastnames_frames)
##########################################################################################################

feature_DTW_lastname = np.zeros((len(lastnames_features),len(lastnames_features)))
for first in range(len(lastnames_features)):
    for second in range( len(lastnames_features)):
        feature_DTW_lastname[first][second] , dummy = fastdtw(lastnames_features[first], lastnames_features[second])


for row in feature_DTW_lastname:
    print(' '.join(map(str, row)))

num = 0
sum = 0
row_avg =[]
for i in range(0,tedad):
    row_sum = 0
    row_num = 0
    for j in range(0,tedad):
        if i != j:
            sum+=feature_DTW_lastname[i][j]
            num+=1
            row_sum+=feature_DTW_lastname[i][j]
            row_num+=1
    temp = row_sum/row_num
    row_avg.append(temp)

avg = sum/num

print("\nTotal AVG = ",avg)
print("Total MAX = ",max(row_avg))
print("Total MIN = ",min(row_avg),"\n")
for i in range(len(row_avg)):
    print("row AVG "+str(i+1),row_avg[i])

reference_signal_feature = lastnames_features[0]
min_dist = abs(avg - row_avg[0])
index = 0
for i in range(len(row_avg)):
    dist = abs(avg - row_avg[i])
    if dist < min_dist:
        reference_signal_feature = lastnames_features[i]
        min_dist = dist
        index = i

print("\nReference = LastName",index+1)
##########################################################################################################
ali_frames = []

for signal in ali_test:
    temp, le = framing(signal,11025)
    ali_frames.append(temp)

ali_test_features = []

ali_test_features = estekhraje_vizhegi(ali_frames)

dtw_ali_feature = []
for i in range(0,tedad_ali):
    temp , dummy = fastdtw(reference_signal_feature, ali_test_features[i])
    # temp , dummy = fastdtw(lastnames_features[1], ali_test_features[i])
    dtw_ali_feature.append(temp)
    
print("Natije Ali Test = ",dtw_ali_feature)
##########################################################################################################
others_frames = []

for signal in others_words:
    temp, le = framing(signal,11025)
    others_frames.append(temp)

others_test_features = []

others_test_features = estekhraje_vizhegi(others_frames)

dtw_others_feature = []
for i in range(0,tedad_ow):
    temp , dummy = fastdtw(reference_signal_feature, others_test_features[i])
    dtw_others_feature.append(temp)
print("Natije Others Test = ",dtw_others_feature)