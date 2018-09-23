#import imageio
#imageio.plugins.ffmpeg.download()
import numpy as np
import matplotlib.pyplot as plt
import librosa as lr
import os
import moviepy.audio.io.AudioFileClip as mpe
import feature_extraction

#lr.display.waveplot(y,sr)
#plt.specgram(sound_array)
#plt.figure()
#log_power = lr.logamplitude(S)
#lr.display.specshow(log_power,x_axis='time' , y_axis='linear')
#plt.colorbar()
#plt.show()
#plt.figure()
#lr.display.specshow(log_power,x_axis='time' , y_axis='log')
#plt.colorbar()
#plt.show()

plt.interactive(False)

class sig():
    def __init__(self):
        self.y=[]
        self.sr=0

class get_signals():

    def __init__(self, violin=False , trupet=False ,flute = False , guitar = False , cello=False , viola = False):
        self.violin = violin
        self.trumpet = trupet
        self.flute = flute
        self.guitar = guitar
        self.cello = cello
        self.viola = viola
        self.X=[] #feature vector array
        self.Y=[] #feature target array

    #read_violin
    def read_violin(self):
        sigs = []
        i=1
        if(self.violin):
            print('start violin feature extraction')
            os.chdir(os.getcwd() + '\\violin')
            for path in os.listdir():
                if (i % 100 == 0):
                    print("violin : " ,i, "/", np.size(os.listdir()))
                #y, sr = lr.load(path)
                signal = sig()
                audio = mpe.AudioFileClip(path)
                signal.y = np.mean(audio.to_soundarray(buffersize=200), 1)
                signal.sr = audio.fps  # sampling rate
                feat = feature_extraction.features(signal.y, signal.sr)
                feat_vec = feat.get_feature_vectors()
                self.X.append(feat_vec.tolist())
                self.Y.append(0)       #0 for violin
                sigs.append(signal)
                i= i + 1
            os.chdir("..")
        return sigs


    #read_trumpet
    def read_trumpet(self):
        sigs = []
        i = 1
        if(self.trumpet):
            print('start trumpet feature extraction')
            os.chdir(os.getcwd() + '\\trumpet')
            for path in os.listdir():
                if(i%100==0):
                    print("trumpet : " ,i ,"/" , np.size(os.listdir()))
                #y, sr = lr.load(path)
                signal = sig()
                audio = mpe.AudioFileClip(path)
                signal.y = np.mean(audio.to_soundarray(buffersize=200), 1)
                signal.sr = audio.fps  # sampling rate
                feat = feature_extraction.features(signal.y, signal.sr)
                feat_vec = feat.get_feature_vectors()
                self.X.append(feat_vec.tolist())
                self.Y.append(1)       #1 for trumpet
                sigs.append(signal)
                i = i + 1
            os.chdir("..")
        return sigs

    # read_flute
    def read_flute(self):
        sigs = []
        i = 1
        if (self.flute):
            print('start flute feature extraction')
            os.chdir(os.getcwd() + '\\flute')
            for path in os.listdir():
                if (i % 100 == 0):
                    print("flute : " , i, "/", np.size(os.listdir()))
                # y, sr = lr.load(path)
                signal = sig()
                audio = mpe.AudioFileClip(path)
                signal.y = np.mean(audio.to_soundarray(buffersize=200), 1)
                signal.sr = audio.fps  # sampling rate
                feat = feature_extraction.features(signal.y, signal.sr)
                feat_vec = feat.get_feature_vectors()
                self.X.append(feat_vec.tolist())
                self.Y.append(2)       #2 for flute
                sigs.append(signal)
                i= i + 1
            os.chdir("..")
        return sigs

    # read_guitar
    def read_guitar(self):
        sigs = []
        i = 1
        if (self.guitar):
            print('start guitar feature extraction')
            os.chdir(os.getcwd() + '\\guitar')
            for path in os.listdir():
                if (i % 100 == 0):
                    print("guitar : " ,i, "/", np.size(os.listdir()))
                # y, sr = lr.load(path)
                signal = sig()
                audio = mpe.AudioFileClip(path)
                signal.y = np.mean(audio.to_soundarray(buffersize=200), 1)
                signal.sr = audio.fps  # sampling rate
                feat = feature_extraction.features(signal.y, signal.sr)
                feat_vec = feat.get_feature_vectors()
                self.X.append(feat_vec.tolist())
                self.Y.append(3)       #3 for guitar
                sigs.append(signal)
                i= i + 1
            os.chdir("..")
        return sigs

    # read_cello
    def read_cello(self):
        sigs = []
        i = 1
        if (self.cello):
            print('start cello feature extraction')
            os.chdir(os.getcwd() + '\\cello')
            for path in os.listdir():
                if (i % 100 == 0):
                    print("cello : ", i, "/", np.size(os.listdir()))
                # y, sr = lr.load(path)
                signal = sig()
                audio = mpe.AudioFileClip(path)
                signal.y = np.mean(audio.to_soundarray(buffersize=200), 1)
                signal.sr = audio.fps  # sampling rate
                feat = feature_extraction.features(signal.y, signal.sr)
                feat_vec = feat.get_feature_vectors()
                self.X.append(feat_vec.tolist())
                self.Y.append(4)  # 4 for cello
                sigs.append(signal)
                i = i + 1
            os.chdir("..")
        return sigs


    # read_viola
    def read_viola(self):
        sigs = []
        i = 1
        if (self.viola):
            print('start viola feature extraction')
            os.chdir(os.getcwd() + '\\viola')
            for path in os.listdir():
                if (i % 100 == 0):
                    print("viola : ", i, "/", np.size(os.listdir()))
                # y, sr = lr.load(path)
                signal = sig()
                audio = mpe.AudioFileClip(path)
                signal.y = np.mean(audio.to_soundarray(buffersize=200), 1)
                signal.sr = audio.fps  # sampling rate
                feat = feature_extraction.features(signal.y, signal.sr)
                feat_vec = feat.get_feature_vectors()
                self.X.append(feat_vec.tolist())
                self.Y.append(5)  # 5 for viola
                sigs.append(signal)
                i = i + 1
            os.chdir("..")
        return sigs


