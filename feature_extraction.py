import librosa as lr
import numpy as np
import moviepy.audio.io.AudioFileClip as mpe
import matplotlib.pyplot as plt

class features():
    def __init__(self,y,sr):
        # Spectrogram using short time fourer transform
        self.sr = sr
        self.y = y
        self.S = np.abs(lr.stft(y))
        self.S_mean = np.mean(self.S, 1)  # take mean
        self.S_filtered = lr.decompose.nn_filter(self.S_mean)  # filter
        self.onset_envelope = lr.onset.onset_strength(S=self.S_filtered)
        self.onsets = lr.onset.onset_detect(onset_envelope=self.onset_envelope)

        #Find maximum frequency
        unique, counts = np.unique(np.diff(self.onsets), return_counts=True)
        try:
            self.F0 = unique[np.argmax(counts)]  # most frequent approximation is F0
        except:
            self.F0 = np.mean(np.diff(self.onsets))
        self.MaxFreq, self.NumStft = self.S.shape
        self.F0_in_Hz = self.sr/2*self.F0/self.MaxFreq


    def find_harmonics(self):
        #Find first 8 harmonics
        onsets = self.onsets[self.onsets >= self.F0 - 1]
        harmonics = onsets[0:9]
        return harmonics

    def find_relative_energies(self,harmonics):
        #Find reletive harmoinc enegies
        Ens = np.zeros(9)
        for i in range(0, harmonics.size):
            h = harmonics[i]
            Ens[i] = np.max(self.S_mean[h - 3:h + 3])
        Rel_ens = Ens[1:9] / Ens[0]
        return Rel_ens

    def get_rms_enveleope(self):
        #time domain envolope
        envelope = lr.feature.rmse(S=self.S).T
        envelope = envelope[envelope > np.mean(envelope) / 100]
        envelope = np.trim_zeros(envelope, 'fb')
        return envelope

    def get_time_features(self):
        envelope = self.get_rms_enveleope()
        #plt.figure();plt.plot(envelope)
        EoA = np.argmax(envelope)
        EoAen = float(envelope[EoA])

        Decay = envelope[EoA:-1]
        Decay = Decay[Decay>EoAen*0.75]
        EoDen = float(envelope[Decay.size + EoA])

        #fidn attack time, decay time , rooll of rate
        attack_time = EoA/self.NumStft
        decay_time = Decay.size/self.NumStft
        roll_of_rate = (EoAen-EoDen)/Decay.size
        Temp_feat = [attack_time, decay_time, roll_of_rate]
        return Temp_feat

    def get_feature_vectors(self):
         harmonics = self.find_harmonics()
         Rel_ens = self.find_relative_energies(harmonics)
         Temp_feat =self.get_time_features()
         feature_set = np.concatenate([Rel_ens, Temp_feat])
         return feature_set
