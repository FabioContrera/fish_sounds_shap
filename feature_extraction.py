##Loading libraries
import numpy as np
import pandas as pd
import librosa
import librosa.display
import scipy
import os
import matplotlib.pyplot as plt
from scipy import signal
from datetime import datetime, timedelta
import wavio
from scipy.signal import butter, lfilter


##Filtering
def filter_signal(y=None, fB1=None,fB2=None,sr=None):
    # filtering ---------------------------------------------------------------
    rp      = 1
    rs      = 20
    Nf      = sr/2
    bt      = fB1*0.2
    fBt1    = fB1 - bt
    fBt2    = fB2 + bt
    
    NN, Wn  = signal.ellipord([fB1/Nf,fB2/Nf], [fBt1/Nf,fBt2/Nf], rp ,rs)
    bB, aB  = signal.ellip(NN, rp, rs, Wn, 'bandpass')
    yfilt   = signal.lfilter(bB,aB,y)
    
    return yfilt

def filter_signal_low(y=None, fB1=None,sr=None):
    # filtering ---------------------------------------------------------------
    rp      = 1
    rs      = 20
    Nf      = sr/2
    bt      = fB1*0.2
    fBt1    = fB1 - bt
    
    NN, Wn  = signal.ellipord([fB1/Nf], [fBt1/Nf], rp ,rs)
    bB, aB  = signal.ellip(NN, rp, rs, Wn, 'lowpass')
    yfilt   = signal.lfilter(bB,aB,y)
    
    return yfilt

def butter_bandpass(lowcut, highcut, fs, order=10):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


##Loading data
tabela = pd.read_excel('features_time.xlsx')

#Reading tabular data
n_arq = np.arange(0, len(tabela))
#n_arq = np.array([3])

#Finding time and min freq of each fish call
for arq in n_arq:
    arquivo1   = tabela.datetime[arq] + timedelta(hours=3) + timedelta(seconds=1)
    x          = arquivo1.strftime('%Y%m%d-%H%M%SU-52734-1x.wav')
    
    #y, sr      = librosa.load(x, sr=52_734)
    sr = 52734
    y0 = wavio.read(x)
    y= y0.data[:, 0]
    
    
    begin_time = tabela.begin_time[arq]
    end_time   = tabela.end_time[arq]
    
    window_time = 0.5*sr
    
    start_sample = int(begin_time*sr-window_time)
    end_sample   = int(end_time*sr+window_time)
    
    if begin_time<window_time:
        sample     = y[0:end_sample]
        
    elif len(y)*sr - end_time<window_time:
        sample     = y[start_sample:-1]
        
    else:
        sample     = y[start_sample:end_sample]

    #filtro
    freq_min   = tabela.lo_freq[arq]
    freq_max   = tabela.hi_freq[arq] 
    #freq_min    = 500
    #freq_max    = 700
    freq_lp     = 1000
    
    q           = 10
    sr_new      = sr/q
    t_filt      = int(len(sample)/q)
    
    #aplicar passa-baixa
    yfilt      = filter_signal_low(sample, freq_lp, sr)
    
    #appying downsampling
    yfilt2    = signal.resample(yfilt, t_filt)
  
    #applyingbandpass
    yfilt3     = filter_signal(yfilt2, freq_min, freq_max, sr_new)
    
    yfilt4 = yfilt3
    sr_new1 = sr_new

##################### FEATURE EXTRACTION ###########################   
    
    ##Beat extraction
    tempo, beat_frames = librosa.beat.beat_track(y=yfilt3, sr=sr_new1, hop_length=64)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr_new) #takes the frames of each beat and
                                                                 #converts in tempo
    beat_time_diff=np.ediff1d(beat_times)  #calculates differences between successive elements in the array
    beat_nums = np.arange(1, np.size(beat_times))  #number of beats
    
    ##Computing MFCCs
    mfccs = librosa.feature.mfcc(y=yfilt4, sr=sr_new1, n_mfcc=26, hop_length=64)
    
    ##Spectral centroid
    cent = librosa.feature.spectral_centroid(y=yfilt4, sr=sr_new1, hop_length=64)
    
    ##Spectral rolloff
    rolloff = librosa.feature.spectral_rolloff(y=yfilt4, sr=sr_new1, hop_length=64)

    ##Zero crossing rate
    zrate=librosa.feature.zero_crossing_rate(yfilt4, hop_length=64)
    
    ##Spectral flux
    onset_env = librosa.onset.onset_strength(y=yfilt4, sr=sr_new1)
    
    spec_flux = np.mean(onset_env)
    

    ##Data frames
    
    #mfccs
    mfccs_mean=np.mean(mfccs,axis=1)  #media mfccs
 
    #Generate the MFCCs Dataframe
    mfccs_df=pd.DataFrame()
    for i in range(0,26):
        mfccs_df['mfccs'+str(i)]=mfccs_mean[i] #media
    mfccs_df.loc[0]=mfccs_mean 
    mfccs_df
    
    #spec centroid
    cent_mean=np.mean(cent, axis=1)[0]
    cent_df=pd.DataFrame()
    cent_df['cent']=cent_mean
    cent_df.loc[0]=cent_mean
    cent_df
    
    #spec rolloff
    rolloff_mean=np.mean(rolloff,axis=1)[0]
    rolloff_df=pd.DataFrame()
    rolloff_df['rolloff']=rolloff_mean
    rolloff_df.loc[0]=rolloff_mean
    rolloff_df
    
    #Zero crossing rates
    zrate_mean=np.mean(zrate,axis=1)[0]
    zrate_df=pd.DataFrame()
    zrate_df['zrate']=zrate_mean
    zrate_df.loc[0]=zrate_mean
    zrate_df
    
    #Spectral flux
    specflux_df=pd.DataFrame()
    specflux_df['spec_flux']=spec_flux
    specflux_df.loc[0]=spec_flux
    specflux_df

    #beat and tempo
    beat_df=pd.DataFrame()
    beat_df['tempo']=tempo
    beat_df.loc[0]=tempo
    beat_df
    
    #classe
    classe=pd.DataFrame()
    classe['class']=tabela['class'][arq]
    classe.loc[0] =tabela['class'][arq]
    
    #final df:
    final_df  = pd.concat((mfccs_df, cent_df, rolloff_df, zrate_df, specflux_df, beat_df, classe),axis=1)
    if arq == 0:

        final_df2 = pd.DataFrame()
    final_df2 =pd.concat([final_df, final_df2], axis=0)
    
    
    final_df2.to_excel('features_librosa.xlsx', sheet_name='featureset2_librosa.xlsx', index=False)