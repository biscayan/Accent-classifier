import librosa
import os
import numpy as np
from tqdm import tqdm

data_path="C:/git/download/Accented speech recognition/Accent-Classifier/data/test/"

def wav_to_mfcc(file):
    sig,rate=librosa.load(file)
    return sig, rate

def downsampling(file,outrate=8000):
    sig,rate=librosa.load(file)
    float_sig=np.array(sig,dtype=float)
    down_sig = librosa.core.resample(float_sig, rate, outrate, scale=True)
    return down_sig

def normalization(file,n_samps=240000):
    down_sig= downsampling(file)
    normed_sig = librosa.util.fix_length(down_sig, n_samps)
    normed_sig = (normed_sig - np.mean(normed_sig))/np.std(normed_sig)
    return normed_sig

def normed_mfcc(file):
    normed_sig = normalization(file)
    normed_mfcc_feat = librosa.feature.mfcc(normed_sig, 8000, n_mfcc=13)
    return normed_mfcc_feat

def mfcc_to_npy(folder):

    mfcc_vector=[]

    for (path, dir, files) in os.walk(folder):
        for file in tqdm(files, "Saving mfcc"):
            if file.endswith("wav"):
                normed_mfcc_feat=normed_mfcc(path+file)
                mfcc_vector.append(normed_mfcc_feat)

    np.save("mfcc.npy", mfcc_vector)

mfcc_to_npy(data_path)