import librosa
import torch
import os
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

DATA_PATH = "C:/git/download/Accented speech recognition/Accent-Classifier/data/test/"

# Input: Folder Path
# Output: Tuple (Label, Indices of the labels, one-hot encoded labels)
def get_labels(path=DATA_PATH):
    #(1) label
    labels = os.listdir(path)
    #(2) label indices
    label_indices = np.arange(0, len(labels))
    torch_labels = torch.arange(1)
    torch_labels = torch_labels.reshape(1, 1)
    #(3) one hot encoding
    num_classes = 3
    one_hot = (torch_labels == torch.arange(num_classes).reshape(1, num_classes)).float()

    return labels,label_indices,one_hot

# Handy function to convert wav to mfcc
def wav_to_mfcc(path=DATA_PATH):
    for (cur_path, dir, files) in os.walk(path):
        for filename in files:
            wav_file = filename.split(".")[1]
            if wav_file == 'wav':
                sig, rate = librosa.load(cur_path+"/"+filename)
                float_sig=np.array(sig,dtype=float)
                down_sig = librosa.core.resample(float_sig, rate, 16000, scale=True)
                mfcc = librosa.feature.mfcc(down_sig, sr=16000,n_mfcc=13)
                return mfcc

# Save mfcc ndarray to npy file 
def save_mfcc(path=DATA_PATH):
    labels, _, _ = get_labels(path)

    for label in labels:
        # Init mfcc vectors
        mfcc_vectors = []

        wavfiles = [path + label + '/' + wavfile for wavfile in os.listdir(path + '/' + label)]

        for wavfile in tqdm(wavfiles, "Saving vectors of label - '{}'".format(label)):
            mfcc = wav_to_mfcc(wavfile)
            mfcc_vectors.append(mfcc)
        np.save(label + '.npy', mfcc_vectors)

def get_train_test(split_ratio=0.7, random_state=42):
    # Get available labels
    labels, indices, _ = get_labels(DATA_PATH)

    # Getting first arrays
    X = np.load(labels[0] + '.npy')
    y = np.zeros(X.shape[0])

    # Append all of the dataset into one single array, same goes for y
    for i, label in enumerate(labels[1:]):
        x = np.load(label + '.npy')
        X = np.vstack((X, x))
        y = np.append(y, np.full(x.shape[0], fill_value= (i + 1)))

    assert X.shape[0] == len(y)

    return train_test_split(X, y, test_size= (1 - split_ratio), random_state=random_state, shuffle=True)

def prepare_dataset(path=DATA_PATH):
    labels, _, _ = get_labels(path)
    data = {}
    
    for label in labels:
        data[label] = {}
        data[label]['path'] = [path  + label + '/' + wavfile for wavfile in os.listdir(path + '/' + label)]

        vectors = []

        for wavfile in data[label]['path']:
            sig, rate = librosa.load(wavfile)
            # Downsampling
            float_sig=np.array(sig,dtype=float)
            down_sig = librosa.core.resample(float_sig, rate, 16000, scale=True)
            mfcc = librosa.feature.mfcc(down_sig, sr=16000,n_mfcc=13)
            vectors.append(mfcc)

        data[label]['mfcc'] = vectors

    return data

def load_dataset(path=DATA_PATH):
    data = prepare_dataset(path)

    dataset = []

    for key in data:
        for mfcc in data[key]['mfcc']:
            dataset.append((key, mfcc))

    return dataset


#if __name__ == '__main__':
    #save_mfcc()
    #print(prepare_dataset())
    #print(load_dataset())