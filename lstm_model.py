import librosa
import os
import random
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder

#data preparation

def data_path(loc, directory, filename, numbering, format):
    delimiter = '/'
    path = ( loc + delimiter + directory + delimiter +
           filename + numbering + format)
    return path

#train_path="C:/git/download/Accented speech recognition/Accent-Classifier/data/train"
#train_path="C:/git/download/Accented speech recognition/Accent-Classifier/data/test"

#feature extraction 

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

#device setting

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(777)

#parameters

num_epochs=500
learning_rate=0.0001
batch_size=10

#dataset

class Accent_dataset(Dataset):
    def __init__(self, train = True, transform = None):
        label_list = [0, 1, 2]
        self.label_accent = ['arabic', 'english', 'spanish']
        self.train = train
        num_data   =  40
        num_train  =  30
        num_test   =  10
        #onehot_encoder = OneHotEncoder(sparse=False)

        if self.train == True: 
            self.train_data   =  []
            self.train_label  =  []
            
            print("\n\n==== Train Data:")
            for item in label_list:
                for i in range(1, num_train + 1):
                    path = data_path(loc = 'C:/git/download/Accented speech recognition/Accent-Classifier/data/train', 
                              directory = self.label_accent[item],filename = self.label_accent[item],
                              numbering = str(i), format = '.wav')
                    mfcc = normed_mfcc(path)
                    self.train_data.append(mfcc)
                    self.train_label.append(item)
                    
            self.train_label = np.array(self.train_label)
            self.train_data = np.concatenate(self.train_data)
            self.train_data = self.train_data.reshape(num_train*3, self.train_data.shape[1], 13) #(90,469,13)

            #self.train_label = onehot_encoder.fit_transform(self.train_label.reshape(len(self.train_label), 1)) #(90,3)

            self.train_label = torch.cuda.LongTensor(self.train_label)
            self.train_data = torch.cuda.FloatTensor(self.train_data)
            print("=== Dataset Download Complete !!")
            print("Shape:",self.train_data.shape)
            print("Shape:",self.train_label.shape)
            
        else:
            self.test_data   =  []
            self.test_label  =  []
            
            print("\n\n=== Test Data:")
            for item in label_list:
                for i in range(num_train + 1, num_data + 1):              
                    path = data_path(loc = 'C:/git/download/Accented speech recognition/Accent-Classifier/data/test', 
                              directory = self.label_accent[item],filename = self.label_accent[item],
                                 numbering = str(i), format = '.wav')
                    mfcc = normed_mfcc(path)
                    self.test_data.append(mfcc)
                    self.test_label.append(item)
                    
            self.test_label = np.array(self.test_label)
            self.test_data = np.concatenate(self.test_data)
            self.test_data = self.test_data.reshape(num_test*3, self.test_data.shape[1], 13)

            #self.test_label = onehot_encoder.fit_transform(self.test_label.reshape(len(self.test_label), 1))

            self.test_label = torch.cuda.LongTensor(self.test_label) 
            self.test_data = torch.cuda.FloatTensor(self.test_data) 
            print("=== Dataset Download Complete !!")
            print("Shape:",self.test_data.shape)
            print("Shape:",self.test_label.shape)
        
    def __getitem__(self, index):
        if self.train:
            return self.train_data[index], self.train_label[index]
        else:
            return self.test_data[index], self.test_label[index]
    
    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

#dataloader
#,transform=transforms.ToTensor()
def data_loader():
    train_dataset = Accent_dataset(train = True)
    test_dataset = Accent_dataset(train = False)

    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = 0, drop_last = True)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False, num_workers = 0, drop_last = True)
    return train_loader, test_loader

#lstm model

class LSTM_model(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,num_layers):
        super(LSTM_model,self).__init__()

        self.input_size=input_size
        self.hidden_size=hidden_size
        self.output_size=output_size
        self.num_layers=num_layers

        self.lstm=nn.LSTM(input_size=self.input_size,hidden_size=self.hidden_size,num_layers=self.num_layers,batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self,x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_().to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_().to(device)
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])

        return out

#training
'''
#total_batch = len(train_loader)
#avg_cost = 0
def training():
    print("Training start")

    for epoch in range(num_epochs+1):
        for accent,label in train_loader:
            optimizer.zero_grad()

            accent=accent.to(device)
            label=label.to(device)

            hypothesis = lstm_model(accent).to(device)
            cost = criterion(hypothesis, torch.max(label, 1)[1])
            #cost = criterion(hypothesis, label)

            cost.backward()
            optimizer.step()

            #avg_cost += cost / total_batch
        if epoch % 10 ==0:
            print('Epoch {:4d}/{} Cost: {:.6f}'.format(epoch, num_epochs, cost.item()))
'''
#training
def training():
    print("Training start")

    for epoch in range(num_epochs+1):
        for train_accent, train_label in train_loader:
            optimizer.zero_grad()

            train_accent=train_accent.to(device)
            train_label=train_label.to(device)

            hypothesis = lstm_model(train_accent).to(device)
            #cost = criterion(hypothesis, torch.max(train_label, 1)[1])
            cost = criterion(hypothesis, train_label)

            cost.backward()
            optimizer.step()

            #avg_cost += cost / total_batch
        if epoch % 50 ==0:
            print('Epoch: {:4d}/{} Cost: {:.6f}'.format(epoch, num_epochs, cost.item()))

#testing
def testing():
    correct = 0
    total = 0

    with torch.no_grad():
        for test_data in test_loader:
            test_accent, test_label = test_data
            #print(test_accent)
            #print(test_label)
            prediction = lstm_model(test_accent)
            #print(prediction)
            _, predicted = torch.max(prediction.data, 1)
            total += test_label.size(0)
            correct += (predicted == test_label).sum().item()

    print('Accuracy of the model on the testset: %d %%' % (100 * correct / total))

if __name__ == '__main__':
    
    #device setting
    if device == 'cuda':
        torch.cuda.manual_seed_all(777)
    print(device)
    
    #dataloader
    train_loader, test_loader=data_loader()
    
    #model
    lstm_model=LSTM_model(13,100,3,2).to(device)
    
    #loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(lstm_model.parameters(), lr=learning_rate)
   
    training()
    testing()