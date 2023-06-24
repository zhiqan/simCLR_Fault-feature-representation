import math
import os
import numpy as np
#import cv2
#from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
#import torchvision.transforms as transforms
from torchvision import datasets

#import os
import torch
#from torch.utils.data import Dataset, DataLoader
#from torchvision.transforms import transforms
#import numpy as np
#import collections
#from PIL import Image
import csv
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    
class CWRUDataset(Dataset):
    def __init__(self, root, mode, resize=84, simclr=False):
        self.simclr = simclr
        self.path = os.path.join(root, 'xichu')  # image path
        csvdata = self.loadCSV(os.path.join(root, mode + '.csv'))  # csv path
        self.data = []
        self.img2label = {}
        for i, (k, v) in enumerate(csvdata.items()):
            self.data.append(v)  # [[v1, v2, ...], [v111, ...]]
            self.img2label[k] = i  # {"v_name[:9]":label}
    def RCNN(self, X_n):  
        N, C, W = X_n.size()
        p = np.random.rand()
        K = [1, 3, 5, 7, 11, 15,17]
        if p > 0.5:
            k = K[np.random.randint(0, len(K))]
            Conv = torch.nn.Conv1d(1, 1, kernel_size=k, stride=1, padding=k//2, bias=False)
            torch.nn.init.xavier_normal_(Conv.weight)
            X_n = Conv(X_n.reshape(-1, C, W)).reshape(N,C,  W)
        return X_n.reshape(C, W).detach()
    
    def add_laplace_noise(self,x, u=0, b=0.2):
        laplace_noise = np.random.laplace(u, b, len(x)).reshape(1024, 1) # 为原始数据添加μ为0，b为0.1的噪声
        return laplace_noise + x
    
    
    def add_wgn(self,x,snr=15):
        snr = 10 ** (snr / 10.0)
        xpower = np.sum(x ** 2) / len(x)
        npower = xpower / snr
        return x + (np.random.randn(len(x)) * np.sqrt(npower)).reshape(1024, 1)
    
    
    def Amplitude_scale(self,x,snr=0.05):
        return x *(1-snr)
    
    
    def Translation(self,x,p=0.5):
        a=len(x)
        return np.concatenate((x[int(a*p):],x[0:int(a*p)]),axis=0)
            
    def loadCSV(self, csvf):
        dictLabels = {}
        with open(csvf) as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            next(csvreader, None)  # skip (filename, label)
            for i, row in enumerate(csvreader):
                filename = row[0]
                label = row[1]
                # append filename to current label
                if label in dictLabels.keys():
                    dictLabels[label].append(filename)
                else:
                    dictLabels[label] = [filename]
        return dictLabels

    def __getitem__(self, index):
        label_ = index // 380
        index_ = index % 380
        pic = pd.read_csv(os.path.join(self.path, self.data[label_][index_]),header=None)
        pic = pic.values
        
        if self.simclr:
            ccc=['self.Amplitude_scale(pic)','self.Translation(pic)','self.add_wgn(pic)','self.add_laplace_noise(pic)']
            n1 = np.random.choice(ccc,2,replace=False)
            a=eval(n1[0]).T
            b=eval(n1[1]).T
            #pic1=torch.tensor(pic,dtype=torch.float).unsqueeze(0)
            return torch.tensor(a,dtype=torch.float).detach(), torch.tensor(b,dtype=torch.float).detach()
            #pic3=torch.tensor(pic.T,dtype=torch.float)
            #return pic3,pic3.clone()
        else:
            pic3=torch.tensor(pic.T,dtype=torch.float)
            return pic3
        
    

    def __len__(self):
        # as we have built up to batchsz of sets, you can sample some small batch size of sets.
        return len(self.data) * 380