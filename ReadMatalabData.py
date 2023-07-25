from RadarData import Radardata3
import torch
import scipy.io
import random


SNR = 25
data_size = 3000

file_name = 'single_target'+str(SNR)+'SNR_'+str(data_size)+'training_samplesV1' + '.mat'
mat = scipy.io.loadmat(file_name)



rangeVec = torch.tensor(mat['rangeVec'][0])
vrVec = torch.tensor(mat['vrVec'][0])

range_label = torch.zeros(data_size)
vr_label = torch.zeros(data_size)
data = torch.zeros((data_size,1,200,64))


k = 0
for i in range(data_size):
        x = mat['data'][0][i]
        label = mat['labels'][0][i][0]
        v = label[0]
        r = label[1]
        print(r,v)
        range_label[i] = r
        vr_label[i] = v
        x = torch.tensor(x)
        data[i,0,:,:] = x

valid_size = 300 # number of valid samples
train_size = data_size-valid_size # number of train samples



## don't change this:

# generate random samples indexes
indices = random.sample(range(data_size), data_size)

# split indexes for train and valid
train_idx = indices[:train_size]
valid_idx = indices[train_size:]

BBox = [60,10,200,54]

train_data = Radardata3(data[train_idx],range_label[train_idx],vr_label[train_idx],rangeVec,vrVec,BBox)
valid_data = Radardata3(data[valid_idx],range_label[valid_idx],vr_label[valid_idx],rangeVec,vrVec,BBox)

torch.save(train_data, f'train_data_{train_size}' + '_SNR25')
torch.save(valid_data, f'valid_data_{valid_size}' + '_SNR25')