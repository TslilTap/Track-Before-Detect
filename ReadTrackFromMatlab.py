import scipy.io
from RadarData import TrackData
import torch


SNR = 20

file_name = 'single_target' + str(SNR) +'SNR_10simplewalk' + '.mat'
mat = scipy.io.loadmat(file_name)

rangeVec = torch.tensor(mat['rangeVec'][0])
vrVec = torch.tensor(mat['vrVec'][0])

num_tracks = 10
num_frames = 50


range_label = torch.zeros((num_tracks,num_frames))
vr_label = torch.zeros((num_tracks,num_frames))
data = torch.zeros((num_tracks,num_frames,200,64))

BBox = [60,10,200,54]

print(rangeVec[60],rangeVec[199])
print(vrVec[10],vrVec[53])


for i in range(num_tracks):
    for j in range(num_frames):
        track = mat['tracks'][i][j][0]
        x = mat['data'][i][j]
        v = track[0]
        r = track[1]
        range_label[i,j] = r
        vr_label[i,j] = v
        x = torch.tensor(x)
        data[i,j,:,:] = x



track_data = TrackData(data,range_label,vr_label,rangeVec,vrVec,1)
torch.save(track_data,'track_data_'+ str(SNR) + 'SNR')
