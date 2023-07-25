
from torch.utils.data import Dataset
import torch


class Radardata3(Dataset):
    def __init__(self, data, r, d, rangevec, dopvec, BBox=None):
        self.data = data
        self.r = r
        self.d = d
        self.BBox = BBox


        if BBox is not None:
            self.r_crop = range(BBox[0],BBox[2])
            self.d_crop = range(BBox[1],BBox[3])


        self.range_vec = rangevec
        self.doppler_vec = dopvec
        self.datasize = r.size()

        (_ ,self.Nr, self.Nd) = data[0].shape # Nr is the amount of range bins, Nd is amount of doppler bins


    def __getitem__(self, index,flat_label=True):
        data = self.data[index]
        r_val = self.r[index]
        d_val = self.d[index]
        r, d = self.val2idx(r_val,d_val)


        if self.BBox is None:
            label = int(r * self.Nd + d)  # Index of the one-hot element
        else:
            data = data[:, self.BBox[0]:self.BBox[2], self.BBox[1]:self.BBox[3]]
            label = int(r * len(self.d_crop) + d)
        return data, label

    def insertBBox(self,BBox):
        self.BBox = BBox
        self.r_crop = range(BBox[0], BBox[2])
        self.d_crop = range(BBox[1], BBox[3])

    def val2idx(self,r_val,d_val):
        if self.BBox is None:
            range_vec = self.range_vec
            doppler_vec = self.doppler_vec
        else:
            range_vec = self.range_vec[self.r_crop]
            doppler_vec = self.doppler_vec[self.d_crop]

        r = torch.argmin(torch.abs(range_vec-r_val))
        d = torch.argmin(torch.abs(doppler_vec-d_val))
        return r, d


    def __len__(self):
        return len(self.data)





class TrackData(Dataset):
    def __init__(self, data, r, d, rangevec, dopvec,T):
        self.data = data
        self.r = r
        self.d = d
        self.T = T
        self.Nd = len(dopvec)
        self.range_vec = rangevec
        self.doppler_vec = dopvec
        self.datasize = r.size()
        self.num_tracks = self.datasize[0]
        self.num_samples = self.datasize[1]

    def __getitem__(self, index):
        data = self.data[index]
        r_val = self.r[index]
        d_val = self.d[index]

        label = list()
        for i in range(self.num_samples):
            r, d = self.val2idx(r_val[i],d_val[i])
            label.append((r,d))
        return data, label

    def __len__(self):
        return len(self.data)

    def val2idx(self,r_val,d_val):
        r = torch.argmin(torch.abs(self.range_vec-r_val))
        d = torch.argmin(torch.abs(self.doppler_vec-d_val))
        return int(r), int(d)
