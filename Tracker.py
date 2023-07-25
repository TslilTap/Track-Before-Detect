import numpy as np
import torch

from Tracker_parts import *
from Hough_Transform import HoughTransform


class Tracker:
    def __init__(self,
                 trans_dist=None):
        if trans_dist is None:
            trans_dist = TransDist(sigma_a=30)
        self.trans_dist = trans_dist
        self.Nr = trans_dist.Nr
        self.Nv = trans_dist.Nv

        self.hough = HoughTransform()

    def Find_Accuracy(self,track,label,name=None):
        acc = 0
        track_len = len(label)
        spm = 0
        for k in range(track_len):
            r = label[k][0][0].item()
            v = label[k][1][0].item()
            if (r,v) == track[k]:
                acc += 1
            r_dist = abs(r-track[k][0])
            v_dist = abs(v-track[k][1])
            if max(r_dist,v_dist) == 1:
                spm += 1
        acc /= track_len
        spm /= track_len
        if name is not None:
            print(f'{name} accuracy =  {acc*100:.2f}%')
            print(f'{name} soft accuracy =  {(spm + acc) * 100:.2f}%')

        return acc, acc + spm

    def SingleFrameDetection(self,emis):
        track_len = emis.size(0)
        SFD = torch.argmax(emis.view(track_len,-1),dim=1)
        track = list()
        for t in range(track_len):
            r = SFD[t].item() // self.Nv
            v = SFD[t].item() % self.Nv
            track.append((r,v))
        return track

    def HoughTacker(self,emis):
        input_np = np.array(emis)
        accumulator = self.hough.transform(input_np,-15)
        r_scale = 15
        v_scale = np.abs(np.linspace(-370, 370, 64)[1] - np.linspace(-370, 370, 64)[0])
        r,v = self.hough.inverse_transform(accumulator, r_scale, v_scale)

        track = list()
        for i in range(emis.shape[0]):
            track.append((r, v))
            r_val,v_val = self.trans_dist.next(r,v,back=False)
            r,v = self.trans_dist.val2idx(r_val,v_val)
        return track

    def forward_viterb(self,observation,best_tracks=1):
        emis = observation.cpu().detach()
        with torch.no_grad():
            emis_0 = emis[0,:,:]
            self.initialize(emis_0)

            for i in range(len(emis) - 1):
                emis_k = emis[i+1,:,:]
                self.step_forward(emis_k)

            if best_tracks > 1:
                mask_best = get_top_n_mask(self.cost, self.mask, best_tracks)
                tracks = list()
                costs = []
                for r in range(self.Nr):
                    for v in range(self.Nv):
                        if mask_best[r,v]:
                            track = self.tracks[r][v]
                            cost = track.cost
                            tracks.append(track)
                            costs.append(cost)
                index = np.sort(costs)
                tracks = tracks[index]
                return tracks
            else:
                r,v = masked_argmax(self.cost, self.mask)
                track = self.tracks[r][v]
                if track is None:
                    print("track failed")
                    estim = list()
                    for i in range(emis.size(0)):
                        estim.append((0, 0))

                else:
                    estim = track.track
                return estim

    def backward_viterbi(self,observation,best_tracks=1,thresh= None,num_tracks=30):
        self.thresh = thresh
        self.num_tracks = num_tracks
        emis = observation.cpu().detach()
        with torch.no_grad():
            emis_0 = emis[0,:,:]

            self.initialize(emis_0)

            for i in range(len(emis) - 1):
                emis_k = emis[i+1,:,:]
                self.step_back(emis_k)

            if best_tracks > 1:
                mask_best = get_top_n_mask(self.cost, self.mask, best_tracks)
                tracks = list()
                costs = []
                for r in range(self.Nr):
                    for v in range(self.Nv):
                        if mask_best[r,v]:
                            track = self.tracks[r][v]
                            cost = track.cost
                            tracks.append(track)
                            costs.append(cost)
                index = np.sort(costs)
                tracks = tracks[index]
                return tracks
            else:
                r,v = masked_argmax(self.cost, self.mask)
                track = self.tracks[r][v]
                if track is None:
                    print("track failed")
                    estim = list()
                    for i in range(emis.size(0)):
                        estim.append((0, 0))

                else:
                    estim = track.track
                return estim

    def ParticleFilter(self,observation,best_tracks=1,thresh=-10,num_particles=30):
        self.thresh = thresh
        emis = observation.cpu().detach()
        with torch.no_grad():
            emis_0 = emis[0,:,:]
            self.initialize(emis_0)

            for i in range(len(emis) - 1):
                emis_k = emis[i+1,:,:]
                self.step_particle(emis_k)

            return 1


    def initialize(self, emis_0):
        cost = torch.full((self.Nr, self.Nv), float('-inf'), dtype=emis_0.dtype)
        tracks = [[None for _ in range(self.Nv)] for _ in range(self.Nr)]


        if self.thresh is None:
            val_max = torch.max(emis_0)
            val_min = torch.min(emis_0)
            thresh = val_min + (val_max-val_min)*0.5
        else:
            thresh = self.thresh
        # consider only pixels with values bigger than the threshold
        mask = (emis_0 > thresh)
        mask[self.Nr-2:self.Nr,:] = False


        while sum(sum(mask)) < 1:
            print("no candidates, lowering threshold")
            self.thresh -= 5
            mask = (emis_0 > self.thresh)
            mask[self.Nr - 2:self.Nr, :] = False

        # update the cost matrix, the inital cost is that of the initial emission


        cost[mask] = emis_0[mask]
        mask = get_top_n_mask(cost, mask, self.num_tracks)
        for r in range(self.Nr):
            for v in range(self.Nv):
                if mask[r,v]:
                    # the tracks will hold the range and velocity pixels as well as the value of the emission
                    track = Track(r,v,emis_0[r,v])
                    tracks[r][v] = track

        # initalize an empty track of a target that is off the grid

        self.cost = cost
        self.mask = mask
        self.tracks = tracks
        self.empty_track = Track(-1,-1,thresh)


    def step_back(self,emis_k):
        # empty tracks structure
        tracks = [[None for _ in range(self.Nv)] for _ in range(self.Nr)]
        # save the current cost
        cost_curr = torch.full((self.Nr, self.Nv), float('-inf'), dtype=self.cost.dtype)


        if self.thresh is None:
            val_max = torch.max(emis_k).item()
            val_min = torch.min(emis_k).item()
            thresh = val_min + (val_max-val_min)*0.5
        else:
            thresh = self.thresh
        # create a new mask of all the candidates
        mask_curr = (emis_k > thresh)
        mask_curr[self.Nr-2:self.Nr,:] = False

        while sum(sum(mask_curr)) == 0:
            thresh -= 5
            mask_curr = (emis_k > self.thresh)
            mask_curr[self.Nr - 2:self.Nr, :] = False


        for r in range(self.Nr):
            for v in range(self.Nv):
                if mask_curr[r,v]:
                    # find the most likely previous

                    r_prev, v_prev, log_prob = self.trans_dist.forward(r, v, self.cost, self.mask,back=True)

                    if r_prev == -1:
                        track = Track(-1,-1,log_prob,self.empty_track)
                    else:
                        track = self.tracks[r_prev][v_prev]
                    cost = log_prob + emis_k[r,v]
                    tracks[r][v] = Track(r, v, cost, track)
                    cost_curr[r,v] = cost

        self.cost = cost_curr
        self.mask = get_top_n_mask(self.cost, mask_curr, self.num_tracks)
        self.empty_track = Track(-1, -1, thresh, self.empty_track)
        self.tracks = tracks



    def step_particle(self,emis_k,particles):
        # normalize weights

        weights = torch.zeros([len(particles),1])
        r_list = torch.zeros([len(particles),1])
        v_list = torch.zeros([len(particles),1])

        for i in range(len(particles)):
            particle = particles[i]
            r = particle.r
            v = particle.v
            w = particle.w
            track = particle.track



