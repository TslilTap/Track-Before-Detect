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


    def __call__(self, observation ,type='viterbi', **kwargs):
        if type == 'viterbi':
            if 'beta' in kwargs:
                beta = kwargs['beta']
            else:
                beta = 0.5
            if 'num_tracks' in kwargs:
                num_tracks = kwargs['num_tracks']
            else:
                num_tracks = 30
            if 'cheat_state' in kwargs:
                cheat_state = kwargs['cheat_state']
            else:
                cheat_state = None
            return self.backward_viterbi(observation,beta=beta,num_tracks=num_tracks,cheat_state=cheat_state)
        elif type == 'pf':
            if 'beta' in kwargs:
                beta = kwargs['beta']
            else:
                beta = 0.5
            if 'num_particles' in kwargs:
                num_particles = kwargs['num_particles']
            else:
                num_particles = 100
            if 'cheat_state' in kwargs:
                cheat_state = kwargs['cheat_state']
            else:
                cheat_state = None
            return self.ParticleFilter(observation,beta=beta,num_particles=num_particles,cheat_state=cheat_state)
        elif type == 'sfd':
            return self.SingleFrameDetection(observation)
        elif type == 'ht':
            return self.HoughTacker(observation)
        else:
            raise ValueError("Unsupported type: {}".format(type))


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





    def Viterbi(self,observation,beta=0.5,cheat_state=None,bbox_drop = False):
        if beta < 0 or beta > 1:
            raise ValueError("beta must be a float value between 0 and 1")
        self.bbox_wanted = bbox_drop
        self.bbox_possible = False
        emis = observation.clone().detach()
        self.beta = beta
        with torch.no_grad():
            for i in range(len(emis)):
                emis_k = emis[i,:,:]
                if i == 0:
                    # initialize first step
                    self.step_viterbi(emis_k,init=True,cheat_state=cheat_state)
                else:
                    # step processing
                    if self.bbox_possible:
                        # if a bbox drop is possible, drop (possible only if wanted)
                        emis_k = self.bbox_drop(emis_k)
                    self.step_viterbi(emis_k)
                if self.bbox_wanted:
                    # if a bbox drop is wanted but not yet possible, check again if possible
                    if not self.bbox_possible:
                        self.bbox_possible = self.check_bbox_possible()



            N = len(self.tracks)
            costs = torch.zeros([N])
            for i in range(N):
                costs[i] = self.tracks[i].cost
            best = torch.argmax(costs).item()
            track_best = self.tracks[best]
            return track_best.track

    def step_viterbi(self,emis_k,init=False,cheat_state=None):
        tracks = list()
        if cheat_state is not None:
            # in this scenario the first state is known
            r = cheat_state[0]
            v = cheat_state[1]
            track = Track(r=r,v=v,cost=0)
            tracks.append(track)
            mask0 = torch.zeros([self.Nr,self.Nv])
            mask0[r,v] = 1
            if self.bbox_wanted:
                self.bbox_possible = True
        else:
            mask_curr = beam_mask(emis_k,self.beta)
            for r in range(self.Nr):
                for v in range(self.Nv):
                    if mask_curr[r,v]:
                        # this state is a candidate
                        if init:
                            track = Track(r=r,v=v,cost=emis_k[r,v])
                            tracks.append(track)
                        else:
                            prev_idx, log_prob = self.trans_dist.backward_step(r,v,self.tracks)
                            track = Track(r=r,v=v,cost=log_prob+emis_k[r,v],prev=self.tracks[prev_idx])
                            tracks.append(track)
        self.tracks = tracks

    def bbox_drop(self,emis_k,type = 'ma'):
        if type == 'ma':
            r_center, v_center = self.moving_center()
        else:
            raise ValueError('needs more bbox drop types')
        bbox_mask = self.get_bbox(r_center,v_center)
        val_min = torch.min(emis_k).item()
        emis_k[~bbox_mask] = val_min
        return emis_k


    def check_bbox_possible(self):
        check = True
        r_true,v_true = self.tracks[0].track[0]
        for i in range(len(self.tracks)):
            r, v = self.tracks[i].track[0]
            if r != r_true or v != v_true:
                check = False
        self.bbox_possible = check

    def moving_center(self):
        N = len(self.tracks)
        weights = torch.zeros([N])
        ranges = torch.zeros([N])
        velocities = torch.zeros([N])
        for i in range(N):
            track = self.tracks[i]
            r,v = self.trans_dist.ind2val(track.r,track.v)
            ranges[i] = r
            velocities[i] = v
            weights[i] = track.cost
        weights = torch.softmax(weights,dim=0)
        r_avg = torch.sum(ranges*weights,dim=0)
        v_avg = torch.sum(velocities*weights,dim=0)
        r_next,v_next = self.trans_dist.next(r_avg,v_avg)
        r_idx,v_idx = self.trans_dist.val2idx(r_next,v_next)
        return r_idx,v_idx


    def get_bbox(self,r_idx,v_idx):
        r_lim = 12
        v_lim = 12

        r = max(r_lim,min(self.Nr-r_lim,r_idx))
        v = max(v_lim,min(self.Nv-v_lim,v_idx))

        r_min = r - r_lim
        r_max = r + r_lim
        v_min = v - v_lim
        v_max = v + v_lim

        mask = torch.zeros([self.Nr,self.Nv])
        mask[r_min:r_max,v_min:v_max] = 1

        mask0 = (mask == 1)
        return mask0

    def ParticleFilter(self,observation,cheat_state = None,num_particles=100,beta=0.5):
        self.num_particles = num_particles
        self.beta = beta
        emis = observation.cpu().detach()
        with torch.no_grad():
            for i in range(len(emis)):
                emis_k = emis[i,:,:]
                if i == 0:
                    self.step_particle(emis_k,init=True,cheat_state=cheat_state)
                else:
                    self.step_particle(emis_k)
            best_part = torch.argmax(self.part_w)
            track = self.tracks[best_part]
            return track.track




    def step_particle(self,emis_k,cheat_state = None,init=False):
        # processing previous information
        if init or cheat_state is not None:
            # initiale case. consider only current information
            # consider the N most likely states
            if cheat_state is None:
                mask = beam_mask(emis_k,self.beta)
                self.part_r = torch.zeros([sum(sum(mask))])
                self.part_v = torch.zeros([sum(sum(mask))])
                self.part_w = torch.zeros([sum(sum(mask))])
                i = 0
                for r in range(self.Nr):
                    for v in range(self.Nv):
                        if mask[r, v]:
                            self.part_r[i] = r
                            self.part_v[i] = v
                            self.part_w[i] = emis_k[r,v]
                            i += 1
                part_lambda = torch.softmax(self.part_w,dim=0)
            else:
                r = int(cheat_state[0])
                v = int(cheat_state[1])
                self.part_r = torch.zeros([1])
                self.part_v = torch.zeros([1])
                part_lambda = torch.zeros([1])

                self.part_r[0] = r
                self.part_v[0] = v
                part_lambda[0] = 1

        else:
            part_b = torch.zeros([self.num_particles])
            part_lambda = torch.zeros([self.num_particles])

            for p in range(self.num_particles):
                r = self.part_r[p]
                v = self.part_v[p]
                w = self.part_w[p]
                r_next, v_next = self.trans_dist.random_next(r,v)
                r_idx, v_idx = self.trans_dist.val2idx(r_next,v_next)
                b = emis_k[r_idx,v_idx] # μ ~ P(x[k+1]|x[k])
                part_b[p] = b
                part_lambda[p] = torch.exp(b)*w # λ = b * w
            part_lambda = part_lambda / torch.sum(part_lambda)

        dist_A = dist.categorical.Categorical(part_lambda)

        A = dist_A.sample(torch.tensor([self.num_particles]))


        part_r = torch.zeros([self.num_particles])
        part_v = torch.zeros([self.num_particles])
        part_w = torch.zeros([self.num_particles])
        tracks = list()

        for p in range(self.num_particles):
            a = A[p]
            if init or cheat_state is not None:
                r_idx = int(self.part_r[a])
                v_idx = int(self.part_v[a])
                w = part_w[a]
                r = self.trans_dist.range_vec[r_idx]
                v = self.trans_dist.vr_vec[v_idx]
                track_prev = None
            else:
                r_prev = int(self.part_r[a])
                v_prev = int(self.part_v[a])
                track_prev = self.tracks[a]
                r, v = self.trans_dist.random_next(r_prev, v_prev)
                r_idx, v_idx = self.trans_dist.val2idx(r,v)
                b = emis_k[r_idx,v_idx]
                w = b - part_b[a]

            part_r[p] = r
            part_v[p] = v
            tracks.append(Track(r_idx,v_idx,w,track_prev))
            part_w[p] = w
        part_w = torch.softmax(part_w,dim=0)
        self.part_r = part_r
        self.part_v = part_v
        self.part_w = part_w
        self.tracks = tracks

    # Thrash

    def backward_viterbi(self,observation,beta = 0.5,num_tracks=30,cheat_state=None,type='beam'):
        self.type = type
        self.beta = beta
        self.num_tracks = num_tracks
        emis = observation.cpu().detach()

        rim_back3 = list()
        rim_movingav = list()
        with torch.no_grad():
            for i in range(len(emis)):
                emis_k = emis[i,:,:]
                'From this line onward is an experiment'
                if i > 2:
                    r, v = masked_argmax(self.cost_prev, self.mask_prev)
                    track = self.tracks[r][v]
                    estim_state = track.track[i-3]
                    rim_back3.append(estim_state)
                    r_idx,v_idx = self.moving_average_center()
                    rim_movingav.append((r_idx,v_idx))

                if i == 0:
                    self.step_viterbi(emis_k,init=True,cheat_state=cheat_state)
                else:
                    self.step_viterbi(emis_k)

            r,v = masked_argmax(self.cost_prev, self.mask_prev)
            track = self.tracks[r][v]
            estim = track.track
            return estim, rim_back3, rim_movingav


    def step_viterbi2(self,emis_k,cheat_state=None,init=False):

        tracks = [[None for _ in range(self.Nv)] for _ in range(self.Nr)]
        cost_curr = torch.full((self.Nr, self.Nv), float('-inf'), dtype=emis_k.dtype)

        if cheat_state is None:
            # in this case the first state is not given. Therefore the inital weights are the log likelihood of the first state

            mask_curr = beam_mask(emis_k,self.beta)


            if init:
                cost_curr[mask_curr] = emis_k[mask_curr]

            for r in range(self.Nr):
                for v in range(self.Nv):
                    if mask_curr[r,v]:
                        if init:
                            track = Track(r, v, emis_k[r,v])
                            tracks[r][v] = track
                        else:
                            # find the most likely previous
                            r_prev, v_prev, log_prob = self.trans_dist.forward(r, v, emis=self.cost_prev, mask=self.mask_prev)
                            track = self.tracks[r_prev][v_prev]

                            cost = log_prob + emis_k[r,v]
                            tracks[r][v] = Track(r, v, cost, track)
                            cost_curr[r,v] = cost
            self.mask_prev = get_top_n_mask(cost_curr, mask_curr, self.num_tracks)

        else:
            r = int(cheat_state[0])
            v = int(cheat_state[1])
            cost_curr[r,v] = 0
            tracks[r][v] = Track(r, v, 0)
            self.mask_prev = (cost_curr == 0)


        self.cost_prev = cost_curr
        self.tracks = tracks

    def moving_average_center(self):
        N = sum(sum(self.mask_prev))
        weights = torch.zeros([N,1])
        ranges = torch.zeros([N,1])
        velocities = torch.zeros([N,1])

        i = 0
        for r in range(self.Nr):
            for v in range(self.Nv):
                if self.mask_prev[r,v]:
                    ranges[i] = self.trans_dist.range_vec[r]
                    velocities[i] = self.trans_dist.vr_vec[v]
                    weights[i] = self.cost_prev[r,v]
                    i += 1
        weights = torch.softmax(weights,dim=0)
        r_avg = torch.sum(ranges*weights,dim=0)
        v_avg = torch.sum(velocities*weights,dim=0)

        r_next = r_avg + self.trans_dist.T * v_avg
        v_next = v_avg

        r_idx, v_idx = self.trans_dist.val2idx(r_next,v_next)
        return r_idx,v_idx