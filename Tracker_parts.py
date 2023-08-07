import torch
import torch.distributions as dist

class TransDist:
    def __init__(self, r_min=0,r_max=2985,Nr=200,vr_min=-369.1406,vr_max=369.1406,Nv=64,T=1,sigma_r=0.00001,sigma_v=0.00001,sigma_a=0.00001,RW = False):
        self.range_vec = torch.linspace(r_min,r_max,Nr)
        self.vr_vec = torch.linspace(vr_min,vr_max,Nv)
        self.r_bin = float((self.range_vec[1]-self.range_vec[0])/2)
        self.v_bin = float((self.vr_vec[1]-self.vr_vec[0])/2)
        self.T = T
        self.Nr = Nr
        self.Nv = Nv
        self.RW = RW
        self.prob = torch.zeros([self.Nr,self.Nv])





        if RW:
            self.sigma_a = sigma_a
            dist_a = dist.Normal(0, sigma_a)
            r_const = (T**2)/2
            for r in range(self.Nr):
                for v in range(self.Nv):
                    r_max = (2 * r + 1) * self.r_bin/r_const
                    r_min = (2 * r - 1) * self.r_bin/r_const
                    v_max = (2 * v + 1) * self.v_bin/T
                    v_min = (2 * v - 1) * self.v_bin/T

                    a_max = torch.min(torch.tensor([r_max,v_max]))
                    a_min = torch.max(torch.tensor([r_min,v_min]))
                    if a_min>a_max:
                        self.prob[r,v] = -torch.inf
                    else:
                        self.prob[r,v] = torch.log(dist_a.cdf(a_max) - dist_a.cdf(a_min))
            self.dist_a = dist_a
        else:
            sigma_r = torch.sqrt(torch.tensor((sigma_a*(T**2)/2)**2+sigma_r**2))
            sigma_v = torch.sqrt(torch.tensor((sigma_a*T)**2+sigma_v**2))
            dist_r = dist.Normal(0, sigma_r)
            dist_v = dist.Normal(0, sigma_v)

            self.dist_r = dist_r
            self.dist_v = dist_v

            for r in range(self.Nr):
                r_max = self.range_vec[r] + self.r_bin
                r_min = self.range_vec[r] - self.r_bin
                prob = (dist_r.cdf(r_max) - dist_r.cdf(r_min))
                self.prob[r,:] = torch.log(prob+0.00001)

            for v in range(self.Nv):
                v_max = (2*v+1)*self.v_bin
                v_min = (2*v-1)*self.v_bin
                prob = (dist_v.cdf(torch.tensor(v_max)) - dist_v.cdf(torch.tensor(v_min)))
                self.prob[:, v] += torch.log(prob+0.00001)

    def ind2val(self,r,v):
        r_val = self.range_vec[r]
        v_val = self.vr_vec[v]
        return r_val,v_val
    def next(self,r,v):
        r_next = r + v*self.T
        v_next = v
        return r_next,v_next

    def backward_step(self,r,v,tracks):
        N = len(tracks)
        costs = torch.zeros([N])
        r_prev, v_prev = self.prev(r,v)
        log_prob = torch.zeros([N])
        for i in range(N):
            track = tracks[i]
            r_diff = abs(r_prev-track.r)
            v_diff = abs(v_prev-track.v)
            costs[i] = track.cost + self.prob[r_diff,v_diff]
            log_prob[i] = self.prob[r_diff,v_diff]
        prev_idx = torch.argmax(costs).item()

        return prev_idx, log_prob[prev_idx]

    def prev(self,r,v):
        v_prev = self.vr_vec[v]
        r_prev = self.range_vec[r] - v_prev * self.T
        r_idx, v_idx = self.val2idx(r_prev, v_prev)
        return r_idx, v_idx


    def random_next(self,r,v,RW=False):
        if RW:
            a = self.dist_a.rsample()
            r_next = r + v * self.T + (self.T**2)*a/2
            v_next = v + self.T*a

        else:
            dr = self.dist_r.rsample()
            dv = self.dist_v.rsample()
            r_next = r + v * self.T + dr
            v_next = v + dv
        return r_next, v_next

    def val2idx(self,r_val,v_val):
        r = torch.argmin(torch.abs(self.range_vec-r_val))
        v = torch.argmin(torch.abs(self.vr_vec-v_val))
        return int(r), int(v)


    # thrash
    def forward(self, r,v,emis,mask=None):
        r_prev, v_prev = self.prev(r,v)
        if mask is None:
            mask = (emis > float('-inf'))

        log_prob, mask0 = self.compute_log_prob(r_prev,v_prev,mask=mask)

        if sum(sum(mask0)) == 0:
            mask1 = (mask==True)
            log_prob[mask1] = emis[mask1] - 100
        else:
            mask1 = mask0
            log_prob[mask1] += emis[mask1]
        r, v = masked_argmax(log_prob,mask1)
        log_prob_best = log_prob[r,v]
        return r,v,log_prob_best


    def compute_log_prob(self,r_idx,v_idx,mask):
        mask0 = (mask == True)
        result = torch.full([self.Nr,self.Nv], float('-inf'))
        for r in range(self.Nr):
            for v in range(self.Nv):
                if mask[r,v]:
                    r_diff = int(abs(r_idx-r))
                    v_diff = int(abs(v_idx-v))
                    prob = self.prob[r_diff,v_diff]

                    if prob == float('-inf'):
                        mask0[r,v] = False
                    else:
                        result[r,v] = prob
        return result, mask0

class Track:
    def __init__(self,r,v,cost,prev=None):
        self.r = r
        self.v = v
        if prev is None:
            prev_cost = 0
            self.track = list()
        else:
            prev_cost = prev.cost
            self.track = prev.track.copy()
        self.cost = prev_cost + cost
        self.track.append((r,v))


def beam_mask(emis_k,beta=0.5):
    val_max = torch.max(emis_k)
    val_min = torch.min(emis_k)
    thresh = val_min + ((val_max - val_min) * beta)
    mask = (emis_k >= thresh)
    return mask

def masked_argmax(original_tensor, mask):
    Nv = original_tensor.size(1)
    Nr = original_tensor.size(0)
    val = float('-inf')
    r_best = -1
    v_best = -1
    for r in range(Nr):
        for v in range(Nv):
            if mask[r,v]:
                if original_tensor[r,v] >= val:
                    val = original_tensor[r,v]
                    r_best = int(r)
                    v_best = int(v)
    return r_best, v_best

def get_top_n_mask(tensor,mask=None,N=100):
    # Check if N is None or greater than the number of True values in the mask

    if mask is None:
        # Apply the mask to the tensor
        masked_tensor = tensor[mask]
    else:
        masked_tensor = tensor
        if N is None or N > sum(sum(mask)):
            return mask  # Return the original mask

    # Find the indices of the N highest values in the masked tensor
    top_indices = torch.topk(masked_tensor, k=N)[1]

    # Create a new mask with the N highest values
    new_mask = torch.zeros_like(mask)
    new_mask[mask] = torch.isin(masked_tensor, masked_tensor[top_indices])

    return new_mask

def Find_Accuracy(track,label,name=None):
    acc = 0
    track_len = len(label)
    spm = 0
    for k in range(track_len):
        r = label[k][0][0].item()
        v = label[k][1][0].item()
        if (r,v) == track[k]:
            acc += 1
        dr = abs(r-track[k][0])
        dv = abs(v-track[k][1])
        if max(dr,dv) == 1:
            spm += 1
    acc /= track_len
    spm /= track_len
    if name is not None:
        print(f'{name} accuracy =  {acc*100:.2f}%')
        print(f'{name} soft accuracy =  {(spm + acc) * 100:.2f}%')
    return acc, acc + spm