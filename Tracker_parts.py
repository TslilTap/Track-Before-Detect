import torch
import torch.distributions as dist

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

class Particle:
    def __init__(self,r,v,w,track):
        self.r = r
        self.v = v
        self.w = w
        self.track = track


def normalize_log(vec):
    vec_lin = torch.exp(vec)
    vec_norm = vec_lin/torch.sum(vec_lin)
    return vec_norm



class TransDist:
    def __init__(self, r_min=0,r_max=2985,Nr=200,vr_min=-369.1406,vr_max=369.1406,Nv=64,T=1,sigma_r=0.00001,sigma_v=0.00001,sigma_a=0.00001,RW = False):
        self.range_vec = torch.linspace(r_min,r_max,Nr)
        self.vr_vec = torch.linspace(vr_min,vr_max,Nv)
        self.r_bin = float((self.range_vec[1]-self.range_vec[0])/2)
        self.v_bin = float((self.vr_vec[1]-self.vr_vec[0])/2)
        self.T = T
        self.Nr = Nr
        self.Nv = Nv

        self.prob = torch.zeros([self.Nr,self.Nv])

        self.RW = RW

        self.sigma_a = sigma_a

        if RW:
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
                self.prob[r,:] = torch.log(prob)

            for v in range(self.Nv):
                v_max = (2*v+1)*self.v_bin
                v_min = (2*v-1)*self.v_bin
                prob = (dist_v.cdf(torch.tensor(v_max)) - dist_v.cdf(torch.tensor(v_min)))
                self.prob[:, v] += torch.log(prob)


    def forward(self, r,v,emis,mask=None,back=False):
        r_next, v_next = self.next(r,v,back)
        if mask is None:
            mask = (emis > float('-inf'))
        log_prob, mask0 = self.compute_log_prob(r_next,v_next,mask=mask)
        if sum(sum(mask0)) == 0:
            mask1 = (mask==True)
            log_prob[mask1] = emis[mask1] - 100
        else:
            mask1 = mask0
            log_prob[mask1] += emis[mask1]
        r, v = masked_argmax(log_prob,mask1)
        log_prob_best = log_prob[r,v]
        return r,v,log_prob_best

    def next(self,r,v,back,random=False):
        v_next = self.vr_vec[v]
        if back:
            r_next = self.range_vec[r] - v_next * self.T
        else:
            r_next = self.range_vec[r] + v_next * self.T
        if random:
            if self.RW:
                delta_a = self.dist_a.rsample()
                delta_r = delta_a*(self.T**2)/2
                delta_v = delta_a*self.T
            else:
                delta_r = self.dist_r.rsample()
                delta_v = self.dist_v.rsample()
            r_next += delta_r
            v_next += delta_v
        r_idx, v_idx = self.val2idx(r_next, v_next)
        return r_idx, v_idx

    def val2idx(self,r_val,v_val):
        r = torch.argmin(torch.abs(self.range_vec-r_val))
        v = torch.argmin(torch.abs(self.vr_vec-v_val))
        return int(r), int(v)



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

def get_top_n_mask(tensor, mask, N):
    # Check if N is None or greater than the number of True values in the mask
    if N is None or N > sum(sum(mask)):
        return mask  # Return the original mask

    # Apply the mask to the tensor
    masked_tensor = tensor[mask]

    # Find the indices of the N highest values in the masked tensor
    top_indices = torch.topk(masked_tensor, k=N)[1]

    # Create a new mask with the N highest values
    new_mask = torch.zeros_like(mask)
    new_mask[mask] = torch.isin(masked_tensor, masked_tensor[top_indices])

    return new_mask